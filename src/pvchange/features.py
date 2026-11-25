from __future__ import annotations

import numpy as np

EPS = 1e-6


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def compute_features(
    reflectance: np.ndarray,
    channels: list[str],
    add_valid_mask: bool = True,
    endmembers: np.ndarray | None = None,
) -> np.ndarray:
    """Compute channel-first Landsat and LSMA features.

    The confuser evidence channels are intentionally explicit. They are not
    treated as hard land-cover labels; they provide penalties for water,
    vegetation, built surfaces and spectrally persistent bare ground during the
    temporal transition stage.
    """
    if reflectance.ndim != 3 or reflectance.shape[0] < 6:
        raise ValueError("Expected reflectance with shape [6, H, W]")
    blue, green, red, nir, swir1, swir2 = [reflectance[i].astype(np.float32) for i in range(6)]
    valid = np.all(np.isfinite(reflectance[:6]), axis=0)
    ndvi = safe_ratio(nir, red)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + EPS)
    ndmi = safe_ratio(nir, swir1)
    ndbi = safe_ratio(swir1, nir)
    mndwi = safe_ratio(green, swir1)
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)
    tc_brightness = 0.3029 * blue + 0.2786 * green + 0.4733 * red + 0.5599 * nir + 0.5080 * swir1 + 0.1872 * swir2
    tc_greenness = -0.2941 * blue - 0.2430 * green - 0.5424 * red + 0.7276 * nir + 0.0713 * swir1 - 0.1608 * swir2
    tc_wetness = 0.1511 * blue + 0.1973 * green + 0.3283 * red + 0.3407 * nir - 0.7117 * swir1 - 0.4559 * swir2

    values: dict[str, np.ndarray] = {
        "BLUE": blue, "GREEN": green, "RED": red, "NIR": nir, "SWIR1": swir1, "SWIR2": swir2,
        "NDVI": ndvi, "EVI": evi, "NDMI": ndmi, "NDBI": ndbi, "MNDWI": mndwi, "BSI": bsi,
        "TC_Brightness": tc_brightness, "TC_Greenness": tc_greenness, "TC_Wetness": tc_wetness,
    }
    values["PV_Spectral_Evidence"] = (
        0.30 * np.clip(ndbi, -1, 1)
        + 0.20 * np.clip(bsi, -1, 1)
        + 0.20 * np.clip(-ndvi, -1, 1)
        + 0.15 * np.clip(-mndwi, -1, 1)
        + 0.15 * np.clip(tc_brightness, -1, 1)
    ).astype(np.float32)

    mixture_names = {
        "LSMA_HA", "LSMA_LA", "LSMA_VG", "LSMA_SH", "LSMA_RMSE",
        "PV_Fraction_Evidence", "PV_Confuser_Penalty",
    }
    fractions = None
    lsma_rmse = None
    if mixture_names.intersection(channels):
        if endmembers is None:
            raise ValueError("Unit endmembers are required for LSMA feature channels")
        from .spectral_mixture import unmix_lsma
        fractions, lsma_rmse = unmix_lsma(reflectance, np.asarray(endmembers, dtype=np.float32))
        values.update({
            "LSMA_HA": fractions[0], "LSMA_LA": fractions[1], "LSMA_VG": fractions[2],
            "LSMA_SH": fractions[3], "LSMA_RMSE": lsma_rmse,
        })
        values["PV_Fraction_Evidence"] = _clip01(0.58 * fractions[1] + 0.42 * fractions[3] - 0.25 * fractions[0])
    else:
        values["PV_Fraction_Evidence"] = _clip01(0.5 + values["PV_Spectral_Evidence"])

    # Confuser evidence is deliberately conservative. Persistent temporal
    # behaviour is still required before any pixel is rejected or retained.
    water = _clip01((mndwi - 0.05) / 0.35) * _clip01((ndmi + 0.10) / 0.50)
    vegetation = _clip01((ndvi - 0.18) / 0.45) * _clip01((tc_greenness + 0.05) / 0.35)
    built = _clip01((ndbi - 0.05) / 0.45) * _clip01((tc_brightness - 0.10) / 0.45)
    bare = _clip01((bsi - 0.05) / 0.50) * _clip01((tc_brightness - 0.08) / 0.50)
    values["Water_Evidence"] = water
    values["Vegetation_Evidence"] = vegetation
    values["Builtup_Evidence"] = built
    values["Bare_Evidence"] = bare
    # Built and bare surfaces are not automatically excluded; the penalty is
    # reduced when LSMA low-albedo/shadow evidence is strong.
    pv_fraction = values["PV_Fraction_Evidence"]
    values["PV_Confuser_Penalty"] = _clip01(
        0.35 * water + 0.25 * vegetation + 0.20 * built + 0.20 * bare - 0.35 * pv_fraction
    )

    out = []
    for name in channels:
        if name not in values:
            raise KeyError(f"Unsupported feature channel: {name}")
        arr = values[name].astype(np.float32)
        arr[~valid] = np.nan
        out.append(arr)
    if add_valid_mask:
        out.append(valid.astype(np.float32))
    return np.stack(out, axis=0)


def robust_stats(array: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim != 4:
        raise ValueError("Expected [N, C, H, W]")
    c = array.shape[1]
    med = np.zeros(c, dtype=np.float32)
    scale = np.ones(c, dtype=np.float32)
    for i in range(c):
        x = array[:, i].reshape(-1)
        if mask is not None:
            x = x[mask.reshape(-1)]
        x = x[np.isfinite(x)]
        if x.size:
            q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
            med[i] = q50
            scale[i] = max(float(q75 - q25), 1e-4)
    return med, scale


def normalize_features(x: np.ndarray, median: np.ndarray, scale: np.ndarray, clip: float = 8.0) -> np.ndarray:
    y = (x - median[:, None, None]) / scale[:, None, None]
    y = np.nan_to_num(y, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(y, -clip, clip).astype(np.float32)
