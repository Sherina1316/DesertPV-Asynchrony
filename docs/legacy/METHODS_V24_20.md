# v24.22 Methods: causal reference freezing and terminal-site extension

## 1. Causal 2024 endpoint

Let \(R_{2024}\) denote the manually interpreted 2024 PV prior and
\(I_{2024}\) the audited image-adjusted inventory. The adjustment is restricted
to a narrow boundary uncertainty band. Crucially,

\[
I_{2024}=f(R_{2024},X_{2011},\ldots,X_{2024}),
\]

and \(X_{2025}\) is excluded. This prevents a true 2025 extension from being
absorbed into the 2024 endpoint.

## 2. Historical reconstruction

For each 2024 component, high-confidence historical seeds are identified from
annual PV evidence. Weak edge pixels are recovered only when connected to a
seed inside the same audited component. Installation timing is estimated by an
ordered change-point model. No component is copied to an earlier year solely
because it exists in 2024.

## 3. Terminal-year expansion

Because 2025 has no following year, three evidence pathways are combined.

### 3.1 Gain-supported adjacent expansion

Pixels must be close to the 2024 inventory and show both high 2025 PV
probability and positive change relative to the 2021–2023 baseline.

### 3.2 Remote new sites

Remote candidates require stronger probability, change, size, solidity and
bounding-box filling constraints.

### 3.3 Component-conditioned perimeter extension

For each 2024 site component, a local prototype is estimated from 2024
spectral/PV-state features. Candidate 2025 pixels in the perimeter search zone
are compared with that component. A connected object can be accepted despite
modest probability gain only when it has:

- high absolute 2025 site likelihood;
- similarity to the neighbouring component prototype;
- low sand and confuser risk;
- sufficient object area and solidity;
- spatial proximity and boundary contact;
- multicue change or a strong absolute panel-state override.

The accepted source raster is passed unchanged to Stage 8. This guarantees that
new terminal extensions receive installation year 2025 and appear as annual
installation gain.

## 4. Ecological response and 2050 projection

Annual PV states are combined with precipitation, temperature,
evapotranspiration, soil moisture, radiation, wind, terrain, land cover and
human-activity covariates. Ecological response is evaluated using panel-core,
edge and matched-background contrasts, climate residualisation, event studies,
response lags and stabilisation time. Future expansion is projected by a
strictly lagged two-part model (occurrence and positive increment) with spatial
capacity constraints and rolling-origin hindcast validation.
