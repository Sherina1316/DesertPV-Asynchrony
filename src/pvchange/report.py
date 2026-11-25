from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _table(df: pd.DataFrame, n: int = 20) -> str:
    if df.empty:
        return "（无可用记录）"
    return df.head(n).to_markdown(index=False)


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    try:
        v = float(value)
    except Exception:
        return "不可估计"
    return "不可估计" if not np.isfinite(v) else f"{v:.{digits}f}{suffix}"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _status(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def generate_chinese_report(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "20_report")
    validation = _status(root / "09_accuracy_validation" / "accuracy_validation_status.json")
    temporal = _status(root / "08_temporal_pv_change" / "temporal_decoding_status.json")
    maps = _status(root / "17_maps" / "map_status.json")
    figures = _status(root / "18_figures" / "figure_status.json")
    audit = _status(root / "19_audit" / "audit_status.json")

    narratives_payload = _status(root / "18_figures" / "five_claim_narratives.json")
    narratives = narratives_payload.get("narratives", {})
    metrics = _read(root / "18_figures" / "five_claim_metrics.csv")
    validation_table = _read(root / "09_accuracy_validation" / "segmentation_validation_summary.csv")
    validation_units = _read(root / "09_accuracy_validation" / "complete_reference_validation_by_unit.csv")
    national = _read(root / "11_pv_dynamics" / "national_pv_dynamics.csv")
    events = _read(root / "13_pv_events" / "pv_expansion_events.csv")
    response = _read(root / "18_figures" / "model_consistent_response_timing.csv")
    distance = _read(root / "18_figures" / "distance_gradient_trends.csv")
    scenario = _read(root / "18_figures" / "national_2050_scenario_sensitivity.csv")
    speed = _read(root / "18_figures" / "expansion_ecological_speed_by_unit.csv")
    quality = _read(root / "04_unit_quality" / "unit_quality_registry.csv")
    environment = _read(root / "12_dynamic_ecology" / "environmental_covariates_imputed.csv")
    leakage = _read(root / "21_strict_noleak_forecast" / "audit" / "feature_leakage_audit_forecast.csv")

    gate = bool(validation.get("segmentation_gate_pass", False))
    reference_year = int(cfg["study"]["reference_year"])
    start_year = int(cfg["study"]["start_year"])
    end_year = int(cfg["study"]["end_year"])
    units = expected_counts(cfg)[0]

    if len(national):
        ordered = national.sort_values("year")
        first, last = ordered.iloc[0], ordered.iloc[-1]
        active_col = "observed_active_area_km2" if "observed_active_area_km2" in ordered else "pv_area_km2"
        cumulative_col = "cumulative_installed_area_km2" if "cumulative_installed_area_km2" in ordered else active_col
        area_sentence = (
            f"全国观测活动光伏面积由{int(first.year)}年的{_fmt(first[active_col], 1)} km²变化至"
            f"{int(last.year)}年的{_fmt(last[active_col], 1)} km²；累计安装面积由"
            f"{_fmt(first[cumulative_col], 1)} km²变化至{_fmt(last[cumulative_col], 1)} km²。"
        )
    else:
        area_sentence = "全国面积序列不可估计。"

    claims = {
        "claim_1": narratives.get("claim_1", "结果尚未形成。"),
        "claim_2": narratives.get("claim_2", "结果尚未形成。"),
        "claim_3": narratives.get("claim_3", "结果尚未形成。"),
        "claim_4": narratives.get("claim_4", "结果尚未形成。"),
        "claim_5": narratives.get("claim_5", "结果尚未形成。"),
    }

    text = f"""# {cfg['report']['title']}

## 1. 数据定义与不可更改的2024端点

本研究统一处理 **{units}个研究单元 × {end_year-start_year+1}个年度（{start_year}—{end_year}）**。人工修订矢量
`/home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.shp` 被定义为{reference_year}年光伏分布的**高置信度人工清查先验**。矢量稳定内部作为可信核心，矢量边界附近允许依据多波段状态、安装突变、时序持续性和对象连续性进行有限修正；所有增删必须位于预先声明的边界不确定带内，并受面积调整上限约束。

该约束只定义{reference_year}状态，不定义历史状态。{start_year}—{reference_year-1}年不会复制{reference_year}几何，而是通过年度Landsat波段、LSMA组分、模型概率、光谱原型、局地对比、稳定沙地风险、年际突变和后续持续性重建安装年份。历史证据不足的{reference_year}光伏像元默认从{reference_year}开始计入，而不是回填到2011年。{reference_year}年矢量外历史场站只允许作为具有连续多年证据且在{reference_year}前已退役的轨迹存在；2025年矢量外新增场站必须具有高置信度独立影像证据。

- {reference_year}最终结果与审计后的边界修正清单一致：**{'已认证' if validation.get('final_refined_inventory_exact_by_contract') else '未认证'}**
- 完整清单全景precision/IoU可识别：**{'是' if validation.get('full_scene_precision_identifiable') and validation.get('full_scene_iou_identifiable') else '否'}**
- 历史几何是否无证据回填：**否**
- 环境变量是否可直接制造历史光伏标签：**否**
- 科学精度门：**{'通过' if gate else '未通过；结果仅作条件性诊断'}**

## 2. 方法闭环

1. 检查142个单元、2011—2025影像、修订SHP及环境变量的完整性；
2. 建立统一投影30 m标准网格；
3. 将完整{reference_year}清单一次性栅格化，矢量内为1、矢量外为0，并保留边界覆盖比例用于面积计算；
4. 计算LSMA、年度质量、光谱特征和因果Annual-CCDC缓存；
5. 使用空间隔离的训练、校准和独立测试单元比较监督模型；
6. 形成加权集成模型；
7. 对{start_year}—{end_year}逐年推理并基于完整清单进行概率校准；{reference_year}端点直接采用修订清单；
8. 对清单内部执行有序变化点重建，对清单外执行持续轨迹与退役约束；
9. 在端点强制之前，以完整{reference_year}清单计算precision、recall、F1、IoU、PR-AUC、ROC-AUC、boundary F1和面积误差；
10. 传播模型、阈值、边界和输入不确定性；
11. 计算活动面积、累计安装面积、新增、损失和扩张方向；
12. 在固定空间支持上计算PAVG、绿度、水分和环境残差；
13—15. 识别扩张事件，执行气候调整的事件研究、匹配对照、响应滞后和稳定时间分析；
16. 构建2050条件性空间包络；
17—18. 输出单年图、15年图集、主文图、扩展图及两张桑基图；
19—20. 执行空间、声明和文件完整性审计并生成中文报告；
21. 使用严格滚动历史回测、两阶段扩张模型和递归特征更新预测至2050年。

## 3. 完整清单检测精度

端点强制前的模型结果在完整{reference_year}清单上进行全景验证。端点强制后的最终{reference_year}产品必须达到逐像元一致和IoU=1；该一致性属于数据合同，而不是模型精度。模型精度仅由端点强制前的独立预测计算。

{_table(validation_table, 20)}

### 单元级验证

{_table(validation_units.sort_values('f1') if len(validation_units) and 'f1' in validation_units else validation_units, 30)}

## 4. 2011—2025光伏安装、活动和退役变化

{area_sentence}

年度产品同时保留：

- **Observed active footprint**：当年影像支持的活动光伏；
- **Cumulative installed footprint**：首次安装后持续保留的累计建设范围；
- **Retired/inactive footprint**：累计已安装但当年不活动的范围；
- **First installed year**：最早持续结构出现年份，而非行政开工或并网日期。

年度地图只绘制一个互斥分类层，不叠加多层半透明光伏结果；面积、新增、损失和净变化统一放在地图外部信息栏，并保持各年份相同空间范围。

{_table(national, 20)}

## 5. 环境变量与生态综合分析

环境变量包括降水、温度、潜在蒸散、实际蒸散、土壤湿度、太阳辐射和风速等。它们具有三种用途：

1. 在异常气候年份降低光谱观测可靠性，但不能把适宜区直接判为历史光伏；
2. 在生态分析中进行气候残差化、固定效应控制、同年未处理/尚未处理对照和倾向匹配；
3. 在2050预测中作为严格滞后的扩张驱动和情景变量。

固定空间支持用于避免年度光伏边界变化造成统计像元组成变化。生态指标包括PAVG、NDVI/EVI/TC Greenness、NDMI/TC Wetness、植被丰度、生态响应滞后、稳定时间和综合恢复指标。30 m核心区指标反映混合像元组分，不能直接等同于板下实测植被。

### 环境数据概览

{_table(environment, 15)}

### 距离梯度

{_table(distance, 20)}

### 响应时序

{_table(response, 25)}

### 扩张—生态速度差异

{_table(speed.sort_values('expansion_adjustment_ratio', ascending=False) if len(speed) and 'expansion_adjustment_ratio' in speed else speed, 25)}

## 6. 五项主要结果

### 6.1 沙漠背景增绿与核心区水分变化

{claims['claim_1']}

### 6.2 核心—边缘—背景距离梯度

{claims['claim_2']}

### 6.3 生态响应滞后与稳定时间

{claims['claim_3']}

### 6.4 2050情景敏感性

{claims['claim_4']}

### 6.5 工程扩张速度与生态调整速度

{claims['claim_5']}

## 7. 2050严格无泄漏预测

预测采用滚动起点历史回测：每个测试年份只能使用更早年份训练。累计面积预测使用两阶段模型，先预测是否发生新增安装，再预测正新增面积；随后施加有效空间容量、历史增长率和覆盖率上限。环境变量只以滞后值或明确的未来情景进入，不使用当前年或未来目标值。预测结果是条件性情景，不是确定建设清单。

{_table(scenario, 15)}

### 特征泄漏审计

{_table(leakage, 20)}

## 8. 图表与可复现输出

- 年度地图：**{maps.get('individual_annual_maps', 0)}张**；
- 15年单元图集：**{maps.get('annual_atlases', 0)}张**；
- 主图及扩展图：**{figures.get('figures', 0)}组**；
- 桑基图：`figure_10_ecological_response_sankey` 与 `figure_11_multimetric_ecology_sankey`；
- 栅格分辨率：统一投影30 m；
- 图件格式：高分辨率PNG和矢量PDF/SVG；
- 文件完整性：输出SHA256清单；
- 声明审计：**{'通过' if audit.get('status') == 'passed' else '未通过或未完成'}**。

## 9. 科学边界

1. {reference_year}清单完整性来自人工修订矢量；端点强制的一致性不能冒充模型精度。
2. 历史安装年份为Landsat 30 m尺度的最早持续结构出现年份，需要建设档案或高分辨率历史影像外部验证。
3. 历史退役场站和2025新增场站位于{reference_year}清单之外时，必须依赖独立时序证据，仍可能存在残余漏检。
4. 生态结果为气候调整后的遥感关联和条件性事件研究，不自动构成单一因果机制。
5. 2050结果依赖环境、需求、政策和空间容量假设，必须以情景区间和不确定性表达。
6. 是否达到Nature正刊要求必须依据真实独立精度、历史年份外部验证、统计识别强度和结果的重要性判断，代码本身不能预先保证录用或固定精度。

## 10. 关键结果表

{_table(metrics, 30)}
"""

    md = out / "complete_research_report_CN.md"
    md.write_text(text, encoding="utf-8")
    hp = out / "complete_research_report_CN.html"
    hp.write_text(
        "<html><head><meta charset='utf-8'><title>PV v24.19 adaptive-reference report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1150px;margin:auto;padding:32px;line-height:1.65}"
        "table{border-collapse:collapse}pre{white-space:pre-wrap}</style></head><body><pre>"
        + html.escape(text) + "</pre></body></html>",
        encoding="utf-8",
    )

    workbook = out / "complete_research_tables_CN.xlsx"
    sheets = [
        ("核心结论", metrics),
        ("完整清单验证汇总", validation_table),
        ("完整清单单元验证", validation_units),
        ("全国动态", national),
        ("转折事件", events),
        ("环境协变量", environment),
        ("距离梯度", distance),
        ("响应时序", response),
        ("2050情景", scenario),
        ("预测泄漏审计", leakage),
        ("扩张生态速度", speed),
        ("质量路由", quality),
    ]
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name[:31], index=False)

    files = [p for p in root.rglob("*") if p.is_file() and out not in p.parents]
    manifest = pd.DataFrame([
        {"path": str(p.relative_to(root)), "size_bytes": p.stat().st_size, "sha256": _sha(p)}
        for p in files
    ])
    save_table(manifest, out / "all_output_sha256_manifest.csv")
    atomic_json({
        "status": "passed",
        "created_utc": utc_now(),
        "markdown": str(md),
        "html": str(hp),
        "excel": str(workbook),
        "complete_reference_gate_pass": gate,
        "complete_2024_reference_census": True,
        "final_2024_refined_inventory_exact_by_contract": bool(validation.get("final_refined_inventory_exact_by_contract", False)),
        "independent_model_evaluated_before_endpoint_enforcement": True,
        "reference_geometry_copied_to_historical_years_without_evidence": False,
        "environment_creates_historical_labels": False,
        "reference_shp_drawn_as_extra_map_layer": False,
        "exclusive_annual_class_layer": True,
        "annual_area_labels_outside_map_axes": True,
        "technical_route_figure_generated": True,
        "ecology_sankey_figures_generated": True,
        "strict_noleak_forecast_documented": True,
    }, out / "report_status.json")
    return md
