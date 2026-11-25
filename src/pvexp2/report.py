
from __future__ import annotations

from pathlib import Path
import os
import html
import pandas as pd


CN_EXTREME = (
    "极端残差主要集中在光伏扩张突变、覆盖率极低或极高、以及少数大面积区域，"
    "说明局部装机非连续性、地表异质性和年度遥感识别误差仍是长期预测不确定性的主要来源。"
)

EN_EXTREME = (
    "Extreme residuals were mainly associated with abrupt or spatially concentrated PV expansion, "
    "indicating that local installation discontinuities and land-surface heterogeneity remain difficult "
    "to fully capture with annual regional predictors."
)


def _table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "<p>No records.</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="tbl")


def _rel(fig: Path, html_path: Path) -> str:
    return os.path.relpath(Path(fig).resolve(), start=Path(html_path).resolve().parent)


def build_report(
    output_path: Path,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    extreme: pd.DataFrame,
    audits: pd.DataFrame,
    dirs: dict,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_files = []
    for k in ["validation", "forecast", "figures", "audit"]:
        d = Path(dirs[k])
        if d.exists():
            fig_files += sorted(d.glob("*.png"))

    imgs = []
    for f in fig_files:
        imgs.append(f"""
<figure>
  <img src="{html.escape(_rel(f, output_path))}" />
  <figcaption>{html.escape(f.name)}</figcaption>
</figure>
""")

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Strict no-leakage PV expansion forecasting experiments</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans CJK SC", sans-serif;
  max-width: 1180px; margin: 34px auto; color: #222; line-height: 1.55;
}}
h1,h2 {{ font-weight: 650; }}
.note {{
  background: #f6f8fa; border-left: 4px solid #0072B2; padding: 12px 16px; margin: 18px 0;
}}
.warn {{
  background: #fff8e1; border-left: 4px solid #D55E00; padding: 12px 16px; margin: 18px 0;
}}
.tbl {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
.tbl th,.tbl td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
.tbl th:first-child,.tbl td:first-child {{ text-align: left; }}
figure {{ margin: 24px 0; padding-bottom: 18px; border-bottom: 1px solid #eee; }}
img {{ max-width: 100%; height: auto; display: block; }}
figcaption {{ color: #555; font-size: 13px; margin-top: 6px; }}
</style>
</head>
<body>
<h1>严格无泄漏光伏扩张—生态响应预测实验报告</h1>

<div class="note">
本版本强制使用 lag-only / fixed-baseline 特征，剔除当前年目标值、目标同义字段和当前年目标派生项；
历史验证图统一标记为 <b>Historical hindcast validation</b>，未来外推图统一标记为 <b>Future projection to 2050</b>。
Bias 统一使用 <b>normalized bias = bias / observed range</b>。
</div>

<div class="warn">
本报告中的未来曲线属于情景式 projection，而不是逐年份精确预测。最终论文应优先报告 historical hindcast 和消融实验，
并将 2050 曲线解释为约束条件下的区域尺度外推。
</div>

<h2>1. 模型对比：严格无泄漏 historical hindcast</h2>
{_table(metrics.sort_values(["target","nrmse"]) if metrics is not None and not metrics.empty else metrics, 120)}

<h2>2. 特征消融实验</h2>
<p>消融组为 base、base+PV、base+PV+rolling/anomaly、full。该版本不再重复输出 ensemble 与单模型的重复柱。</p>
{_table(ablation.sort_values(["target","feature_group"]) if ablation is not None and not ablation.empty else ablation, 120)}

<h2>3. 泄漏审计结果</h2>
{_table(audits, 120)}

<h2>4. 极端残差与鲁棒性说明</h2>
<p>{CN_EXTREME}</p>
<p><i>{EN_EXTREME}</i></p>
{_table(extreme.sort_values("abs_residual", ascending=False) if extreme is not None and not extreme.empty else extreme, 80)}

<h2>5. 图表</h2>
{''.join(imgs)}
</body>
</html>"""
    output_path.write_text(html_text, encoding="utf-8")
