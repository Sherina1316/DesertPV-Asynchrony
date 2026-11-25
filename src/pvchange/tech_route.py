from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .utils import ensure_dir


def _box(ax, x, y, w, h, title, body, facecolor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.1, edgecolor="#3D3D3D", facecolor=facecolor))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(x + 0.018, y + h - 0.085, body, ha="left", va="top", fontsize=10.2, linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17, linewidth=1.3, color="#555555", shrinkA=5, shrinkB=5))


def generate_chinese_technical_route(cfg: dict[str, Any]) -> dict[str, str]:
    out = ensure_dir(Path(cfg["paths"]["output_root"]) / "20_report")
    font_candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    selected_family = "DejaVu Sans"
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            selected_family = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams.update({
        "font.family": selected_family,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.2, 9.3), dpi=220, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "中国内陆沙漠光伏权威端点重建、生态响应与2050预测技术路线", ha="center", va="top", fontsize=21, fontweight="bold")
    ax.text(0.5, 0.932, "2011–2025 Landsat 30 m 年度影像 × 2024完整光伏清单 × 环境协变量 × 生态多指标 × 2050情景", ha="center", va="top", fontsize=11.5, color="#444444")
    boxes = [
        (0.04,0.70,"模块 1  多源数据与统一网格","Landsat 六波段年度影像\n2024 光伏 SHP\n环境与土地覆盖数据\n统一投影、30 m 对齐和质量检查","#EAF2FB"),
        (0.37,0.70,"模块 2  监督样本与年度概率","SHP 内确定正类\n显式混淆地物硬负样本\nXGBoost / RF / HGB 比较\n年度校准光伏概率图","#EEF7EA"),
        (0.70,0.70,"模块 3  光谱与时序证据","PV spectral evidence\nLSMA 组分与 RMSE\nCCDC 风格概率/组分断点\n水体、植被、建筑、裸地惩罚","#F8F1E5"),
        (0.04,0.41,"模块 4  权威端点与历史变化点","2024输出与修订SHP严格一致\n逐像元回看波段、概率与LSMA轨迹\n有序变化点估计安装年份\n证据不足时默认从2024开始","#FDEDEC"),
        (0.37,0.41,"模块 5  连续扩张与对象拓扑","删除孤立小斑块\n优先接受与既有光伏相连的扩张前沿\n近锚点新阵列需对象尺度与高置信支持\n远程新场站需大而连续","#F4ECF7"),
        (0.70,0.41,"模块 6  双状态年度产品","累计安装足迹与活动足迹分离\n2024活动足迹 = 修订SHP精确栅格\n历史年份不复制2024几何\n2025按新证据识别扩张","#EBF5FB"),
        (0.04,0.12,"模块 7  验证与不确定性","模型独立正类召回\n对象覆盖与时序一致性\n阈值、边界、模型扰动\n碎片度和混淆审计","#E8F8F5"),
        (0.37,0.12,"模块 8  动态与生态响应","年度面积、新增、累计与扩张方向\n事件检测与响应滞后\n核心—边缘—背景距离梯度\nCBC-PAVG、绿度、水分和环境控制","#FEF9E7"),
        (0.70,0.12,"模块 9  预测、制图与报告","2050 条件性情景\nNature 风格单一互斥分类层\n面积变化放在图外信息栏\nPNG/PDF/SVG 与中文总报告","#FDEDEC"),
    ]
    for x,y,title,body,fc in boxes: _box(ax,x,y,0.26,0.19,title,body,fc)
    for y in (0.79,0.50,0.21):
        _arrow(ax,0.30,y,0.37,y); _arrow(ax,0.63,y,0.70,y)
    for x in (0.17,0.50,0.83):
        _arrow(ax,x,0.70,x,0.60); _arrow(ax,x,0.41,x,0.31)
    ax.text(0.5,0.045,"最终产品遵循：2024人工矢量稳定内部保持不变，边界仅在限定缓冲带内依据多波段突变和时序持续性修正；2011–2023依据影像与变化点重建；2025依据独立证据识别；环境变量不制造历史光伏；地图不叠加半透明光伏图层。",ha="center",va="center",fontsize=10.5,color="#333333")
    outputs={}
    for ext,dpi in (("png",320),("pdf",None),("svg",None)):
        path=out/f"chinese_technical_route.{ext}"
        kwargs={"bbox_inches":"tight","facecolor":"white"}
        if dpi: kwargs["dpi"]=dpi
        fig.savefig(path,**kwargs); outputs[ext]=str(path)
    plt.close(fig)
    md=out/"chinese_technical_route_explanation.md"
    md.write_text("# 中文技术路线说明\n\n技术路线包含完整2024清单监督、年度影像证据、有序安装变化点、历史退役与2025新增识别、环境协变量控制、双状态时序、完整清单验证、不确定性、生态响应桑基图、严格无泄漏2050预测与Nature风格制图。\n",encoding="utf-8")
    outputs["markdown"]=str(md)
    return outputs
