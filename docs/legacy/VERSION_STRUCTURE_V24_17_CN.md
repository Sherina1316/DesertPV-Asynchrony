# v24.18 完整版本说明与可运行结构

## 一、核心数据定义

`/home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.shp` 是2024年活动光伏板分布的完整清单：

- 矢量范围内在2024年一定是光伏；
- 矢量范围外在2024年一定不是光伏；
- 2024最终活动光伏图与该矢量的统一30 m栅格结果逐像元一致；
- 2024结果不再由影像增加、删除或修边；
- 该矢量只约束2024年，不会复制到2011—2023年。

边界像元同时保留矢量覆盖比例，用于精确面积计算。二值分布使用“任意非零矢量覆盖即属于2024光伏清单”的统一规则，Stage 3、7、8和9复用同一份二值栅格，避免不同阶段二次阈值化造成边界不一致。

## 二、新的历史重建算法

### 1. 完整清单监督

2024矢量内作为正样本，矢量外作为2024负样本。训练、概率校准和独立测试采用空间隔离单元，从而可以正式计算全景precision、recall、F1、IoU、boundary F1和面积误差。

### 2. 年度综合证据

2011—2025逐年综合：

- Landsat BLUE、GREEN、RED、NIR、SWIR1、SWIR2；
- NDVI、EVI、NDMI、NDBI、MNDWI、BSI；
- tasseled-cap brightness/greenness/wetness；
- LSMA高反照率、低反照率、植被、阴影组分及RMSE；
- 集成模型概率和不确定性；
- 与2024真实光伏原型、沙地原型的相似度；
- 局地暗度、对象尺度、纹理方向和稳定沙地风险；
- 年际光谱/概率变化及后续持续性。

环境变量只能降低异常年份的影像证据可靠度，不能直接把某个位置判为历史光伏。

### 3. 有序安装变化点

对2024清单内每个像元依次评估可能的安装年份。只有安装后证据、支持年数和安装前后增益均达到要求，才允许回溯到历史年份。证据不足时，首次安装年份默认设为2024，而不是2011。

### 4. 2024清单外历史场站

矢量外在2024年强制为非光伏。历史年份若存在真实场站，必须形成连续多年的独立轨迹，并在2024年前结束；单年沙地纹理或暗斑不能进入结果。2025年的新场站需要极强的独立证据。

### 5. 双状态产品

- `Observed active PV`：当年活动光伏，2024年严格等于修订SHP；
- `Cumulative installed PV`：历史上曾经安装过的累计范围，可包含2024年前已经拆除或退役的历史场站。

因此，2024“当前光伏分布”不会被历史退役范围修改，同时仍可开展退役和土地遗留影响分析。

## 三、环境与生态综合分析

环境变量包括降水、温度、潜在蒸散、实际蒸散、土壤湿度、太阳辐射、风速等。分析包括：

- PV核心、边缘缓冲区和3—5 km背景的固定空间支持；
- PAVG、NDVI/EVI、绿度、水分、湿度和植被丰度；
- 降水、温度、PET、土壤湿度、辐射和风速控制；
- 同年尚未建设/未建设单元对照；
- 倾向匹配和分期事件研究；
- 响应滞后年数和稳定年数；
- 光伏扩张速度与生态调整速度对比；
- 两张桑基图展示扩张—生态响应—滞后—2050压力路径。

## 四、2050预测

使用严格无泄漏滚动历史回测：预测年份只能使用此前年份。模型先判断下一年是否发生新增安装，再预测新增面积，并施加有效空间容量、历史增长率和覆盖率限制。环境变量以滞后值或未来情景进入。结果输出区域年度中位数、10%—90%区间、模型分歧和空间候选范围。

## 五、21阶段结构

```text
01 输入与环境验证
02 统一投影30 m网格
03 完整2024清单栅格化与空间样本划分
04 LSMA、质量路由与Annual-CCDC缓存
05 空间阻断模型基准
06 最终集成模型
07 年度推理、完整清单校准与权威2024端点
08 2011—2025安装/退役/2025新增重建
09 完整2024清单独立验证
10 全链不确定性
11 光伏动态
12 动态生态与环境控制
13 扩张事件
14 条件关联和对照
15 响应滞后与稳定分析
16 2050空间情景包络
17 年度地图和15年图集
18 Nature风格图件及两张桑基图
19 空间与科学声明审计
20 中文报告、Excel与SHA256清单
21 严格无泄漏历史回测和2050递归预测
```

## 六、服务器目录

```text
/home/wsy/pv_nature_experiment/
├── code/china_inland_desert_pv_v24_18_adaptive_reference_boundary/
├── input_tables/pv_desert_fix.shp
├── output_v24_18_adaptive_reference_boundary/
├── cache_v24_18_adaptive_reference_boundary/
├── logs_v24_18_adaptive_reference_boundary/
├── models_v24_18_adaptive_reference_boundary/
├── output_v24_18_test_units_6_28/
├── cache_v24_18_test_units_6_28/
├── logs_v24_18_test_units_6_28/
└── models_v24_18_test_units_6_28/
```

## 七、先测试Unit 6和28

```bash
cd /home/wsy
source /home/gis007/.venvs/pv_nature/bin/activate
sha256sum -c /home/wsy/china_inland_desert_pv_v24_18_adaptive_reference_boundary_SHA256.txt
chmod +x /home/wsy/*v24_18*.sh
bash /home/wsy/test_v24_18_authoritative_units_6_28.sh --yes --foreground --no-self-test
```

检查：

```bash
bash /home/wsy/check_china_inland_desert_pv_v24_18_adaptive_reference_boundary.sh test
```

恢复：

```bash
bash /home/wsy/resume_china_inland_desert_pv_v24_18_adaptive_reference_boundary.sh \
  --mode test --from-step 7 --rerun-completed --foreground
```

## 八、全量运行

```bash
bash /home/wsy/deploy_china_inland_desert_pv_v24_18_adaptive_reference_boundary.sh \
  --yes --foreground --no-self-test
```

## 九、关键输出

```text
07_authoritative_reference/reference_anchor_2024_30m/
07_authoritative_reference/annual_evidence_probability_30m/
08_temporal_pv_change/observed_active_pv_masks_30m/
08_temporal_pv_change/cumulative_installed_pv_masks_30m/
08_temporal_pv_change/first_installed_year_30m/
09_accuracy_validation/complete_reference_validation_by_unit.csv
11_pv_dynamics/annual_pv_dynamics.csv
12_dynamic_ecology/
17_maps/
18_figures/figure_10_ecological_response_sankey.*
18_figures/figure_11_multimetric_ecology_sankey.*
21_strict_noleak_forecast/
20_report/complete_research_report_CN.html
```

代码提供出版级方法约束、审计和图件输出，但不能在真实全量独立验证完成前保证固定精度或Nature录用。
