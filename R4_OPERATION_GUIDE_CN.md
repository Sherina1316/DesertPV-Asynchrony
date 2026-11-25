# v24.22.2 混合像元光伏地块版操作流程

## 一、部署原则

本修订改变了 Stage 4 的质量路由，以及 Stage 7/8 的混合像元地块重建。因此旧版只从 Stage 7 续跑的方式不够：Unit 35 如果此前被判为面积路线，必须重新经过 Stage 4 才会切换为 `anchor_guided_candidate` 空间路线。

推荐顺序：

```text
部署 R4 → 自检 → 清理并重跑 Unit 35/107 → 人工验收 → 正式结果从 Stage 4 重跑
```

## 二、服务器文件

上传到 `/home/wsy/`：

```text
china_inland_desert_pv_v24_22_mixed_parcel_optimized_r4.zip
pv_v24_22_mixed_parcel_ops_r4.sh
```

## 三、部署和自检

```bash
cd /home/wsy
source /home/gis007/.venvs/pv_nature/bin/activate
chmod +x /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh

bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh verify
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh deploy
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh self-test
```

部署目录保持不变：

```text
/home/wsy/pv_nature_experiment/code/china_inland_desert_pv_v24_22_detail_optimized
```

部署脚本会先备份旧代码目录，然后安装 R4。

## 四、重新运行 Unit 35、107

不要继续使用旧测试输出。执行：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh test 35,107 foreground
```

后台运行：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh test 35,107 background
```

查看状态：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh status-test 35,107
```

只查看新日志：

```bash
tail -n 0 -f \
/home/wsy/pv_nature_experiment/logs_v24_22_mixed_parcel_test_units_35_107/test_units_35_107.log
```

## 五、先检查 Stage 4 路由

```bash
python - <<'PY'
import pandas as pd
p = "/home/wsy/pv_nature_experiment/output_v24_22_mixed_parcel_test_units_35_107/04_unit_quality/unit_routing_audit.csv"
df = pd.read_csv(p)
cols = [
    "Id", "quality_score", "estimation_route", "direct_detection_eligible",
    "anchor_guided_eligible", "mixed_pixel_anchor_rescue",
    "anchor_sample_pixels", "mixed_anchor_sample_pixels",
    "pixel_mapping_eligible", "anchor_guided_route_reason",
]
print(df.loc[df["Id"].isin([35, 107]), cols].to_string(index=False))
PY
```

Unit 35 必须满足：

```text
pixel_mapping_eligible = True
estimation_route = direct_pixel 或 anchor_guided_candidate
```

若未达到 direct 门槛，正常结果应为：

```text
mixed_pixel_anchor_rescue = True
anchor_guided_route_reason = mixed_pixel_anchor_rescue
```

## 六、检查 Stage 7/8 算法版本

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("/home/wsy/pv_nature_experiment/output_v24_22_mixed_parcel_test_units_35_107")
for rel in [
    "07_authoritative_reference/authoritative_reference_status.json",
    "08_temporal_pv_change/temporal_decoding_status.json",
]:
    p = root / rel
    d = json.loads(p.read_text(encoding="utf-8"))
    print(rel)
    print(" status:", d.get("status"))
    print(" schema_version:", d.get("schema_version"))
    print(" algorithm_revision:", d.get("algorithm_revision"))
PY
```

应显示：

```text
schema_version = 24.22.0
algorithm_revision = 24.22.2-mixed-parcel-optimized
```

## 七、图件验收

重点检查：

```text
17_maps/annual_pv_change_atlas/
08_temporal_pv_change/first_installed_year_30m/
08_temporal_pv_change/cumulative_installed_pv_masks_30m/
08_temporal_pv_change/historical_growth_front_rescue_30m/
07_authoritative_reference/reference_inventory_2024_30m/
07_authoritative_reference/reference_boundary_correction_code_30m/
```

### Unit 35

- 2024 年权威地块不能为空；
- 地块内混合像元应形成空间掩膜；
- 历史年份只有在当年影像证据广泛分布并持续时才整块出现；
- 不能把 2024 地块无条件复制到 2011 年。

### Unit 107

- 2016—2018 年红框内已经存在的面板区域应被恢复；
- 首次安装年份不应整体推迟到 2019—2020 年；
- 若早期只是局部建设，只能恢复局部，不能提前填满整个 2024 地块。

### 通用检查

- 2024 外边界不出现完整一圈新增；
- 2025 不出现连续窄环伪扩张；
- 小型正确 SHP 块不能消失；
- 沙地、道路和单个黑斑不能触发整地块填充。

## 八、测试中断后的续跑

从指定阶段继续：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh \
  resume-test 35,107 7 foreground
```

由于首次应用 R4 必须重新计算 Stage 4 路由，第一次不要直接从 Stage 7 续跑；只有 R4 的 Stage 4 已完成后才使用该命令。

## 九、正式 142 单元从 Stage 4 重跑

测试满意后：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh full-from4 foreground
```

后台运行：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh full-from4 background
```

该命令会：

- 保留 Stage 1—3 的输入、30 m 对齐和 2024 SHP 栅格化；
- 备份 Stage 4—21 的旧输出；
- 备份 Stage 4 训练缓存和模型目录；
- 重新执行质量路由、训练、年度推理、时序重建、生态分析、2050 预测、地图和报告；
- 继续写入原正式输出目录，保持所有下游路径不变。

查看状态：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh status-full
```

中断后从相应阶段继续，例如 Stage 8：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh resume-full 8 foreground
```

不要再次执行 `full-from4`，否则会再次备份 Stage 4 后的当前结果。

## 十、从头重跑

只有 Stage 1—3 本身不完整时才执行：

```bash
bash /home/wsy/pv_v24_22_mixed_parcel_ops_r4.sh fresh-full foreground
```
