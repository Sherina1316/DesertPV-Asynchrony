# v24.22.5（R4 双阶段融合版）服务器操作指南

## 一、文件位置

上传到服务器：

```text
/home/wsy/china_inland_desert_pv_v24_22_r4_dual_stage_fusion_r7.zip
/home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh
```

本版只以 R4 为基线，不使用 R5/R6。

## 二、检查、部署和自检

```bash
cd /home/wsy
source /home/gis007/.venvs/pv_nature/bin/activate
chmod +x /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh

bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh verify
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh deploy
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh self-test
```

部署目录仍为：

```text
/home/wsy/pv_nature_experiment/code/china_inland_desert_pv_v24_22_detail_optimized
```

## 三、重新测试 Unit 35、107

Stage 3 的“权威 2024 地块”和“强训练核心”语义已经分离，因此不能复用旧 R4/R5/R6 的 Stage 3 结果。测试必须从 Stage 1 开始：

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh test 35,107 foreground
```

后台运行：

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh test 35,107 background
```

查看状态和新日志：

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh status-test 35,107

tail -n 0 -f \
/home/wsy/pv_nature_experiment/logs_v24_22_r4_dual_stage_test_units_35_107/test_units_35_107.log
```

## 四、检查两阶段结果

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh inspect-test 35,107
```

重点文件：

```text
08_temporal_pv_change/dual_stage_global_reference_masks_30m
08_temporal_pv_change/dual_stage_internal_shp_buildout_30m
08_temporal_pv_change/dual_stage_fusion_source_30m
08_temporal_pv_change/dual_stage_internal_evidence_30m
08_temporal_pv_change/dual_stage_reference_spectral_similarity_30m
08_temporal_pv_change/cumulative_installed_pv_masks_30m
08_temporal_pv_change/first_installed_year_30m
17_maps/annual_pv_change_atlas
```

Unit 35 应满足：

- `has_authoritative_2024_pv=True`；
- 即使 `has_strong_training_core=False`，仍保持空间分析路线；
- 2011—2023 年只按实际年度影像证据恢复，不复制 2024 SHP；
- 局部修建期只显示局部，广泛覆盖后才显示完整地块；
- 2024 年等于边界审计后的权威库存；
- 面积表与年度空间标注一致。

## 五、测试中断续跑

例如从 Stage 8 继续：

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh resume-test 35,107 8 foreground
```

## 六、正式 142 单元重算

因为 Stage 3 标签语义和 Stage 4 路由均已改变，正式结果应从 Stage 3 重算，保留 Stage 1—2 对齐数据：

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh full-from3 foreground
```

后台运行：

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh full-from3 background
```

该命令会自动备份 Stage 3—21 的旧结果、相关训练缓存和模型，然后仍写入原正式目录：

```text
/home/wsy/pv_nature_experiment/output_v24_22_novelty_gated_remote_change
```

中断后，例如从 Stage 8 继续：

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh resume-full 8 foreground
```

不要再次执行 `full-from3`，否则会再次备份当前结果。

## 七、完整从头运行

仅当 Stage 1—2 也不完整时使用：

```bash
bash /home/wsy/pv_v24_22_r4_dual_stage_ops_r7.sh fresh-full foreground
```
