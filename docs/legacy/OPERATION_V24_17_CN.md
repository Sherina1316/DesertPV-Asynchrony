# v24.18 中文运行说明

2024 参考矢量固定为 `/home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.shp`。该矢量是 2024 年光伏分布的充分必要条件，2024 年输出不允许增加或删除。

## Unit 6、28 测试

```bash
cd /home/wsy
source /home/gis007/.venvs/pv_nature/bin/activate
sha256sum -c /home/wsy/china_inland_desert_pv_v24_18_adaptive_reference_boundary_SHA256.txt
chmod +x /home/wsy/*v24_18*.sh
bash /home/wsy/test_v24_18_authoritative_units_6_28.sh --yes --foreground --no-self-test
```

## 全量运行

```bash
bash /home/wsy/deploy_china_inland_desert_pv_v24_18_adaptive_reference_boundary.sh --yes --foreground --no-self-test
```

## 恢复与检查

```bash
bash /home/wsy/resume_china_inland_desert_pv_v24_18_adaptive_reference_boundary.sh --mode test --from-step 7 --rerun-completed --foreground
bash /home/wsy/check_china_inland_desert_pv_v24_18_adaptive_reference_boundary.sh test
```
