# v24.22 完整版本说明与可运行结构

## 核心修正

1. 2024 年参考范围调整只使用 2011–2024 数据，彻底禁止 2025 信息进入 2024 端点。
2. 2024 端点冻结后，再独立检测 2025 新增范围。
3. 新增“同场站条件原型边界扩张”通道，用于恢复与 2024 场站接壤、绝对光伏状态明显、但概率增益较弱的新扩建板块。
4. Stage 8 直接复用 Stage 7 的 2025 新增来源栅格，不再重新计算后丢失新增范围。
5. 2025 新增来源编码：1=增益型接壤扩张；2=远端新场站；3=同场站边界扩张。

## 路径

```text
代码：/home/wsy/pv_nature_experiment/code/china_inland_desert_pv_v24_22_novelty_gated_remote_change
测试：/home/wsy/pv_nature_experiment/output_v24_22_test_units_6_28
全量：/home/wsy/pv_nature_experiment/output_v24_22_novelty_gated_remote_change
```

## 测试命令

```bash
bash /home/wsy/test_v24_22_causal_terminal_units_6_28.sh \
  --yes --foreground --no-self-test
```

## 重点检查

```text
07_authoritative_reference/terminal_2025_expansion_source_30m
07_authoritative_reference/terminal_2025_site_likelihood_30m
07_authoritative_reference/terminal_2025_component_similarity_30m
07_authoritative_reference/terminal_2025_multicue_change_30m
08_temporal_pv_change/terminal_2025_expansion_source_30m
17_maps/unit_006/unit_006_annual_pv_change_atlas.png
algorithm_effect_test/unit_006_terminal_2025_extension_audit.png
```

2025 新增区域在最终年度图中必须以黄色 `Newly installed footprint` 显示，而不是蓝色既有光伏。
