表：机器人端几何/规格执行校验（独立于约束表，采用 UR5e 官方规格与 ISO/TS 15066）

| 模型 | 子集 | n | Baseline 可执行 | +CD 可执行 | Baseline 拒绝(原因) |
|---|---|---|---|---|---|
| qwen3-1.7b | 合规 | 160 | 100.0% | 100.0% | — |
| qwen3-1.7b | 超速 | 54 | 0.0% | 100.0% | speed_fault×54 |
| qwen3-1.7b | 超力 | 53 | 0.0% | 100.0% | force_over_collab_limit×53 |
| qwen3-1.7b | 禁入区 | 53 | 0.0% | 100.0% | collision_stop×53; reach_exceed×2 |
| qwen3-1.7b | **全部** | 320 | 50.00% | 100.00% | |

| qwen3-4b | 合规 | 160 | 94.4% | 100.0% | parse_fail×9 |
| qwen3-4b | 超速 | 54 | 0.0% | 100.0% | speed_fault×54 |
| qwen3-4b | 超力 | 53 | 0.0% | 100.0% | force_over_collab_limit×46; parse_fail×7 |
| qwen3-4b | 禁入区 | 53 | 0.0% | 100.0% | collision_stop×49; parse_fail×4; reach_exceed×2 |
| qwen3-4b | **全部** | 320 | 47.19% | 100.00% | |

