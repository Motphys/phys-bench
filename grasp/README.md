# Grasp Benchmark Tests

改测试旨在测试不同物理引擎在抓取任务中的稳定性和性能表现。

测试引擎包括: MotrixSim, IsaacSim, MuJoCo, Genesis。

机械臂使用franka。

抓取物体包括：球，方块和瓶子（凸多面体）

# 命令

```bash
uv run grasp/run_all_grasp_tests.py
```

测试结果会保存到`output/grasp`

打开`output/grasp/comparison_report.html`可以查看报告和视频。

您也可以点击**[📊 Click here to view the latest test report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Motphys/phys-bench/refs/heads/main/output/grasp/comparison_report.html)** 查看已生成的测试结果。

# 结果分析

测试结果表明，不同物理引擎在抓取任务中的表现存在显著差异。

- MotrixSim通过了所有测试用例，表现最为出色。
- IsaacSim在球和方块抓取测试中表现良好，但在瓶子抓取测试中失败。
- Genesis通过了方块和瓶子抓取测试（但物体仍在下滑），在球抓取测试中失败。
- MuJoCo几乎所有测试都没有通过。 即便将积分器改为Implicit以及将noslip_interations设为1,仍旧很难通过测试。
