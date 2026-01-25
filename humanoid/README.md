# Humanoid Benchmark Report

该测试用于评估不同物理引擎在场景复杂度增加时的性能表现。

测试包含两个维度：

1. 单个Env中，Humanoid模型的数量(N)
2. 并行仿真的Env数量(B, BatchSize)

> **说明**：motrixsim采用完全隐式的约束模型求解器。因此在对比测试时，我们将mujoco的integrator也设置为了implicit。 尽管如此，mujoco的implicit integrator仍然没有所有的force都采用implicit的方法。详情参考mujoco的文档: https://mujoco.readthedocs.io/en/stable/computation/index.html#integrators

测试结果查看： **[📊 Click here to view the latest test report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Motphys/phys-bench/refs/heads/main/output/humanoid/comparison_report.html)**

从该测试结果发现：

1. mujoco在单线程下的性能表现优于motrixsim。特别是当humanoid只有一个，且BatchSize规模上来时，MuJoCo的scale能力比MotrixSim更强。这是由于mujoco的rollout在实现时，单独从data中提取了state给每个Env，而Data在单个线程下是共享的。rollout的设计参考:[rollout](https://mujoco.readthedocs.io/en/stable/python.html#rollout)。这种设计使得mujoco在单线程下的切换Env时，能很好的控制cpu cache miss，但同时也带来了使用的复杂度。MotrixSim则将线程的调度以及数据的存储方式隐藏起来不向用户暴露，在接口使用上更为简洁。 后续MotrixSim将在保持接口简洁的前提下，优化单线程多Env下的cpu cache命中表现。

2. 当场景复杂度增加时（单环境的Humanoid数量从1到10），MuJoCo的性能快速下降，而MotrixSim的性能下降较为平缓。这表明MotrixSim在处理复杂场景时具有更好的扩展性。尤其是在单环境下仿真50个Humanoid时，MotrixSim可以跑到500+fps，而MuJoCo只有17FPS左右。 这是因为MotrixSim即使在单环境下仿真也能使用多线程加速。

# Run bench on local

```
cd phys-bench
uv run humanoid/run_all.py
```
