# 物理仿真器性能基准测试

这个目录包含了多个物理仿真器的统一性能基准测试工具，支持 Genesis、MotrixSim、MotrixSim-Warp、IsaacSim 和 MuJoCo-Warp。

## 目录结构

```
bench/
├── benchmark_all.py          # 主编排器，批量运行所有测试
├── bench_genesis.py          # Genesis 仿真器测试
├── bench_isaacsim.py         # IsaacSim 仿真器测试
├── bench_motrixsim.py        # MotrixSim 仿真器测试
├── bench_motrixsim_warp.py   # MotrixSim-Warp 仿真器测试
├── bench_mujocowarp.py       # MuJoCo-Warp 仿真器测试
├── bench_output_utils.py     # 输出和统计工具
└── bench_result_visualizer.py # HTML 报告生成器
```

## 测试模式

### 1. Franka Only 模式（`franka_only`）
- **目的**：测试多个 Franka 机械臂的随机运动性能
- **场景**：机械臂进行随机关节运动
- **支持选项**：
  - 标准场景（无杂物）
  - 杂乱场景（`--clutter`）：添加 200+ 个动态瓶子

### 2. Franka Grasp 模式（`franka_grasp`）
- **目的**：测试抓取任务的性能
- **场景**：机械臂抓取物体并提升
- **支持物体**：球（ball）、立方体（cube）、瓶子（bottle）
- **支持选项**：
  - 静态抓取（无抖动）
  - 抖动抓取（`-r`）：在提升过程中添加随机扰动

## 快速开始

### 1. 运行单个仿真器测试

```bash
# Genesis - Franka Only 模式，1 个机器人，1 个环境
python bench/bench_genesis.py -N 1 -B 1 --mode franka_only

# MotrixSim - Franka Grasp 模式，抓取球，带可视化
python bench/bench_motrixsim.py -N 1 -B 1024 --mode franka_grasp --object ball -v

# MuJoCo-Warp - Franka Only 模式，5 个机器人，4096 个并行环境
python bench/bench_mujocowarp.py -N 5 -B 4096 --mode franka_only

# IsaacSim - 杂乱场景测试
python bench/bench_isaacsim.py -N 1 -B 64 --mode franka_only --clutter
```

### 2. 批量运行所有测试

```bash
# 运行所有默认配置（N=1,5,10, 自动选择 B 值）
python bench/benchmark_all.py

# 只测试特定配置
python bench/benchmark_all.py -N 1 -B 1024 --modes franka_only

# 测试多个 N 和 B 的组合
python bench/benchmark_all.py -N 1 5 -B 1024 4096

# 只测试特定仿真器
python bench/benchmark_all.py --simulators genesis motrixsimwarp

# 带可视化运行（警告：会很慢）
python bench/benchmark_all.py -N 1 -B 1 -v
```

### 3. 生成报告（不运行测试）

```bash
# 从已有的 JSON 结果生成 HTML 报告
python bench/benchmark_all.py --report-only
```

## 命令行参数详解

### 单个仿真器脚本参数

所有单个仿真器脚本（`bench_*.py`）支持以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-N` | int | 1 | 机器人数量（1, 5, 或 10） |
| `-B` | int | 1 | 批量大小/并行环境数 |
| `-v` | flag | False | 启用可视化 |
| `--mode` | str | franka_only | 测试模式：`franka_only` 或 `franka_grasp` |
| `--object` | str | ball | Grasp 模式的物体：`ball`, `cube`, `bottle` |
| `-r` | flag | False | Grasp 模式抖动（添加随机扰动） |
| `--clutter` | flag | False | Franka Only 模式杂乱场景（仅支持部分仿真器） |

### benchmark_all.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-N, --robots` | int[] | [1,5,10] | 要测试的机器人数量列表 |
| `-B, --batches` | int[] | 自动 | 批量大小列表（未指定时自动缩放） |
| `-v` | flag | False | 启用可视化 |
| `--modes` | str[] | 全部 | 测试模式：`franka_only`, `franka_grasp` |
| `--simulators` | str[] | 全部 | 仿真器列表 |
| `--objects` | str[] | [ball,cube,bottle] | Grasp 模式物体列表 |
| `--no-clutter` | flag | False | 跳过 Franka Only 杂乱场景测试 |
| `--no-shake` | flag | False | 跳过 Franka Grasp 抖动测试 |
| `--report-only` | flag | False | 只生成报告，不运行测试 |
| `--no-report` | flag | False | 跳过 HTML 报告生成 |
| `--force` | flag | False | 强制重新运行所有测试，忽略已有结果 |
| `--force-fail` | flag | False | 强制重新运行失败的测试 |

## 批量大小自动缩放规则

当未指定 `-B` 参数时，系统会根据机器人数量和模式自动选择合适的批量大小：

### Franka Only 模式
- **标准场景**：
  - N=1: [1, 1024, 4096, 8192]
  - N=5: [1, 512, 2048, 4096]
  - N=10: [1, 256, 1024, 2048]
- **杂乱场景**：[1, 16, 32, 64]（不论 N 值）

### Franka Grasp 模式
- N=1: [1, 1024, 4096, 8192]
- N=5: [1, 512, 2048, 4096]
- N=10: [1, 256, 1024, 2048]

## 输出结果

### JSON 结果文件

所有测试结果保存在 `output/bench/` 目录下，按硬件分组：

```
output/bench/
├── NVIDIA_GeForce_RTX_4090/      # GPU 结果
│   ├── genesis_franka_only_n1_b1024.json
│   ├── genesis_franka_grasp_ball_n1_b1024.json
│   └── ...
└── Intel_R__Xeon_R__CPU.../      # CPU 结果
    ├── motrixsim_franka_only_n1_b1024.json
    └── ...
```

### JSON 格式示例

```json
{
  "simulator": "genesis",
  "simulator_name": "Genesis",
  "mode": "franka_only",
  "n_robots": 1,
  "batch_size": 1024,
  "per_env_fps": 1234.56,
  "total_fps": 1234567.89,
  "object": null,
  "clutter": false,
  "release": false,
  "status": "success",
  "timestamp": "2024-01-30T12:34:56",
  "hardware_name": "NVIDIA_GeForce_RTX_4090"
}
```

### HTML 报告

HTML 报告生成在 `output/bench/html/` 目录：

- `franka_only.html` - Franka Only 标准场景
- `franka_only_clutter.html` - Franka Only 杂乱场景
- `franka_grasp.html` - Franka Grasp 标准抓取
- `franka_grasp_shake.html` - Franka Grasp 抖动抓取

报告包含：
- 按机器人数量分组的性能图表
- 按批量大小分组的性能图表
- 详细的性能矩阵表格
- 硬件信息和时间戳

## 使用示例

### 示例 1：快速性能测试

```bash
# 测试单个配置，快速查看性能
python bench/bench_genesis.py -N 1 -B 1024 --mode franka_only

# 输出示例：
# per env: 1,234.56 FPS
# total  : 1,264,189.44 FPS
```

### 示例 2：完整基准测试

```bash
# 运行所有仿真器的完整测试套件
python bench/benchmark_all.py

# 这将测试：
# - 所有 5 个仿真器
# - 两种模式（franka_only, franka_grasp）
# - N=1,5,10
# - 自动缩放的批量大小
# - 生成 4 个 HTML 报告
```

### 示例 3：自定义测试

```bash
# 只测试 GPU 仿真器，大批量
python bench/benchmark_all.py \
  --simulators genesis motrixsimwarp mujocowarp \
  -N 1 \
  -B 4096 8192 \
  --modes franka_only

# 只测试 Grasp 模式，所有物体
python bench/benchmark_all.py \
  --modes franka_grasp \
  --objects ball cube bottle \
  -N 1 5 \
  -B 1024
```

### 示例 4：杂乱场景压力测试

```bash
# 测试杂乱场景（200+ 动态物体）
python bench/bench_genesis.py -N 1 -B 32 --mode franka_only --clutter

# 或使用 benchmark_all
python bench/benchmark_all.py \
  --modes franka_only \
  -N 1 \
  -B 16 32 64 \
  --no-clutter  # 如果只想测试标准场景
```

### 示例 5：可视化调试

```bash
# 启用可视化，小批量，方便观察
python bench/bench_genesis.py -N 1 -B 1 --mode franka_grasp --object ball -v

# 注意：可视化会显著降低性能，仅用于调试
```

### 示例 6：增量测试

```bash
# 第一次运行完整测试
python bench/benchmark_all.py

# 后续只重新运行失败的测试
python bench/benchmark_all.py --force-fail

# 或强制重新运行所有测试
python bench/benchmark_all.py --force
```

### 示例 7：仅生成报告

```bash
# 已经有了 JSON 结果，只想更新 HTML 报告
python bench/benchmark_all.py --report-only

# 自定义报告输出路径（已弃用，现在固定使用 html/ 子目录）
```

## 性能优化建议

### 1. 批量大小选择
- **小批量 (B=1-64)**：适合调试和可视化
- **中批量 (B=512-2048)**：平衡性能和内存
- **大批量 (B=4096-8192)**：最大吞吐量，需要大内存

### 2. 机器人数量
- **N=1**：最简单场景，基准性能
- **N=5**：中等复杂度
- **N=10**：高复杂度，测试可扩展性

### 3. 仿真器选择
- **GPU 仿真器**：Genesis, MotrixSim-Warp, MuJoCo-Warp, IsaacSim
  - 适合大批量并行
  - 需要 NVIDIA GPU
- **CPU 仿真器**：MotrixSim
  - 适合中小批量
  - CPU 多核并行

## 故障排查

### 常见问题

**Q: 测试超时 (TIMEOUT 错误)**
```
{"status": "error", "error_code": "TIMEOUT", ...}
```
A: 降低批量大小或机器人数量，或增加硬件资源。

**Q: 内存不足**
A: 减小批量大小。杂乱场景特别消耗内存。

**Q: 找不到已有结果**
A: 确保 JSON 文件在正确的硬件子目录下（`output/bench/<hardware>/`）。

**Q: HTML 报告为空**
A: 检查 `output/bench/` 目录下是否有 JSON 结果文件。使用 `--force` 重新运行测试。

**Q: 可视化不显示**
A:
- 检查是否在有显示的环境中运行
- 某些仿真器可能需要额外的依赖
- IsaacSim 需要特殊的窗口系统配置

## 环境要求

### 基础依赖
- Python 3.8+
- NumPy

### 仿真器特定依赖
- **Genesis**: `genesis` 包
- **MotrixSim**: `motrixsim` 包
- **MotrixSim-Warp**: `motrixsim_warp`, `warp`
- **IsaacSim**: Isaac Sim 安装（需要 NVIDIA Omniverse）
- **MuJoCo-Warp**: `mujoco`, `mujoco_warp`, `warp`

### 可视化依赖
- 各仿真器的渲染后端
- 显示服务器（X11, Wayland 等）

## 重要注意事项

⚠️ **Breaking Changes**:
- 模式名称已从 `random`/`grasp` 改为 `franka_only`/`franka_grasp`
- 旧的 JSON 结果文件需要手动重命名或重新生成
- 移除了 `--clutter-only` 和 `--release-only` 标志
- `--no-release` 改名为 `--no-shake`

## 开发者信息

### 添加新的仿真器

1. 创建新的 `bench_<simulator>.py` 脚本
2. 实现相同的参数接口（`-N`, `-B`, `--mode` 等）
3. 在 `benchmark_all.py` 中添加配置
4. 更新硬件映射（如果是 GPU 仿真器）

### 扩展测试场景

1. 修改 `get_batch_sizes_for_robot_count()` 添加新的缩放规则
2. 更新报告配置以包含新场景
3. 确保 JSON 输出格式一致

## 许可证

请参考项目根目录的许可证文件。
