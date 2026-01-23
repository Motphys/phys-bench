# CPU Parallel Physics Simulation Benchmark

**Comparison of MuJoCo Rollout, Motrixsim, and Genesis for CPU batched physics simulation**

## Experimental Setup

### Hardware Configuration

| Parameter | Value |
|-----------|-------|
| CPU | Intel(R) Core(TM) i9-14900K |
| Physical Cores | 24 |
| Logical Cores | 32 |
| Operating System | Linux 6.8.0-90-generic |
| Python Version | 3.10.12 |
| Timestamp | 2026-01-23 02:26:18 |

### Benchmark Parameters

- **Scenario**: Franka Panda robot arm grasping task
- **Simulation Steps**: 500 steps per benchmark run
- **Timestep**: 0.01s (100 Hz)
- **Batch Sizes**: 1, 64, 512, 1024
- **Thread Counts (MuJoCo)**: 1, 4, 8, 16, 32

## Performance Comparison

### Throughput (Total FPS)

| Batch Size | MuJoCo | Threads | Motrixsim | Genesis |
|:----------:|-------:|:-------:|----------:|--------:|
| 1 | 117,984 | 1 | 47,716 | 5,908 |
| 64 | 1,153,146 | 32 | 546,378 | 50,808 |
| 512 | 1,447,244 | 32 | 580,189 | 49,643 |
| 1024 | 1,442,141 | 32 | 548,360 | 37,144 |

### Performance Ratios

| Batch Size | MuJoCo/Motrixsim | MuJoCo/Genesis | Motrixsim/Genesis |
|:----------:|-----------------:|---------------:|------------------:|
| 1 | 2.47x | 19.97x | 8.08x |
| 64 | 2.11x | 22.70x | 10.75x |
| 512 | 2.49x | 29.15x | 11.69x |
| 1024 | 2.63x | 38.83x | 14.76x |

## MuJoCo Rollout: Detailed Results

### Per-Environment FPS by Thread Count

| Batch Size | T=1 | T=4 | T=8 | T=16 | T=32 |
|:----------:|------:|------:|------:|------:|------:|
| 1 | 117,984 | 113,515 | 111,500 | 105,317 | 101,898 |
| 64 | 1,861 | 6,699 | 11,369 | 12,387 | 18,018 |
| 512 | 233 | 880 | 1,704 | 2,056 | 2,827 |
| 1024 | 117 | 432 | 836 | 1,064 | 1,408 |

### Total FPS by Thread Count

| Batch Size | T=1 | T=4 | T=8 | T=16 | T=32 |
|:----------:|------:|------:|------:|------:|------:|
| 1 | 117,984 | 113,515 | 111,500 | 105,317 | 101,898 |
| 64 | 119,088 | 428,709 | 727,602 | 792,743 | 1,153,146 |
| 512 | 119,105 | 450,596 | 872,242 | 1,052,483 | 1,447,244 |
| 1024 | 119,500 | 442,823 | 856,004 | 1,089,892 | 1,442,141 |

## Motrixsim: Results

| Batch Size | Per-Env FPS | Total FPS |
|:----------:|------------:|----------:|
| 1 | 47,716 | 47,716 |
| 64 | 8,537 | 546,378 |
| 512 | 1,133 | 580,189 |
| 1024 | 536 | 548,360 |

## Genesis: Results

| Batch Size | Per-Env FPS | Total FPS |
|:----------:|------------:|----------:|
| 1 | 5,908 | 5,908 |
| 64 | 794 | 50,808 |
| 512 | 97 | 49,643 |
| 1024 | 36 | 37,144 |

## Thread Scaling Analysis (MuJoCo)

Speedup is calculated relative to single-threaded performance (T=1).
Parallel efficiency is defined as: Efficiency = Speedup / Thread_Count × 100%

| Batch Size | Threads | Total FPS | Speedup | Parallel Efficiency |
|:----------:|--------:|----------:|--------:|--------------------:|
| 1 | 1 | 117,984 | 1.00x | 100.0% |
| 1 | 4 | 113,515 | 0.96x | 24.1% |
| 1 | 8 | 111,500 | 0.95x | 11.8% |
| 1 | 16 | 105,317 | 0.89x | 5.6% |
| 1 | 32 | 101,898 | 0.86x | 2.7% |
| 64 | 1 | 119,088 | 1.00x | 100.0% |
| 64 | 4 | 428,709 | 3.60x | 90.0% |
| 64 | 8 | 727,602 | 6.11x | 76.4% |
| 64 | 16 | 792,743 | 6.66x | 41.6% |
| 64 | 32 | 1,153,146 | 9.68x | 30.3% |
| 512 | 1 | 119,105 | 1.00x | 100.0% |
| 512 | 4 | 450,596 | 3.78x | 94.6% |
| 512 | 8 | 872,242 | 7.32x | 91.5% |
| 512 | 16 | 1,052,483 | 8.84x | 55.2% |
| 512 | 32 | 1,447,244 | 12.15x | 38.0% |
| 1024 | 1 | 119,500 | 1.00x | 100.0% |
| 1024 | 4 | 442,823 | 3.71x | 92.6% |
| 1024 | 8 | 856,004 | 7.16x | 89.5% |
| 1024 | 16 | 1,089,892 | 9.12x | 57.0% |
| 1024 | 32 | 1,442,141 | 12.07x | 37.7% |
