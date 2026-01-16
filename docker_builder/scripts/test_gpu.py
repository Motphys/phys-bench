#!/usr/bin/env python3
"""
GPU test script for multi-simulator environment
Tests GPU availability and compute capability for each simulator
"""

import sys
from typing import Dict, Any


def test_cuda_availability() -> Dict[str, Any]:
    """Test basic CUDA availability"""
    print("=" * 60)
    print("CUDA Availability Test")
    print("=" * 60)

    result = {
        "available": False,
        "device_count": 0,
        "devices": []
    }

    try:
        import torch
        result["available"] = torch.cuda.is_available()
        result["device_count"] = torch.cuda.device_count()

        if result["available"]:
            for i in range(result["device_count"]):
                cap = torch.cuda.get_device_capability(i)
                device_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": f"{cap[0]}.{cap[1]}",
                    "sm": f"sm_{cap[0]}{cap[1]}",
                    "memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9
                }
                result["devices"].append(device_info)

                print(f"\nGPU {i}:")
                print(f"  Name: {device_info['name']}")
                print(f"  Compute Capability: {device_info['compute_capability']} ({device_info['sm']})")
                print(f"  Memory: {device_info['memory_gb']:.2f} GB")

            print(f"\n✓ CUDA is available with {result['device_count']} device(s)")
        else:
            print("\n✗ CUDA is not available")

    except ImportError:
        print("\n✗ PyTorch not installed")
    except Exception as e:
        print(f"\n✗ Error: {e}")

    return result


def test_isaac_gym():
    """Test Isaac Gym environment"""
    print("\n" + "=" * 60)
    print("Isaac Gym Environment Test (Python 3.8)")
    print("=" * 60)

    try:
        import sys
        print(f"Python version: {sys.version}")

        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            print(f"Compute capability: {cap[0]}.{cap[1]}")

        try:
            import isaacgym
            print(f"Isaac Gym: Installed ✓")
        except ImportError:
            print(f"Isaac Gym: Not installed (optional)")

        print("\n✓ Isaac Gym environment OK")
        return True

    except Exception as e:
        print(f"\n✗ Isaac Gym environment error: {e}")
        return False


def test_genesis():
    """Test Genesis environment"""
    print("\n" + "=" * 60)
    print("Genesis Environment Test (Python 3.10)")
    print("=" * 60)

    try:
        import sys
        print(f"Python version: {sys.version}")

        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            print(f"Compute capability: {cap[0]}.{cap[1]}")

        try:
            import genesis as gs
            print(f"Genesis version: {gs.__version__}")
        except ImportError:
            print(f"Genesis: Not installed (run 'uv add genesis-world')")

        print("\n✓ Genesis environment OK")
        return True

    except Exception as e:
        print(f"\n✗ Genesis environment error: {e}")
        return False


def test_mjx():
    """Test MJX environment"""
    print("\n" + "=" * 60)
    print("MJX Environment Test (Python 3.10)")
    print("=" * 60)

    try:
        import sys
        print(f"Python version: {sys.version}")

        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        print(f"Default backend: {jax.default_backend()}")

        try:
            import mujoco
            print(f"MuJoCo version: {mujoco.__version__}")

            from mujoco import mjx
            print(f"MJX: Available ✓")
        except ImportError as e:
            print(f"MuJoCo/MJX: Not installed (run 'uv add mujoco mujoco-mjx')")

        print("\n✓ MJX environment OK")
        return True

    except Exception as e:
        print(f"\n✗ MJX environment error: {e}")
        return False


def test_performance():
    """Run basic GPU performance test"""
    print("\n" + "=" * 60)
    print("GPU Performance Test")
    print("=" * 60)

    try:
        import torch
        import time

        if not torch.cuda.is_available():
            print("CUDA not available, skipping performance test")
            return

        device = torch.device("cuda:0")
        size = 8192

        # Warmup
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        flops = 2 * size**3 * 10  # matrix multiply FLOPS
        tflops = flops / elapsed / 1e12

        print(f"\nMatrix multiplication ({size}x{size}):")
        print(f"  Time: {elapsed:.3f} seconds (10 iterations)")
        print(f"  Performance: {tflops:.2f} TFLOPS")

        print("\n✓ Performance test completed")

    except Exception as e:
        print(f"\n✗ Performance test error: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Multi-Simulator GPU Environment Test")
    print("=" * 60)
    print()

    # Test CUDA
    cuda_result = test_cuda_availability()

    if not cuda_result["available"]:
        print("\n" + "=" * 60)
        print("WARNING: No CUDA devices detected!")
        print("=" * 60)
        print("Possible issues:")
        print("1. GPU not passed to container (use --gpus all)")
        print("2. NVIDIA driver not installed on host")
        print("3. NVIDIA Container Toolkit not installed")
        print()
        print("To fix:")
        print("  docker run --gpus all ...")
        sys.exit(1)

    # Test each environment
    print("\n" + "=" * 60)
    print("Testing Individual Environments")
    print("=" * 60)
    print("Note: Run these tests within each environment for full validation:")
    print("  run-isaacgym /workspace/scripts/test_gpu.py")
    print("  run-genesis /workspace/scripts/test_gpu.py")
    print("  run-mjx /workspace/scripts/test_gpu.py")

    # Run performance test
    test_performance()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✓ CUDA devices: {cuda_result['device_count']}")
    for device in cuda_result["devices"]:
        print(f"  - {device['name']} ({device['sm']})")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
