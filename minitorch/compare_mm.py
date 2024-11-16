import time
from typing import Dict, List
import random
import matplotlib.pyplot as plt
import minitorch
from minitorch import TensorBackend


def benchmark_matrix_multiplication(
    matrix_sizes: List[int], backend: TensorBackend
) -> List[float]:
    """Benchmark matrix multiplication for the given backend."""
    times = []
    for N in matrix_sizes:
        print(f"Matrix size: {N}x{N}")
        # Initialize NxN tensors with random values
        x1 = [[random.random() for _ in range(N)] for _ in range(N)]
        y1 = [[random.random() for _ in range(N)] for _ in range(N)]
        x = minitorch.tensor(x1, backend=backend)
        y = minitorch.tensor(y1, backend=backend)

        # Start timer
        start_time = time.time()

        # Perform tensor matrix multiplication
        result = x @ y

        # Stop timer and record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    return times


def plot_and_save_results(
    matrix_sizes: List[int], cpu_times: List[float], gpu_times: List[float]
) -> None:
    """Plot and save the benchmarking results."""
    # Plot CPU and GPU performance
    plt.plot([m*m for m in matrix_sizes], cpu_times, label="CPU (Intel Xeon Platinum 8358)")
    plt.plot([m*m for m in matrix_sizes], gpu_times, label="GPU (NVIDIA H100)")
    plt.xlabel("Matrix Size (NxN)")
    plt.ticklabel_format(style='plain')
    plt.ylabel("Time (seconds)")
    plt.title("Performance of Matrix Multiplication (CPU vs GPU)")
    plt.grid()
    plt.legend()

    # Save the figure with high DPI
    output_filename = "../matrix_multiplication_performance.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")


def main():

    matrix_sizes = [50, 100, 200, 500, 1000, 2500]

    # Set up tensor backends for CPU and GPU
    FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
    CudaTensorBackend = minitorch.TensorBackend(minitorch.CudaOps)
    shared: Dict[str, TensorBackend] = {
        "fast": FastTensorBackend,
        "cuda": CudaTensorBackend,
    }

    # Benchmark matrix multiplication on CPU
    print("Benchmarking CPU...")
    cpu_times = benchmark_matrix_multiplication(matrix_sizes, shared["fast"])

    # Benchmark matrix multiplication on GPU
    print("Benchmarking GPU...")
    gpu_times = benchmark_matrix_multiplication(matrix_sizes, shared["cuda"])

    # Plot and save results
    plot_and_save_results(matrix_sizes, cpu_times, gpu_times)

    # Print results
    print("Matrix sizes (NxN):", [m*m for m in matrix_sizes])
    print("CPU (Intel® Xeon® Platinum 8358) times:", cpu_times)
    print("GPU (NVIDIA H100) times:", gpu_times)


if __name__ == "__main__":
    main()