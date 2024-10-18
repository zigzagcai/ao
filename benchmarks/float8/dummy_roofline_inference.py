import torch
import torchao
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from torchao.quantization.quant_api import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only
import copy
from utils import (
    get_name_to_shapes_iter,
)
import tqdm
from tabulate import tabulate

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.linear1 = torch.nn.Linear(k, n, bias=False).to(dtype)

    def example_inputs(self, m=1, device="cuda"):
        return (torch.randn(m, self.linear1.in_features, dtype=self.dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        return x

# Function to benchmark model evaluation with profiling
def benchmark_model_with_profiling(model, input_data, dtype):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        # with record_function("model_inference"):
        for _ in range(5):  # Run the model multiple times to warm up the cache
            with torch.no_grad():
                _ = model(*input_data)
                torch.cuda.synchronize()

    # Return the profiler output
    return prof


def get_gpu_kernel_times(profiler_chrome_trace, gpu_op_name):
    # Filter CUDA events
    event_data = [(event.key, event.device_time)
                  for event in profiler_chrome_trace.key_averages()
                  if event.device_type == torch.autograd.DeviceType.CUDA]

    # Calculate overhead time and op time
    gpu_op_time, gpu_overhead_time = 0, 0
    for event in event_data:
        if gpu_op_name in event[0]:
            gpu_op_time += event[1]
        else:
            gpu_overhead_time += event[1]
    return gpu_op_time, gpu_overhead_time

def run_gemm_benchmarks(name_to_shapes, float8_dtype=torch.float8_e4m3fn, other_dtype=torch.bfloat16, quantization_technique=float8_weight_only):
    # Dictionary to store performance data
    performance_data = {
        'Input Size': [],
        'float8 Op Kernel Times (ms)': [],
        'bf16 Op Kernel Times (ms)': [],
        'float8 Overhead Kernel Times (ms)': [],
        'bf16 Overhead Kernel Times (ms)': [],
        'float8 Total Kernel Times (ms)': [],
        'bf16 Total Kernel Times (ms)': [],
    }
    # Run benchmarks for each input size
    for idx, (name, (m, k, n)) in enumerate(tqdm.tqdm(name_to_shapes)):
        print(f"Profiling model with input size: {m, k, n} for quantization technique: {quantization_technique}, dtype: {float8_dtype} vs {other_dtype}")

        # Initialize the model with the specified dimensions
        model = ToyLinearModel(m, k, n).eval().to(device)
        example_inputs = model.example_inputs(m)
        model_bf16 = copy.deepcopy(model).to(device)  # Copy the model to bf
        model_ref = copy.deepcopy(model).to(device)  # Copy the model for quantization
        quantize_(model_ref, quantization_technique())  # Quantize model to float8

        # Profile float8 model evaluation
        prof_float8 = benchmark_model_with_profiling(model_ref, example_inputs, float8_dtype)
        prof_float8.export_chrome_trace(f"fp8_model_{example_inputs[0].size()[0]}.json")  # Save profiling details

        # Profile bf16 model evaluation
        prof_bf16 = benchmark_model_with_profiling(model_bf16, example_inputs, other_dtype)
        prof_bf16.export_chrome_trace(f"bf16_model_{example_inputs[0].size()[0]}.json")  # Save profiling details

        # Calculate and store GPU kernel times -> op time, overhead time
        float8_gpu_op_time, float8_gpu_overhead_time = get_gpu_kernel_times(prof_float8, 'gemm')
        bf16_gpu_op_time, bf16_gpu_overhead_time = get_gpu_kernel_times(prof_bf16, 'gemm')

        # # Print profiling details
        # print(f"bfloat16_gpu_overhead_time: {bf16_gpu_overhead_time} gpu_op_time: {bf16_gpu_op_time}")
        # print(f"float8_gpu_overhead_time: {float8_gpu_overhead_time} float8_gpu_op_time: {float8_gpu_op_time}")

        # Add the performance data to the dictionary
        # time/1000 -> Convert from microseconds to milliseconds
        performance_data['Input Size'].append(f"{tuple(example_inputs[0].shape)}")
        performance_data['float8 Total Kernel Times (ms)'].append((float8_gpu_op_time + float8_gpu_overhead_time) / 1000)
        performance_data['bf16 Total Kernel Times (ms)'].append((bf16_gpu_op_time + bf16_gpu_overhead_time) / 1000)
        performance_data['float8 Op Kernel Times (ms)'].append(float8_gpu_op_time / 1000)
        performance_data['bf16 Op Kernel Times (ms)'].append(bf16_gpu_op_time / 1000)
        performance_data['float8 Overhead Kernel Times (ms)'].append(float8_gpu_overhead_time / 1000)
        performance_data['bf16 Overhead Kernel Times (ms)'].append(bf16_gpu_overhead_time / 1000)

    return performance_data


def plot_performance_data(performance_data):
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(performance_data['Input Size'], performance_data['float8 Total Kernel Times (ms)'], marker='o', label='float8')
    plt.plot(performance_data['Input Size'], performance_data['bf16 Total Kernel Times (ms)'], marker='s', label='bf16')
    plt.xlabel('Batch Size')
    plt.ylabel('Kernel Time (ms)')
    plt.title('Model Evaluation GPU Kernel Performance: float8 vs bf16')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_evaluation_gpu_kernel_performance.png')


if __name__ == '__main__':
    # Set the data types
    name_to_shapes = get_name_to_shapes_iter("square", None, None, None)
    float8_dtype = torch.float8_e4m3fn  # Change to the float8 dtype you want to use
    bf16_dtype = torch.bfloat16  # Change to the comparing dtype you want to use
    quantization_technique = float8_weight_only  # Change to the quantization technique you want to use

    performance_data = run_gemm_benchmarks(
                            name_to_shapes=name_to_shapes,
                            float8_dtype=float8_dtype,
                            other_dtype=bf16_dtype,
                            quantization_technique=quantization_technique
                        )
    print('Performance data: \n', tabulate(performance_data, headers=performance_data.keys()))
