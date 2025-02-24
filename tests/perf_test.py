import numpy as np
import time

VECTOR_LENGTH = 8
NUM_ITERATIONS = 10000
WARMUP_ITERATIONS = 100

def get_time_ns():
    return time.time_ns()

def init_vector(start):
    return np.arange(start, start + VECTOR_LENGTH, dtype=np.int32)

# 辅助函数：位反转置换
def bit_reverse(x, bits):
    result = 0
    for i in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

# 高精度 FFT 实现
def high_precision_fft(input_data):
    # 将输入数据转换为64位整数并左移16位以提供更高精度
    data_64 = input_data.astype(np.int64) << 16
    
    # 位反转置换
    temp = np.zeros_like(data_64)
    for i in range(VECTOR_LENGTH):
        reversed_idx = bit_reverse(i, 3)  # 3 bits for 8 elements
        temp[reversed_idx] = data_64[i]
    
    # 8点FFT的三个阶段
    for stage in range(3):
        distance = 1 << stage
        butterfly = 1 << (stage + 1)
        
        for i in range(0, VECTOR_LENGTH, butterfly):
            for j in range(distance):
                a_idx = i + j
                b_idx = i + j + distance
                
                # 计算旋转因子
                angle = -2.0 * np.pi * j / butterfly
                twiddle_real = int(np.cos(angle) * 32768 * 65536)
                twiddle_imag = int(np.sin(angle) * 32768 * 65536)
                
                # 蝶形运算
                temp_real = (data_64[b_idx] * twiddle_real) >> 31
                temp_imag = (data_64[b_idx] * twiddle_imag) >> 31
                
                a_val = data_64[a_idx]
                data_64[b_idx] = a_val - temp_real
                data_64[a_idx] = a_val + temp_real
    
    # 使用四舍五入转换回32位整数
    result = np.zeros_like(input_data)
    for i in range(VECTOR_LENGTH):
        half = 1 << 15
        result[i] = int((data_64[i] + half) >> 16)
    
    return result

# 高精度 IFFT 实现
def high_precision_ifft(input_data):
    # 将输入数据转换为64位整数并左移16位以提供更高精度
    data_64 = input_data.astype(np.int64) << 16
    
    # 位反转置换
    temp = np.zeros_like(data_64)
    for i in range(VECTOR_LENGTH):
        reversed_idx = bit_reverse(i, 3)  # 3 bits for 8 elements
        temp[reversed_idx] = data_64[i]
    
    # 8点IFFT的三个阶段
    for stage in range(3):
        distance = 1 << stage
        butterfly = 1 << (stage + 1)
        
        for i in range(0, VECTOR_LENGTH, butterfly):
            for j in range(distance):
                a_idx = i + j
                b_idx = i + j + distance
                
                # 计算旋转因子（注意IFFT使用正角度）
                angle = 2.0 * np.pi * j / butterfly
                twiddle_real = int(np.cos(angle) * 32768 * 65536)
                twiddle_imag = int(np.sin(angle) * 32768 * 65536)
                
                # 蝶形运算
                temp_real = (data_64[b_idx] * twiddle_real) >> 31
                temp_imag = (data_64[b_idx] * twiddle_imag) >> 31
                
                a_val = data_64[a_idx]
                data_64[b_idx] = a_val - temp_real
                data_64[a_idx] = a_val + temp_real
    
    # 归一化并使用四舍五入转换回32位整数
    result = np.zeros_like(input_data)
    for i in range(VECTOR_LENGTH):
        data_64[i] = data_64[i] >> 3  # 归一化
        half = 1 << 15
        result[i] = int((data_64[i] + half) >> 16)
    
    return result

def test_arithmetic_perf():
    print("\n=== Testing Arithmetic Operations Performance ===")
    a = init_vector(1)
    b = init_vector(2)
    result = np.zeros(VECTOR_LENGTH, dtype=np.int32)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        result = a + b
        result = a - b
        result = a * b
        result = a // b  # Using integer division to match C version
    
    # Test addition
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        result = a + b
    total_time = get_time_ns() - start_time
    print(f"Add: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    
    # Test subtraction
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        result = a - b
    total_time = get_time_ns() - start_time
    print(f"Sub: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    
    # Test multiplication
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        result = a * b
    total_time = get_time_ns() - start_time
    print(f"Mul: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    
    # Test division
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        result = a // b
    total_time = get_time_ns() - start_time
    print(f"Div: {total_time/NUM_ITERATIONS:.2f} ns per operation")

def test_fft_perf():
    print("\n=== Testing FFT/IFFT Performance ===")
    # 使用与 C 测试相同的输入数据
    input_data = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    fft_result = np.zeros(VECTOR_LENGTH, dtype=np.int32)
    ifft_result = np.zeros(VECTOR_LENGTH, dtype=np.int32)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        fft_result = high_precision_fft(input_data)
        ifft_result = high_precision_ifft(fft_result)
    
    # Test FFT
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        fft_result = high_precision_fft(input_data)
    total_time = get_time_ns() - start_time
    print(f"FFT: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    print("FFT result:", fft_result)
    
    # Test IFFT
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        ifft_result = high_precision_ifft(fft_result)
    total_time = get_time_ns() - start_time
    print(f"IFFT: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    print("IFFT result:", ifft_result)
    
    # 验证结果
    diffs = np.abs(ifft_result - input_data)
    print("Differences:", diffs)
    assert np.all(diffs <= 3), "IFFT(FFT(x)) should approximately equal x"

def test_sort_perf():
    print("\n=== Testing Sort Performance ===")
    input_data = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
    result = np.zeros(VECTOR_LENGTH, dtype=np.int32)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        result = np.sort(input_data)
    
    # Test sort
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        result = np.sort(input_data)
    total_time = get_time_ns() - start_time
    print(f"Sort: {total_time/NUM_ITERATIONS:.2f} ns per operation")

def test_reduction_perf():
    print("\n=== Testing Reduction Operations Performance ===")
    input_data = init_vector(1)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        sum_result = np.sum(input_data)
        prod_result = np.prod(input_data)
        max_result = np.max(input_data)
        min_result = np.min(input_data)
    
    # Test sum reduction
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        sum_result = np.sum(input_data)
    total_time = get_time_ns() - start_time
    print(f"Sum reduction: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    
    # Test product reduction
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        prod_result = np.prod(input_data)
    total_time = get_time_ns() - start_time
    print(f"Product reduction: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    
    # Test max reduction
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        max_result = np.max(input_data)
    total_time = get_time_ns() - start_time
    print(f"Max reduction: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    
    # Test min reduction
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        min_result = np.min(input_data)
    total_time = get_time_ns() - start_time
    print(f"Min reduction: {total_time/NUM_ITERATIONS:.2f} ns per operation")

def test_scan_perf():
    print("\n=== Testing Scan Operations Performance ===")
    input_data = init_vector(1)
    result = np.zeros(VECTOR_LENGTH, dtype=np.int32)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        result = np.cumsum(input_data)
        result = np.maximum.accumulate(input_data)
        result = np.minimum.accumulate(input_data)
    
    # Test sum scan
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        result = np.cumsum(input_data)
    total_time = get_time_ns() - start_time
    print(f"Sum scan: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    
    # Test max scan
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        result = np.maximum.accumulate(input_data)
    total_time = get_time_ns() - start_time
    print(f"Max scan: {total_time/NUM_ITERATIONS:.2f} ns per operation")
    
    # Test min scan
    start_time = get_time_ns()
    for _ in range(NUM_ITERATIONS):
        result = np.minimum.accumulate(input_data)
    total_time = get_time_ns() - start_time
    print(f"Min scan: {total_time/NUM_ITERATIONS:.2f} ns per operation")

if __name__ == "__main__":
    print("Running K-ISA Performance Tests")
    print(f"Vector length: {VECTOR_LENGTH}")
    print(f"Number of iterations: {NUM_ITERATIONS}")
    print(f"Warmup iterations: {WARMUP_ITERATIONS}")
    
    test_arithmetic_perf()
    test_fft_perf()
    test_sort_perf()
    test_reduction_perf()
    test_scan_perf()
    
    print("\nAll performance tests completed successfully!") 