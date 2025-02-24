#include "../include/kisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define NUM_ITERATIONS 10000
#define WARMUP_ITERATIONS 100

// 辅助函数：获取当前时间（纳秒）
static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// 辅助函数：初始化向量数据
static void init_vector(vector_reg_t* v, int32_t start) {
    #ifdef __aarch64__
    int32x4_t* vec_low = &v->low;
    int32x4_t* vec_high = &v->high;
    
    // Initialize low part
    *vec_low = vdupq_n_s32(0);
    *vec_high = vdupq_n_s32(0);
    
    // Set values using switch statements
    for (int i = 0; i < 4; i++) {
        switch(i) {
            case 0:
                *vec_low = vsetq_lane_s32(start + 0, *vec_low, 0);
                *vec_high = vsetq_lane_s32(start + 4, *vec_high, 0);
                break;
            case 1:
                *vec_low = vsetq_lane_s32(start + 1, *vec_low, 1);
                *vec_high = vsetq_lane_s32(start + 5, *vec_high, 1);
                break;
            case 2:
                *vec_low = vsetq_lane_s32(start + 2, *vec_low, 2);
                *vec_high = vsetq_lane_s32(start + 6, *vec_high, 2);
                break;
            case 3:
                *vec_low = vsetq_lane_s32(start + 3, *vec_low, 3);
                *vec_high = vsetq_lane_s32(start + 7, *vec_high, 3);
                break;
        }
    }
    #else
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        (*v)[i] = start + i;
    }
    #endif
}

// 测试基本算术运算性能
void test_arithmetic_perf() {
    printf("\n=== Testing Arithmetic Operations Performance ===\n");
    vector_reg_t a, b, result;
    uint64_t start_time, total_time;
    
    // 初始化测试数据
    init_vector(&a, 1);
    init_vector(&b, 2);
    
    // 预热
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        vector_add(&result, &a, &b);
        vector_sub(&result, &a, &b);
        vector_mul(&result, &a, &b);
        vector_div(&result, &a, &b);
    }
    
    // 测试加法
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_add(&result, &a, &b);
    }
    total_time = get_time_ns() - start_time;
    printf("Add: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试减法
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_sub(&result, &a, &b);
    }
    total_time = get_time_ns() - start_time;
    printf("Sub: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试乘法
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_mul(&result, &a, &b);
    }
    total_time = get_time_ns() - start_time;
    printf("Mul: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试除法
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_div(&result, &a, &b);
    }
    total_time = get_time_ns() - start_time;
    printf("Div: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
}

// 测试 FFT/IFFT 性能
void test_fft_perf() {
    printf("\n=== Testing FFT/IFFT Performance ===\n");
    vector_reg_t input, fft_result, ifft_result;
    uint64_t start_time, total_time;
    
    // 初始化测试数据
    init_vector(&input, 1);
    
    // 预热
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        vector_fft(&fft_result, &input);
        vector_ifft(&ifft_result, &fft_result);
    }
    
    // 测试 FFT
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_fft(&fft_result, &input);
    }
    total_time = get_time_ns() - start_time;
    printf("FFT: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试 IFFT
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_ifft(&ifft_result, &fft_result);
    }
    total_time = get_time_ns() - start_time;
    printf("IFFT: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
}

// 测试排序性能
void test_sort_perf() {
    printf("\n=== Testing Sort Performance ===\n");
    vector_reg_t input, sorted;
    uint64_t start_time, total_time;
    
    // 初始化测试数据
    init_vector(&input, 8);  // 使用逆序数据以测试最坏情况
    
    // 预热
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        vector_sort(&sorted, &input);
    }
    
    // 测试排序
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_sort(&sorted, &input);
    }
    total_time = get_time_ns() - start_time;
    printf("Sort: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
}

// 测试归约操作性能
void test_reduction_perf() {
    printf("\n=== Testing Reduction Operations Performance ===\n");
    vector_reg_t input;
    uint64_t start_time, total_time;
    volatile int32_t result;  // 使用 volatile 防止编译器优化
    
    // 初始化测试数据
    init_vector(&input, 1);
    
    // 预热
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        result = vector_reduce(&input, RED_SUM);
        result = vector_reduce(&input, RED_PROD);
        result = vector_reduce(&input, RED_MAX);
        result = vector_reduce(&input, RED_MIN);
    }
    
    // 测试求和归约
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result = vector_reduce(&input, RED_SUM);
    }
    total_time = get_time_ns() - start_time;
    printf("Sum reduction: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试乘积归约
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result = vector_reduce(&input, RED_PROD);
    }
    total_time = get_time_ns() - start_time;
    printf("Product reduction: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试最大值归约
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result = vector_reduce(&input, RED_MAX);
    }
    total_time = get_time_ns() - start_time;
    printf("Max reduction: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试最小值归约
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result = vector_reduce(&input, RED_MIN);
    }
    total_time = get_time_ns() - start_time;
    printf("Min reduction: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
}

// 测试扫描操作性能
void test_scan_perf() {
    printf("\n=== Testing Scan Operations Performance ===\n");
    vector_reg_t input, result;
    uint64_t start_time, total_time;
    
    // 初始化测试数据
    init_vector(&input, 1);
    
    // 预热
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        vector_scan(&result, &input, RED_SUM);
        vector_scan(&result, &input, RED_PROD);
        vector_scan(&result, &input, RED_MAX);
        vector_scan(&result, &input, RED_MIN);
    }
    
    // 测试求和扫描
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_scan(&result, &input, RED_SUM);
    }
    total_time = get_time_ns() - start_time;
    printf("Sum scan: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试乘积扫描
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_scan(&result, &input, RED_PROD);
    }
    total_time = get_time_ns() - start_time;
    printf("Product scan: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试最大值扫描
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_scan(&result, &input, RED_MAX);
    }
    total_time = get_time_ns() - start_time;
    printf("Max scan: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
    
    // 测试最小值扫描
    start_time = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_scan(&result, &input, RED_MIN);
    }
    total_time = get_time_ns() - start_time;
    printf("Min scan: %.2f ns per operation\n", (double)total_time / NUM_ITERATIONS);
}

int main() {
    printf("Performance Testing K-ISA Vector Operations\n");
    printf("Platform: %s\n", 
    #ifdef __aarch64__
        "ARM NEON"
    #else
        "Scalar fallback"
    #endif
    );
    printf("Vector length: %d\n", VECTOR_LENGTH);
    printf("Number of iterations: %d\n", NUM_ITERATIONS);
    printf("Warmup iterations: %d\n", WARMUP_ITERATIONS);
    
    test_arithmetic_perf();
    test_fft_perf();
    test_sort_perf();
    test_reduction_perf();
    test_scan_perf();
    
    return 0;
} 