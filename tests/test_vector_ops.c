#include "../include/kisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#ifdef __aarch64__
#include <arm_neon.h>

// Helper functions for lane access
static int32_t get_lane_s32(int32x4_t vec, int lane) {
    switch (lane) {
        case 0: return vgetq_lane_s32(vec, 0);
        case 1: return vgetq_lane_s32(vec, 1);
        case 2: return vgetq_lane_s32(vec, 2);
        case 3: return vgetq_lane_s32(vec, 3);
        default: return 0;
    }
}

static int32x4_t set_lane_s32(int32_t val, int32x4_t vec, int lane) {
    switch (lane) {
        case 0: return vsetq_lane_s32(val, vec, 0);
        case 1: return vsetq_lane_s32(val, vec, 1);
        case 2: return vsetq_lane_s32(val, vec, 2);
        case 3: return vsetq_lane_s32(val, vec, 3);
        default: return vec;
    }
}
#endif

// 辅助函数: 打印向量内容
void print_vector(const char* name, const vector_reg_t* v) {
    printf("%s: [", name);
    #ifdef __aarch64__
    for(int i = 0; i < 4; i++) {
        printf("%d ", get_lane_s32(v->low, i));
    }
    for(int i = 0; i < 4; i++) {
        printf("%d%s", get_lane_s32(v->high, i), i < 3 ? ", " : "");
    }
    #else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        printf("%d%s", (*v)[i], i < VECTOR_LENGTH-1 ? ", " : "");
    }
    #endif
    printf("]\n");
}

// 测试基本向量运算
void test_basic_ops() {
    printf("\n=== Testing Basic Vector Operations ===\n");
    
    vector_reg_t a, b, result;
    
    // 初始化测试数据
    #ifdef __aarch64__
    a.low = vdupq_n_s32(2);
    a.high = vdupq_n_s32(2);
    b.low = vdupq_n_s32(3);
    b.high = vdupq_n_s32(3);
    #else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        a[i] = 2;
        b[i] = 3;
    }
    #endif
    
    // 测试加法
    vector_add(&result, &a, &b);
    print_vector("Add result", &result);
    
    // 测试减法
    vector_sub(&result, &a, &b);
    print_vector("Sub result", &result);
    
    // 测试乘法
    vector_mul(&result, &a, &b);
    print_vector("Mul result", &result);
    
    // 测试除法
    vector_div(&result, &a, &b);
    print_vector("Div result", &result);
}

// 测试 FFT/IFFT
void test_fft() {
    printf("\n=== Testing FFT/IFFT ===\n");
    
    vector_reg_t input, fft_result, ifft_result;
    
    // 初始化输入数据为简单的信号
    #ifdef __aarch64__
    input.low = vdupq_n_s32(0);
    input.high = vdupq_n_s32(0);
    for(int i = 0; i < 4; i++) {
        input.low = set_lane_s32(i + 1, input.low, i);
        input.high = set_lane_s32(8 - i, input.high, i);
    }
    #else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        input[i] = i + 1;
    }
    #endif
    
    print_vector("Input signal", &input);
    
    // 执行 FFT
    vector_fft(&fft_result, &input);
    print_vector("FFT result", &fft_result);
    
    // 执行 IFFT
    vector_ifft(&ifft_result, &fft_result);
    print_vector("IFFT result (should match input)", &ifft_result);
}

// 测试排序
void test_sort() {
    printf("\n=== Testing Sort ===\n");
    
    vector_reg_t input, sorted;
    
    // 初始化乱序数据
    #ifdef __aarch64__
    int32_t data[] = {5, 2, 8, 1, 9, 3, 7, 4};
    input.low = vdupq_n_s32(0);
    input.high = vdupq_n_s32(0);
    for(int i = 0; i < 4; i++) {
        input.low = set_lane_s32(data[i], input.low, i);
        input.high = set_lane_s32(data[i+4], input.high, i);
    }
    #else
    int32_t data[] = {5, 2, 8, 1, 9, 3, 7, 4};
    memcpy(&input, data, sizeof(data));
    #endif
    
    print_vector("Input array", &input);
    
    // 执行排序
    vector_sort(&sorted, &input);
    print_vector("Sorted array", &sorted);
}

// 测试归约操作
void test_reduction() {
    printf("\n=== Testing Reduction Operations ===\n");
    
    vector_reg_t input;
    
    // 初始化测试数据
    #ifdef __aarch64__
    int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    input.low = vdupq_n_s32(0);
    input.high = vdupq_n_s32(0);
    for(int i = 0; i < 4; i++) {
        input.low = set_lane_s32(data[i], input.low, i);
        input.high = set_lane_s32(data[i+4], input.high, i);
    }
    #else
    int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(&input, data, sizeof(data));
    #endif
    
    print_vector("Input array", &input);
    
    // 测试各种归约操作
    printf("Sum reduction: %d\n", vector_reduce(&input, RED_SUM));
    printf("Product reduction: %d\n", vector_reduce(&input, RED_PROD));
    printf("Max reduction: %d\n", vector_reduce(&input, RED_MAX));
    printf("Min reduction: %d\n", vector_reduce(&input, RED_MIN));
}

// 测试前缀扫描
void test_scan() {
    printf("\n=== Testing Scan Operations ===\n");
    
    vector_reg_t input, result;
    
    // 初始化测试数据
    #ifdef __aarch64__
    int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    input.low = vdupq_n_s32(0);
    input.high = vdupq_n_s32(0);
    for(int i = 0; i < 4; i++) {
        input.low = set_lane_s32(data[i], input.low, i);
        input.high = set_lane_s32(data[i+4], input.high, i);
    }
    #else
    int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(&input, data, sizeof(data));
    #endif
    
    print_vector("Input array", &input);
    
    // 测试各种扫描操作
    vector_scan(&result, &input, RED_SUM);
    print_vector("Sum scan", &result);
    
    vector_scan(&result, &input, RED_PROD);
    print_vector("Product scan", &result);
    
    vector_scan(&result, &input, RED_MAX);
    print_vector("Max scan", &result);
    
    vector_scan(&result, &input, RED_MIN);
    print_vector("Min scan", &result);
}

int main() {
    printf("Testing K-ISA Vector Operations\n");
    printf("Vector length: %d\n", VECTOR_LENGTH);
    printf("Platform: %s\n\n", 
    #ifdef __aarch64__
        "ARM NEON"
    #else
        "Scalar fallback"
    #endif
    );
    
    test_basic_ops();
    test_fft();
    test_sort();
    test_reduction();
    test_scan();
    
    printf("\nAll tests completed.\n");
    return 0;
} 