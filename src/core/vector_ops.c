/**
 * vector_ops.c - Vector operations implementation for K-ISA
 */

#include "../../include/kisa.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// 位反转置换辅助函数
static uint32_t reverse_bits_helper(uint32_t x, int bits) {
    uint32_t result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// 向量位反转置换函数声明
static void bit_reverse(vector_reg_t* result, vector_reg_t* input);

#ifdef __aarch64__
// Helper functions for NEON lane access
static inline int32_t get_lane_value_low(int32x4_t vec, int index) {
    switch(index) {
        case 0: return vgetq_lane_s32(vec, 0);
        case 1: return vgetq_lane_s32(vec, 1);
        case 2: return vgetq_lane_s32(vec, 2);
        case 3: return vgetq_lane_s32(vec, 3);
        default: return 0;
    }
}

static inline int32x4_t set_lane_value_low(int32x4_t vec, int32_t value, int index) {
    switch(index) {
        case 0: return vsetq_lane_s32(value, vec, 0);
        case 1: return vsetq_lane_s32(value, vec, 1);
        case 2: return vsetq_lane_s32(value, vec, 2);
        case 3: return vsetq_lane_s32(value, vec, 3);
        default: return vec;
    }
}
#endif

// 向量位反转置换实现
static void bit_reverse(vector_reg_t* result, vector_reg_t* input) {
    #ifdef __aarch64__
    int32_t data[8];
    vst1q_s32(data, input->low);
    vst1q_s32(data + 4, input->high);
    
    int32_t temp[8];
    for (int i = 0; i < 8; i++) {
        int reversed_idx = reverse_bits_helper(i, 3);  // 3 bits for 8 elements
        temp[reversed_idx] = data[i];
    }
    
    result->low = vld1q_s32(temp);
    result->high = vld1q_s32(temp + 4);
    #else
    int32_t temp[VECTOR_LENGTH];
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        int reversed_idx = reverse_bits_helper(i, 3);  // 3 bits for 8 elements
        temp[reversed_idx] = (*input)[i];
    }
    memcpy(*result, temp, sizeof(temp));
    #endif
}

// 辅助函数：带四舍五入的位移
static inline int32_t round_shift(int64_t value, int shift) {
    int64_t half = (int64_t)1 << (shift - 1);
    return (int32_t)((value + half) >> shift);
}

// 向量操作实现
void vector_add(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b) {
    #ifdef __aarch64__
    result->low = vaddq_s32(a->low, b->low);
    result->high = vaddq_s32(a->high, b->high);
    #else
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = (*a)[i] + (*b)[i];
    }
    #endif
}

void vector_sub(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b) {
    #ifdef __aarch64__
    result->low = vsubq_s32(a->low, b->low);
    result->high = vsubq_s32(a->high, b->high);
    #else
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = (*a)[i] - (*b)[i];
    }
    #endif
}

void vector_mul(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b) {
    #ifdef __aarch64__
    result->low = vmulq_s32(a->low, b->low);
    result->high = vmulq_s32(a->high, b->high);
    #else
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = (*a)[i] * (*b)[i];
    }
    #endif
}

void vector_div(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b) {
    #ifdef __aarch64__
    int32_t a_vals[8], b_vals[8];
    
    // 提取值
    vst1q_s32(a_vals, a->low);
    vst1q_s32(a_vals + 4, a->high);
    vst1q_s32(b_vals, b->low);
    vst1q_s32(b_vals + 4, b->high);
    
    // 执行除法
    for (int i = 0; i < 8; i++) {
        if (b_vals[i] != 0) {
            a_vals[i] /= b_vals[i];
        }
    }
    
    // 加载回NEON寄存器
    result->low = vld1q_s32(a_vals);
    result->high = vld1q_s32(a_vals + 4);
    #else
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        if ((*b)[i] != 0) {
            (*result)[i] = (*a)[i] / (*b)[i];
        }
    }
    #endif
}

void vector_fft(vector_reg_t* result, vector_reg_t* input) {
    vector_reg_t temp;
    bit_reverse(&temp, input);
    
    #ifdef __aarch64__
    int32_t data[8];
    vst1q_s32(data, temp.low);
    vst1q_s32(data + 4, temp.high);
    
    // 使用64位整数进行中间计算以减少误差
    int64_t data_64[8];
    for (int i = 0; i < 8; i++) {
        data_64[i] = (int64_t)data[i] << 16;  // 左移16位以提供更多精度
    }
    
    // 8点FFT的三个阶段
    for (int stage = 0; stage < 3; stage++) {
        int distance = 1 << stage;
        int butterfly = 1 << (stage + 1);
        
        for (int i = 0; i < VECTOR_LENGTH; i += butterfly) {
            for (int j = 0; j < distance; j++) {
                int a_idx = i + j;
                int b_idx = i + j + distance;
                
                // 计算旋转因子
                double angle = -2.0 * M_PI * j / butterfly;
                int64_t twiddle_real = (int64_t)(cos(angle) * 32768 * 65536);  // 更高精度
                int64_t twiddle_imag = (int64_t)(sin(angle) * 32768 * 65536);  // 更高精度
                
                // 蝶形运算
                int64_t temp_real = (data_64[b_idx] * twiddle_real) >> 31;  // 31 = 15 + 16
                int64_t temp_imag = (data_64[b_idx] * twiddle_imag) >> 31;  // 31 = 15 + 16
                
                int64_t a_val = data_64[a_idx];
                data_64[b_idx] = a_val - temp_real;
                data_64[a_idx] = a_val + temp_real;
            }
        }
    }
    
    // 转换回32位整数，使用四舍五入
    for (int i = 0; i < 8; i++) {
        data[i] = round_shift(data_64[i], 16);  // 使用四舍五入
    }
    
    result->low = vld1q_s32(data);
    result->high = vld1q_s32(data + 4);
    #else
    memcpy(*result, temp, sizeof(vector_reg_t));
    #endif
}

void vector_ifft(vector_reg_t* result, vector_reg_t* input) {
    vector_reg_t temp;
    bit_reverse(&temp, input);
    
    #ifdef __aarch64__
    int32_t data[8];
    vst1q_s32(data, temp.low);
    vst1q_s32(data + 4, temp.high);
    
    // 使用64位整数进行中间计算以减少误差
    int64_t data_64[8];
    for (int i = 0; i < 8; i++) {
        data_64[i] = (int64_t)data[i] << 16;  // 左移16位以提供更多精度
    }
    
    // 8点IFFT的三个阶段
    for (int stage = 0; stage < 3; stage++) {
        int distance = 1 << stage;
        int butterfly = 1 << (stage + 1);
        
        for (int i = 0; i < VECTOR_LENGTH; i += butterfly) {
            for (int j = 0; j < distance; j++) {
                int a_idx = i + j;
                int b_idx = i + j + distance;
                
                // 计算旋转因子（注意IFFT使用正角度）
                double angle = 2.0 * M_PI * j / butterfly;
                int64_t twiddle_real = (int64_t)(cos(angle) * 32768 * 65536);  // 更高精度
                int64_t twiddle_imag = (int64_t)(sin(angle) * 32768 * 65536);  // 更高精度
                
                // 蝶形运算
                int64_t temp_real = (data_64[b_idx] * twiddle_real) >> 31;  // 31 = 15 + 16
                int64_t temp_imag = (data_64[b_idx] * twiddle_imag) >> 31;  // 31 = 15 + 16
                
                int64_t a_val = data_64[a_idx];
                data_64[b_idx] = a_val - temp_real;
                data_64[a_idx] = a_val + temp_real;
            }
        }
    }
    
    // 转换回32位整数并进行归一化，使用四舍五入
    for (int i = 0; i < 8; i++) {
        data_64[i] = data_64[i] >> 3;  // 先进行归一化
        data[i] = round_shift(data_64[i], 16);  // 使用四舍五入
    }
    
    result->low = vld1q_s32(data);
    result->high = vld1q_s32(data + 4);
    #else
    memcpy(*result, temp, sizeof(vector_reg_t));
    #endif
}

void vector_sort(vector_reg_t* result, vector_reg_t* input) {
    #ifdef __aarch64__
    int32_t data[8];
    vst1q_s32(data, input->low);
    vst1q_s32(data + 4, input->high);
    
    // 8元素双调排序网络
    for (int i = 0; i < 4; i++) {
        if (data[i] > data[i + 4]) {
            int32_t temp = data[i];
            data[i] = data[i + 4];
            data[i + 4] = temp;
        }
    }
    
    // 对每个四元素子序列排序
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int idx1 = i * 4 + j * 2;
            int idx2 = idx1 + 1;
            if (data[idx1] > data[idx2]) {
                int32_t temp = data[idx1];
                data[idx1] = data[idx2];
                data[idx2] = temp;
            }
        }
    }
    
    // 最后的比较和交换
    for (int i = 1; i < 7; i += 2) {
        if (data[i] > data[i + 1]) {
            int32_t temp = data[i];
            data[i] = data[i + 1];
            data[i + 1] = temp;
        }
    }
    
    result->low = vld1q_s32(data);
    result->high = vld1q_s32(data + 4);
    #else
    memcpy(*result, *input, sizeof(vector_reg_t));
    #endif
}

int32_t vector_reduce(vector_reg_t* input, reduction_op_t op) {
    int32_t result;
    
    #ifdef __aarch64__
    int32_t data[8];
    vst1q_s32(data, input->low);
    vst1q_s32(data + 4, input->high);
    
    switch(op) {
        case RED_SUM:
            result = 0;
            for (int i = 0; i < 8; i++) {
                result += data[i];
            }
            break;
            
        case RED_PROD:
            result = 1;
            for (int i = 0; i < 8; i++) {
                result *= data[i];
            }
            break;
            
        case RED_MAX:
            result = data[0];
            for (int i = 1; i < 8; i++) {
                if (data[i] > result) {
                    result = data[i];
                }
            }
            break;
            
        case RED_MIN:
            result = data[0];
            for (int i = 1; i < 8; i++) {
                if (data[i] < result) {
                    result = data[i];
                }
            }
            break;
    }
    #else
    result = (*input)[0];
    #endif
    
    return result;
}

void vector_scan(vector_reg_t* result, vector_reg_t* input, reduction_op_t op) {
    #ifdef __aarch64__
    int32_t data[8];
    vst1q_s32(data, input->low);
    vst1q_s32(data + 4, input->high);
    
    switch(op) {
        case RED_SUM:
            for (int i = 1; i < 8; i++) {
                data[i] += data[i-1];
            }
            break;
            
        case RED_PROD:
            for (int i = 1; i < 8; i++) {
                data[i] *= data[i-1];
            }
            break;
            
        case RED_MAX:
            for (int i = 1; i < 8; i++) {
                if (data[i-1] > data[i]) {
                    data[i] = data[i-1];
                }
            }
            break;
            
        case RED_MIN:
            for (int i = 1; i < 8; i++) {
                if (data[i-1] < data[i]) {
                    data[i] = data[i-1];
                }
            }
            break;
    }
    
    result->low = vld1q_s32(data);
    result->high = vld1q_s32(data + 4);
    #else
    memcpy(*result, *input, sizeof(vector_reg_t));
    #endif
}

void vector_compare(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b) {
    #ifdef __aarch64__
    result->low = vcgtq_s32(a->low, b->low);
    result->high = vcgtq_s32(a->high, b->high);
    #else
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = ((*a)[i] > (*b)[i]) ? -1 : 0;
    }
    #endif
}

void vector_select(vector_reg_t* result, vector_reg_t* mask, vector_reg_t* a, vector_reg_t* b) {
    #ifdef __aarch64__
    result->low = vbslq_s32(mask->low, a->low, b->low);
    result->high = vbslq_s32(mask->high, a->high, b->high);
    #else
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = (*mask)[i] ? (*a)[i] : (*b)[i];
    }
    #endif
}

// Print vector register contents
void print_register(const char* name, const vector_reg_t* reg) {
    if (!reg) {
        printf("%s: [NULL]\n", name);
        return;
    }
    
    printf("%s: [", name);
    
#ifdef __aarch64__
    // ARM64 NEON implementation
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        int32_t val = 0;
        if (i < 4) {
            // Use switch-case for constant indices
            switch(i) {
                case 0: val = vgetq_lane_s32(reg->low, 0); break;
                case 1: val = vgetq_lane_s32(reg->low, 1); break;
                case 2: val = vgetq_lane_s32(reg->low, 2); break;
                case 3: val = vgetq_lane_s32(reg->low, 3); break;
            }
        } else {
            // Use switch-case for constant indices
            switch(i - 4) {
                case 0: val = vgetq_lane_s32(reg->high, 0); break;
                case 1: val = vgetq_lane_s32(reg->high, 1); break;
                case 2: val = vgetq_lane_s32(reg->high, 2); break;
                case 3: val = vgetq_lane_s32(reg->high, 3); break;
            }
        }
        printf("%d%s", val, i < VECTOR_LENGTH - 1 ? ", " : "");
    }
#else
    // Generic implementation
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        printf("%d%s", (*reg)[i], i < VECTOR_LENGTH - 1 ? ", " : "");
    }
#endif
    
    printf("]\n");
}

// 状态标志
static struct {
    bool zero;
    bool negative;
    bool carry;
    bool overflow;
} flags = {false, false, false, false};

// 更新标志位
void update_flags(int32_t result, bool check_carry) {
    flags.zero = (result == 0);
    flags.negative = (result < 0);
    
    if (check_carry) {
        // 进位判断 (简化版)
        flags.carry = (result < 0);
        
        // 溢出判断 (简化版)
        flags.overflow = (result == INT32_MIN);
    }
}

// 等于时分支
bool branch_if_equal(int32_t value) {
    return flags.zero;
}

// 不等于时分支
bool branch_if_not_equal(int32_t value) {
    return !flags.zero;
}

// 大于时分支
bool branch_if_greater(int32_t value) {
    return !flags.negative && !flags.zero;
}

// 小于时分支
bool branch_if_less(int32_t value) {
    return flags.negative;
}
