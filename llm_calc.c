/**
 * llm_calc.c - 使用K-ISA实现简化版LLM计算
 * 
 * 这个程序实现了一个简化的LLM前向传播计算，包括：
 * 1. 矩阵-向量乘法（模拟线性层）
 * 2. 向量激活函数（使用ReLU）
 * 3. 注意力机制的简化版本
 * 4. 使用FFT进行序列处理
 * 5. 层归一化（Layer Normalization）
 */

#include "include/kisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// 模型参数
#define EMBEDDING_DIM 8  // 嵌入维度，与向量寄存器大小匹配
#define HIDDEN_DIM 8     // 隐藏层维度
#define SEQ_LENGTH 4     // 序列长度
#define NUM_LAYERS 2     // 层数
#define EPSILON 1e-5     // 归一化中防止除零的小值

// 辅助函数：获取向量元素
static inline int32_t get_vector_element(const vector_reg_t* reg, int i) {
#ifdef __aarch64__
    switch(i) {
        case 0: return vgetq_lane_s32(reg->low, 0);
        case 1: return vgetq_lane_s32(reg->low, 1);
        case 2: return vgetq_lane_s32(reg->low, 2);
        case 3: return vgetq_lane_s32(reg->low, 3);
        case 4: return vgetq_lane_s32(reg->high, 0);
        case 5: return vgetq_lane_s32(reg->high, 1);
        case 6: return vgetq_lane_s32(reg->high, 2);
        case 7: return vgetq_lane_s32(reg->high, 3);
        default: return 0;
    }
#else
    return (*reg)[i];
#endif
}

// 辅助函数：设置向量元素
static inline void set_vector_element(vector_reg_t* reg, int i, int32_t value) {
#ifdef __aarch64__
    switch(i) {
        case 0: reg->low = vsetq_lane_s32(value, reg->low, 0); break;
        case 1: reg->low = vsetq_lane_s32(value, reg->low, 1); break;
        case 2: reg->low = vsetq_lane_s32(value, reg->low, 2); break;
        case 3: reg->low = vsetq_lane_s32(value, reg->low, 3); break;
        case 4: reg->high = vsetq_lane_s32(value, reg->high, 0); break;
        case 5: reg->high = vsetq_lane_s32(value, reg->high, 1); break;
        case 6: reg->high = vsetq_lane_s32(value, reg->high, 2); break;
        case 7: reg->high = vsetq_lane_s32(value, reg->high, 3); break;
    }
#else
    (*reg)[i] = value;
#endif
}

// 辅助函数：打印向量内容
void print_vector(const char* name, const vector_reg_t* v) {
    printf("%s: [", name);
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        printf("%d%s", get_vector_element(v, i), i < VECTOR_LENGTH-1 ? ", " : "");
    }
    printf("]\n");
}

// 辅助函数：初始化向量
void init_vector(vector_reg_t* v, int32_t values[VECTOR_LENGTH]) {
#ifdef __aarch64__
    v->low = vdupq_n_s32(0);
    v->high = vdupq_n_s32(0);
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        set_vector_element(v, i, values[i]);
    }
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*v)[i] = values[i];
    }
#endif
}

// 辅助函数：ReLU激活函数
void relu_activation(vector_reg_t* result, vector_reg_t* input) {
    // 创建一个全零向量
    vector_reg_t zero;
#ifdef __aarch64__
    zero.low = vdupq_n_s32(0);
    zero.high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        zero[i] = 0;
    }
#endif

    // 比较输入与零
    vector_reg_t mask;
    vector_compare(&mask, input, &zero);
    
    // 选择大于零的值，否则为零
    vector_select(result, &mask, input, &zero);
}

// 新增：层归一化函数
void layer_normalization(vector_reg_t* result, vector_reg_t* input) {
    // 计算均值
    int64_t sum = 0;
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        sum += get_vector_element(input, i);
    }
    int32_t mean = sum / VECTOR_LENGTH;
    
    // 计算方差
    int64_t variance_sum = 0;
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        int32_t diff = get_vector_element(input, i) - mean;
        variance_sum += diff * diff;
    }
    int32_t variance = variance_sum / VECTOR_LENGTH;
    
    // 防止除零
    if (variance == 0) variance = 1;
    
    // 标准差
    int32_t std_dev = (int32_t)sqrt((double)variance);
    
    // 归一化
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        int32_t normalized = ((get_vector_element(input, i) - mean) * 1000) / std_dev;
        set_vector_element(result, i, normalized);
    }
    
    printf("层归一化 - 均值: %d, 标准差: %d\n", mean, std_dev);
}

// 辅助函数：矩阵-向量乘法（简化版，假设矩阵已经按行存储在向量寄存器中）
void matrix_vector_mul(vector_reg_t* result, vector_reg_t matrix[EMBEDDING_DIM], vector_reg_t* vec) {
    vector_reg_t temp;
    
    // 初始化结果为零
#ifdef __aarch64__
    result->low = vdupq_n_s32(0);
    result->high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = 0;
    }
#endif
    
    // 计算每一行的点积并累加
    for(int i = 0; i < EMBEDDING_DIM; i++) {
        // 计算点积
        vector_mul(&temp, &matrix[i], vec);
        
        // 规约求和
        int32_t dot_product = vector_reduce(&temp, RED_SUM);
        
        // 存储到结果向量
        set_vector_element(result, i, dot_product);
    }
}

// 辅助函数：简化的自注意力机制
void self_attention(vector_reg_t* result, vector_reg_t* query, vector_reg_t* key, vector_reg_t* value) {
    vector_reg_t attention_scores;
    
    // 计算注意力分数（简化为向量乘法）
    vector_mul(&attention_scores, query, key);
    
    // 应用softmax（简化为归一化）
    int32_t sum = vector_reduce(&attention_scores, RED_SUM);
    
    // 避免除以零
    if(sum == 0) sum = 1;
    
    // 归一化（简化版softmax）
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        int32_t score = get_vector_element(&attention_scores, i);
        set_vector_element(&attention_scores, i, (score * 1000) / sum);
    }
    
    // 加权求和
    vector_mul(result, &attention_scores, value);
}

// 辅助函数：使用FFT进行序列处理
void process_sequence_with_fft(vector_reg_t* result, vector_reg_t* input) {
    vector_reg_t fft_result, ifft_result;
    
    // 应用FFT
    vector_fft(&fft_result, input);
    
    // 在频域进行处理（简化为放大某些频率）
    for(int i = 0; i < VECTOR_LENGTH/2; i++) {
        int32_t val = get_vector_element(&fft_result, i);
        set_vector_element(&fft_result, i, val * 2);
        
        val = get_vector_element(&fft_result, i + VECTOR_LENGTH/2);
        set_vector_element(&fft_result, i + VECTOR_LENGTH/2, val / 2);
    }
    
    // 应用IFFT
    vector_ifft(result, &fft_result);
}

// 主函数：实现简化的Transformer层
void transformer_layer(vector_reg_t* output, vector_reg_t* input, 
                      vector_reg_t weights_q[EMBEDDING_DIM],
                      vector_reg_t weights_k[EMBEDDING_DIM],
                      vector_reg_t weights_v[EMBEDDING_DIM],
                      vector_reg_t weights_out[EMBEDDING_DIM]) {
    
    vector_reg_t query, key, value, attention_output, temp, normalized;
    
    // 1. 计算查询、键、值向量
    matrix_vector_mul(&query, weights_q, input);
    matrix_vector_mul(&key, weights_k, input);
    matrix_vector_mul(&value, weights_v, input);
    
    // 2. 应用自注意力机制
    self_attention(&attention_output, &query, &key, &value);
    
    // 3. 应用输出投影
    matrix_vector_mul(&temp, weights_out, &attention_output);
    
    // 4. 残差连接
    vector_add(output, &temp, input);
    
    // 5. 应用层归一化
    layer_normalization(&normalized, output);
    
    // 6. 应用ReLU激活函数
    relu_activation(output, &normalized);
    
    // 7. 使用FFT进行序列处理
    process_sequence_with_fft(output, output);
}

// 主函数
int main() {
    printf("=== 简化版LLM计算演示 ===\n\n");
    
    // 初始化执行单元
    init_execution_unit();
    
    // 初始化输入向量
    vector_reg_t input;
    int32_t input_values[VECTOR_LENGTH] = {10, 20, 30, 40, 50, 60, 70, 80};
    init_vector(&input, input_values);
    print_vector("输入向量", &input);
    
    // 初始化权重矩阵（简化为随机值）
    vector_reg_t weights_q[EMBEDDING_DIM];
    vector_reg_t weights_k[EMBEDDING_DIM];
    vector_reg_t weights_v[EMBEDDING_DIM];
    vector_reg_t weights_out[EMBEDDING_DIM];
    
    // 初始化权重（简化为简单模式）
    for(int i = 0; i < EMBEDDING_DIM; i++) {
        int32_t w_q[VECTOR_LENGTH], w_k[VECTOR_LENGTH], w_v[VECTOR_LENGTH], w_out[VECTOR_LENGTH];
        
        for(int j = 0; j < VECTOR_LENGTH; j++) {
            w_q[j] = (i + j) % 5 + 1;
            w_k[j] = (i * j) % 5 + 1;
            w_v[j] = (i - j + 8) % 5 + 1;
            w_out[j] = (i + j * 2) % 5 + 1;
        }
        
        init_vector(&weights_q[i], w_q);
        init_vector(&weights_k[i], w_k);
        init_vector(&weights_v[i], w_v);
        init_vector(&weights_out[i], w_out);
    }
    
    // 输出向量
    vector_reg_t output = input;
    
    // 应用多层Transformer
    printf("\n开始LLM计算...\n");
    for(int layer = 0; layer < NUM_LAYERS; layer++) {
        printf("\n=== 第 %d 层 ===\n", layer + 1);
        
        transformer_layer(&output, &output, weights_q, weights_k, weights_v, weights_out);
        
        print_vector("层输出", &output);
    }
    
    // 最终输出
    printf("\n=== 最终输出 ===\n");
    print_vector("LLM输出向量", &output);
    
    // 计算输出的统计信息
    int32_t sum = vector_reduce(&output, RED_SUM);
    int32_t max = vector_reduce(&output, RED_MAX);
    int32_t min = vector_reduce(&output, RED_MIN);
    
    printf("\n输出统计信息:\n");
    printf("总和: %d\n", sum);
    printf("最大值: %d\n", max);
    printf("最小值: %d\n", min);
    
    return 0;
} 