/**
 * llm_calc.c - 使用K-ISA实现简化版LLM计算
 * 
 * 这个程序实现了一个简化的LLM前向传播计算，包括：
 * 1. 矩阵-向量乘法（模拟线性层）
 * 2. 向量激活函数（使用ReLU）
 * 3. 多头注意力机制
 * 4. 使用FFT进行高级序列处理（频域卷积和特征提取）
 * 5. 层归一化（Layer Normalization）
 * 6. 位置编码（Positional Encoding）
 * 7. 经验动态建模（Empirical Dynamic Modeling）
 */

#include "kisa.h"
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
#define NUM_HEADS 2      // 注意力头数量
#define HEAD_DIM 4       // 每个头的维度 (EMBEDDING_DIM / NUM_HEADS)
#define POS_ENCODING_SCALE 100  // 位置编码缩放因子

// EDM参数
#define EDM_EMBEDDING_DIM 3     // EDM嵌入维度
#define EDM_TIME_DELAY 1        // 时间延迟
#define EDM_NUM_NEIGHBORS 3     // 近邻数量
#define EDM_PREDICTION_STEPS 1  // 预测步数

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

// 层归一化函数
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
    if (variance < 10) variance = 10;  // 使用更大的最小值防止标准差过小
    
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

// 辅助函数：简化的单头注意力机制
void single_head_attention(vector_reg_t* result, vector_reg_t* query, vector_reg_t* key, vector_reg_t* value) {
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

// 辅助函数：使用FFT进行序列处理（简单版本）
void process_sequence_with_fft_simple(vector_reg_t* result, vector_reg_t* input) {
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

// 新增：频域卷积操作
void frequency_domain_convolution(vector_reg_t* result, vector_reg_t* input, vector_reg_t* kernel) {
    vector_reg_t input_fft, kernel_fft, conv_result;
    
    // 1. 对输入和卷积核应用FFT
    vector_fft(&input_fft, input);
    vector_fft(&kernel_fft, kernel);
    
    // 2. 在频域中进行点乘（卷积在频域中等价于点乘）
    vector_mul(&conv_result, &input_fft, &kernel_fft);
    
    // 3. 应用IFFT得到时域结果
    vector_ifft(result, &conv_result);
    
    printf("频域卷积完成\n");
}

// 新增：频域特征提取
void extract_frequency_features(vector_reg_t* result, vector_reg_t* input) {
    vector_reg_t fft_result;
    
    // 1. 应用FFT
    vector_fft(&fft_result, input);
    
    // 2. 提取频域特征
    // 低频特征（前半部分）
    int32_t low_freq_energy = 0;
    for(int i = 0; i < VECTOR_LENGTH/4; i++) {
        int32_t val = get_vector_element(&fft_result, i);
        low_freq_energy += abs(val);
    }
    
    // 中频特征
    int32_t mid_freq_energy = 0;
    for(int i = VECTOR_LENGTH/4; i < 3*VECTOR_LENGTH/4; i++) {
        int32_t val = get_vector_element(&fft_result, i);
        mid_freq_energy += abs(val);
    }
    
    // 高频特征
    int32_t high_freq_energy = 0;
    for(int i = 3*VECTOR_LENGTH/4; i < VECTOR_LENGTH; i++) {
        int32_t val = get_vector_element(&fft_result, i);
        high_freq_energy += abs(val);
    }
    
    // 3. 根据频域特征调整原始信号
    // 如果低频能量高，增强高频部分
    if(low_freq_energy > mid_freq_energy && low_freq_energy > high_freq_energy) {
        for(int i = 3*VECTOR_LENGTH/4; i < VECTOR_LENGTH; i++) {
            int32_t val = get_vector_element(&fft_result, i);
            set_vector_element(&fft_result, i, val * 3);
        }
    }
    // 如果高频能量高，增强低频部分
    else if(high_freq_energy > low_freq_energy && high_freq_energy > mid_freq_energy) {
        for(int i = 0; i < VECTOR_LENGTH/4; i++) {
            int32_t val = get_vector_element(&fft_result, i);
            set_vector_element(&fft_result, i, val * 3);
        }
    }
    // 如果中频能量高，均衡增强
    else {
        for(int i = 0; i < VECTOR_LENGTH; i++) {
            int32_t val = get_vector_element(&fft_result, i);
            set_vector_element(&fft_result, i, val * 2);
        }
    }
    
    // 4. 应用IFFT得到增强后的时域信号
    vector_ifft(result, &fft_result);
    
    printf("频域特征提取 - 低频能量: %d, 中频能量: %d, 高频能量: %d\n", 
           low_freq_energy, mid_freq_energy, high_freq_energy);
}

// 新增：高级FFT处理（结合卷积和特征提取）
void advanced_fft_processing(vector_reg_t* result, vector_reg_t* input, int layer) {
    vector_reg_t conv_kernel, conv_result, feature_result;
    
    // 1. 创建卷积核（简化为不同的模式）
    int32_t kernel_values[VECTOR_LENGTH];
    if(layer % 2 == 0) {
        // 低通滤波器模式
        for(int i = 0; i < VECTOR_LENGTH; i++) {
            kernel_values[i] = (i < VECTOR_LENGTH/2) ? 100 : 10;
        }
    } else {
        // 高通滤波器模式
        for(int i = 0; i < VECTOR_LENGTH; i++) {
            kernel_values[i] = (i >= VECTOR_LENGTH/2) ? 100 : 10;
        }
    }
    init_vector(&conv_kernel, kernel_values);
    
    // 2. 应用频域卷积
    frequency_domain_convolution(&conv_result, input, &conv_kernel);
    
    // 3. 应用频域特征提取
    extract_frequency_features(&feature_result, &conv_result);
    
    // 4. 将结果复制到输出
#ifdef __aarch64__
    result->low = feature_result.low;
    result->high = feature_result.high;
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = feature_result[i];
    }
#endif
    
    printf("高级FFT处理完成\n");
}

// 新增：多头注意力机制
void multi_head_attention(vector_reg_t* result, 
                         vector_reg_t* query, 
                         vector_reg_t* key, 
                         vector_reg_t* value,
                         vector_reg_t weights_proj[EMBEDDING_DIM]) {
    
    vector_reg_t head_results[NUM_HEADS];
    vector_reg_t q_heads[NUM_HEADS], k_heads[NUM_HEADS], v_heads[NUM_HEADS];
    vector_reg_t concat_result;
    
    // 初始化结果向量
    for(int h = 0; h < NUM_HEADS; h++) {
#ifdef __aarch64__
        head_results[h].low = vdupq_n_s32(0);
        head_results[h].high = vdupq_n_s32(0);
        q_heads[h].low = vdupq_n_s32(0);
        q_heads[h].high = vdupq_n_s32(0);
        k_heads[h].low = vdupq_n_s32(0);
        k_heads[h].high = vdupq_n_s32(0);
        v_heads[h].low = vdupq_n_s32(0);
        v_heads[h].high = vdupq_n_s32(0);
#else
        for(int i = 0; i < VECTOR_LENGTH; i++) {
            head_results[h][i] = 0;
            q_heads[h][i] = 0;
            k_heads[h][i] = 0;
            v_heads[h][i] = 0;
        }
#endif
    }
    
    // 1. 将查询、键、值分割为多个头
    for(int h = 0; h < NUM_HEADS; h++) {
        // 简化：我们只是将向量分成几个部分
        for(int i = 0; i < HEAD_DIM; i++) {
            // 查询向量分割
            int32_t q_val = get_vector_element(query, h * HEAD_DIM + i);
            set_vector_element(&q_heads[h], i, q_val);
            
            // 键向量分割
            int32_t k_val = get_vector_element(key, h * HEAD_DIM + i);
            set_vector_element(&k_heads[h], i, k_val);
            
            // 值向量分割
            int32_t v_val = get_vector_element(value, h * HEAD_DIM + i);
            set_vector_element(&v_heads[h], i, v_val);
        }
    }
    
    // 2. 对每个头应用注意力机制
    for(int h = 0; h < NUM_HEADS; h++) {
        printf("处理注意力头 %d\n", h + 1);
        single_head_attention(&head_results[h], &q_heads[h], &k_heads[h], &v_heads[h]);
    }
    
    // 3. 拼接多头结果
#ifdef __aarch64__
    concat_result.low = vdupq_n_s32(0);
    concat_result.high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        concat_result[i] = 0;
    }
#endif
    
    for(int h = 0; h < NUM_HEADS; h++) {
        for(int i = 0; i < HEAD_DIM; i++) {
            int32_t val = get_vector_element(&head_results[h], i);
            set_vector_element(&concat_result, h * HEAD_DIM + i, val);
        }
    }
    
    // 4. 应用最终的线性投影
    matrix_vector_mul(result, weights_proj, &concat_result);
    
    printf("多头注意力完成\n");
}

// 新增：位置编码函数
void add_positional_encoding(vector_reg_t* result, vector_reg_t* input, int position) {
    // 复制输入到结果
#ifdef __aarch64__
    result->low = input->low;
    result->high = input->high;
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = (*input)[i];
    }
#endif
    
    // 应用位置编码
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        // 计算位置编码
        int32_t pos_enc;
        if(i % 2 == 0) {
            // 使用正弦函数对偶数位置
            pos_enc = (int32_t)(sin(position / pow(10000, i / (double)EMBEDDING_DIM)) * POS_ENCODING_SCALE);
        } else {
            // 使用余弦函数对奇数位置
            pos_enc = (int32_t)(cos(position / pow(10000, (i - 1) / (double)EMBEDDING_DIM)) * POS_ENCODING_SCALE);
        }
        
        // 将位置编码添加到输入
        int32_t val = get_vector_element(result, i) + pos_enc;
        set_vector_element(result, i, val);
    }
    
    printf("添加位置编码 %d: ", position);
    print_vector("位置编码后", result);
}

// 新增：EDM时间延迟嵌入
void time_delay_embedding(vector_reg_t* result, vector_reg_t* input, int delay, int embedding_dim) {
    printf("执行时间延迟嵌入 (延迟=%d, 嵌入维度=%d)\n", delay, embedding_dim);
    
    // 初始化结果向量为零
#ifdef __aarch64__
    result->low = vdupq_n_s32(0);
    result->high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = 0;
    }
#endif
    
    // 对于向量的每个元素，创建延迟嵌入
    // 注意：这里我们只使用向量的前embedding_dim个元素
    for(int i = 0; i < embedding_dim; i++) {
        // 计算当前时间点
        int time_point = i * delay;
        
        // 如果时间点在向量范围内
        if(time_point < VECTOR_LENGTH) {
            int32_t value = get_vector_element(input, time_point);
            set_vector_element(result, i, value);
        }
    }
    
    print_vector("时间延迟嵌入结果", result);
}

// 新增：计算欧几里得距离
int32_t euclidean_distance(vector_reg_t* v1, vector_reg_t* v2, int dim) {
    int64_t sum_sq = 0;
    
    for(int i = 0; i < dim; i++) {
        int32_t diff = get_vector_element(v1, i) - get_vector_element(v2, i);
        sum_sq += (int64_t)diff * diff;
    }
    
    return (int32_t)sqrt((double)sum_sq);
}

// 新增：EDM近邻搜索
void find_nearest_neighbors(int* neighbor_indices, vector_reg_t* target, 
                           vector_reg_t library[], int library_size, 
                           int num_neighbors, int embedding_dim) {
    printf("执行近邻搜索 (库大小=%d, 近邻数=%d)\n", library_size, num_neighbors);
    
    // 距离数组
    typedef struct {
        int index;
        int32_t distance;
    } DistanceItem;
    
    DistanceItem* distances = (DistanceItem*)malloc(library_size * sizeof(DistanceItem));
    
    // 计算目标向量与库中每个向量的距离
    for(int i = 0; i < library_size; i++) {
        distances[i].index = i;
        distances[i].distance = euclidean_distance(target, &library[i], embedding_dim);
    }
    
    // 简单的冒泡排序找出最近的邻居
    for(int i = 0; i < library_size - 1; i++) {
        for(int j = 0; j < library_size - i - 1; j++) {
            if(distances[j].distance > distances[j + 1].distance) {
                DistanceItem temp = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = temp;
            }
        }
    }
    
    // 获取最近的邻居索引
    for(int i = 0; i < num_neighbors && i < library_size; i++) {
        neighbor_indices[i] = distances[i].index;
        printf("近邻 %d: 索引 %d, 距离 %d\n", i+1, neighbor_indices[i], distances[i].distance);
    }
    
    free(distances);
}

// 新增：EDM预测
void edm_predict(vector_reg_t* result, vector_reg_t* current_state, 
                vector_reg_t library[], int library_size, 
                int num_neighbors, int embedding_dim, int prediction_steps) {
    printf("执行EDM预测 (预测步数=%d)\n", prediction_steps);
    
    // 初始化结果向量为零
#ifdef __aarch64__
    result->low = vdupq_n_s32(0);
    result->high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = 0;
    }
#endif
    
    // 找到最近的邻居
    int* neighbor_indices = (int*)malloc(num_neighbors * sizeof(int));
    find_nearest_neighbors(neighbor_indices, current_state, library, library_size, num_neighbors, embedding_dim);
    
    // 基于近邻的加权平均进行预测
    int32_t total_weight = 0;
    
    for(int i = 0; i < num_neighbors; i++) {
        int neighbor_idx = neighbor_indices[i];
        
        // 计算权重（简化为距离的倒数）
        int32_t distance = euclidean_distance(current_state, &library[neighbor_idx], embedding_dim);
        int32_t weight = distance == 0 ? 1000 : 1000 / distance; // 避免除以零
        total_weight += weight;
        
        // 对于每个预测步骤，将未来值加入结果
        for(int step = 1; step <= prediction_steps; step++) {
            // 确保我们不会超出库的范围
            if(neighbor_idx + step < library_size) {
                for(int j = 0; j < VECTOR_LENGTH; j++) {
                    int32_t future_val = get_vector_element(&library[neighbor_idx + step], j);
                    int32_t current_val = get_vector_element(result, j);
                    set_vector_element(result, j, current_val + future_val * weight);
                }
            }
        }
    }
    
    // 归一化结果
    if(total_weight > 0) {
        for(int j = 0; j < VECTOR_LENGTH; j++) {
            int32_t val = get_vector_element(result, j);
            set_vector_element(result, j, val / total_weight);
        }
    }
    
    free(neighbor_indices);
    print_vector("EDM预测结果", result);
}

// 新增：完整的EDM处理流程
void apply_empirical_dynamic_modeling(vector_reg_t* result, vector_reg_t* input) {
    printf("\n=== 应用经验动态建模 (EDM) ===\n");
    
    // 1. 创建一个简单的时间序列库（在实际应用中，这将是历史数据）
    int library_size = VECTOR_LENGTH;
    vector_reg_t* library = (vector_reg_t*)malloc(library_size * sizeof(vector_reg_t));
    
    // 为简化起见，我们使用输入向量的移位版本作为库
    for(int i = 0; i < library_size; i++) {
        // 创建移位版本
#ifdef __aarch64__
        library[i].low = vdupq_n_s32(0);
        library[i].high = vdupq_n_s32(0);
#else
        for(int j = 0; j < VECTOR_LENGTH; j++) {
            library[i][j] = 0;
        }
#endif
        
        for(int j = 0; j < VECTOR_LENGTH; j++) {
            int src_idx = (j + i) % VECTOR_LENGTH;
            int32_t val = get_vector_element(input, src_idx);
            set_vector_element(&library[i], j, val);
        }
    }
    
    // 2. 对当前状态进行时间延迟嵌入
    vector_reg_t embedded_state;
    time_delay_embedding(&embedded_state, input, EDM_TIME_DELAY, EDM_EMBEDDING_DIM);
    
    // 3. 对库中的每个向量进行时间延迟嵌入
    vector_reg_t* embedded_library = (vector_reg_t*)malloc(library_size * sizeof(vector_reg_t));
    for(int i = 0; i < library_size; i++) {
        time_delay_embedding(&embedded_library[i], &library[i], EDM_TIME_DELAY, EDM_EMBEDDING_DIM);
    }
    
    // 4. 使用EDM进行预测
    edm_predict(result, &embedded_state, embedded_library, library_size, 
               EDM_NUM_NEIGHBORS, EDM_EMBEDDING_DIM, EDM_PREDICTION_STEPS);
    
    // 5. 清理
    free(library);
    free(embedded_library);
    
    printf("EDM处理完成\n");
}

// 主函数：实现简化的Transformer层
void transformer_layer(vector_reg_t* output, vector_reg_t* input, 
                      vector_reg_t weights_q[EMBEDDING_DIM],
                      vector_reg_t weights_k[EMBEDDING_DIM],
                      vector_reg_t weights_v[EMBEDDING_DIM],
                      vector_reg_t weights_out[EMBEDDING_DIM],
                      int position) {
    
    vector_reg_t query, key, value, attention_output, temp, normalized, pos_encoded;
    
    // 0. 添加位置编码
    add_positional_encoding(&pos_encoded, input, position);
    
    // 1. 计算查询、键、值向量
    matrix_vector_mul(&query, weights_q, &pos_encoded);
    matrix_vector_mul(&key, weights_k, &pos_encoded);
    matrix_vector_mul(&value, weights_v, &pos_encoded);
    
    // 2. 应用多头注意力机制
    multi_head_attention(&attention_output, &query, &key, &value, weights_out);
    
    // 3. 应用输出投影
    matrix_vector_mul(&temp, weights_out, &attention_output);
    
    // 4. 残差连接
    vector_add(output, &temp, &pos_encoded);
    
    // 5. 应用层归一化
    layer_normalization(&normalized, output);
    
    // 6. 应用ReLU激活函数
    relu_activation(output, &normalized);
    
    // 7. 使用高级FFT处理
    advanced_fft_processing(output, output, position);
    
    // 8. 应用经验动态建模（EDM）
    vector_reg_t edm_result;
    apply_empirical_dynamic_modeling(&edm_result, output);
    
    // 9. 将EDM结果与当前输出结合
    vector_add(output, output, &edm_result);
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
        
        // 为每一层使用不同的位置编码
        transformer_layer(&output, &output, weights_q, weights_k, weights_v, weights_out, layer);
        
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