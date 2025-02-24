/**
 * orderbook_edm_analysis.c - 使用经验动态建模(EDM)分析订单簿事件的泊松到达率
 * 
 * 这个程序模拟订单簿事件流，并使用EDM分析不同类型事件的到达率和模式：
 * 1. 新订单事件 (New)
 * 2. 修改订单事件 (Modify)
 * 3. 删除订单事件 (Delete)
 * 4. 交易执行事件 (Execute)
 * 
 * 程序使用EDM来预测未来事件率并分析事件之间的动态关系。
 */

#include "include/kisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// 参数定义
#define MAX_EVENTS 1000          // 最大事件数
#define VECTOR_SIZE 8            // 向量大小
#define EVENT_TYPES 4            // 事件类型数量
#define TIME_WINDOW 100          // 时间窗口大小
#define EDM_EMBEDDING_DIM 3      // EDM嵌入维度
#define EDM_TIME_DELAY 1         // 时间延迟
#define EDM_NUM_NEIGHBORS 3      // 近邻数量
#define EDM_PREDICTION_STEPS 5   // 预测步数

// 事件类型枚举
typedef enum {
    EVENT_NEW = 0,
    EVENT_MODIFY = 1,
    EVENT_DELETE = 2,
    EVENT_EXECUTE = 3
} EventType;

// 事件结构体
typedef struct {
    EventType type;
    int timestamp;
    double price;
    int quantity;
} OrderEvent;

// 事件率结构体
typedef struct {
    int window_start;
    int window_end;
    int counts[EVENT_TYPES];
    double rates[EVENT_TYPES];
} EventRates;

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

// 生成模拟的订单簿事件
void generate_simulated_events(OrderEvent events[], int num_events) {
    srand(time(NULL));
    
    // 设置不同事件类型的基础泊松率
    double base_rates[EVENT_TYPES] = {0.4, 0.3, 0.2, 0.1}; // 新订单、修改、删除、执行的基础率
    
    // 为每种事件类型生成周期性变化
    double amplitudes[EVENT_TYPES] = {0.2, 0.15, 0.1, 0.05}; // 振幅
    double periods[EVENT_TYPES] = {200, 150, 100, 50}; // 周期
    
    int current_time = 0;
    for (int i = 0; i < num_events; i++) {
        // 计算每种事件类型的当前概率（基础率+周期变化）
        double probs[EVENT_TYPES];
        double total_prob = 0;
        
        for (int j = 0; j < EVENT_TYPES; j++) {
            // 添加周期性变化
            probs[j] = base_rates[j] + amplitudes[j] * sin(2 * M_PI * current_time / periods[j]);
            // 确保概率非负
            if (probs[j] < 0.01) probs[j] = 0.01;
            total_prob += probs[j];
        }
        
        // 归一化概率
        for (int j = 0; j < EVENT_TYPES; j++) {
            probs[j] /= total_prob;
        }
        
        // 随机选择事件类型
        double r = (double)rand() / RAND_MAX;
        double cumulative = 0;
        EventType selected_type = EVENT_NEW; // 默认
        
        for (int j = 0; j < EVENT_TYPES; j++) {
            cumulative += probs[j];
            if (r <= cumulative) {
                selected_type = (EventType)j;
                break;
            }
        }
        
        // 生成事件
        events[i].type = selected_type;
        events[i].timestamp = current_time;
        events[i].price = 100.0 + (rand() % 1000) / 100.0; // 100-110范围内的价格
        events[i].quantity = 1 + rand() % 100; // 1-100范围内的数量
        
        // 更新时间（使用泊松过程模拟事件间隔）
        double lambda = 0.5; // 平均事件率
        double u = (double)rand() / RAND_MAX;
        int time_increment = (int)(-log(u) / lambda); // 泊松间隔
        if (time_increment < 1) time_increment = 1;
        current_time += time_increment;
    }
    
    printf("生成了%d个模拟订单簿事件\n", num_events);
}

// 计算时间窗口内的事件率
void calculate_event_rates(OrderEvent events[], int num_events, EventRates rates[], int *num_rates) {
    if (num_events == 0) {
        *num_rates = 0;
        return;
    }
    
    int start_time = events[0].timestamp;
    int end_time = events[num_events-1].timestamp;
    
    // 计算需要多少个时间窗口
    int num_windows = (end_time - start_time) / TIME_WINDOW + 1;
    *num_rates = num_windows;
    
    // 初始化事件率数组
    for (int i = 0; i < num_windows; i++) {
        rates[i].window_start = start_time + i * TIME_WINDOW;
        rates[i].window_end = rates[i].window_start + TIME_WINDOW - 1;
        
        for (int j = 0; j < EVENT_TYPES; j++) {
            rates[i].counts[j] = 0;
            rates[i].rates[j] = 0.0;
        }
    }
    
    // 统计每个窗口内的事件数量
    for (int i = 0; i < num_events; i++) {
        int window_idx = (events[i].timestamp - start_time) / TIME_WINDOW;
        if (window_idx >= 0 && window_idx < num_windows) {
            rates[window_idx].counts[events[i].type]++;
        }
    }
    
    // 计算事件率
    for (int i = 0; i < num_windows; i++) {
        for (int j = 0; j < EVENT_TYPES; j++) {
            rates[i].rates[j] = (double)rates[i].counts[j] / TIME_WINDOW;
        }
    }
    
    printf("计算了%d个时间窗口的事件率\n", num_windows);
}

// 将事件率转换为向量
void event_rates_to_vector(EventRates rates[], int window_idx, vector_reg_t* result) {
    // 初始化结果向量为零
#ifdef __aarch64__
    result->low = vdupq_n_s32(0);
    result->high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = 0;
    }
#endif
    
    // 将事件率转换为向量元素
    // 前4个元素是各类事件的率
    for (int i = 0; i < EVENT_TYPES && i < VECTOR_LENGTH; i++) {
        int32_t rate_scaled = (int32_t)(rates[window_idx].rates[i] * 1000); // 缩放以转换为整数
        set_vector_element(result, i, rate_scaled);
    }
    
    // 后4个元素是各类事件的计数
    for (int i = 0; i < EVENT_TYPES && i + EVENT_TYPES < VECTOR_LENGTH; i++) {
        set_vector_element(result, i + EVENT_TYPES, rates[window_idx].counts[i]);
    }
}

// EDM时间延迟嵌入
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

// 计算欧几里得距离
int32_t euclidean_distance(vector_reg_t* v1, vector_reg_t* v2, int dim) {
    int64_t sum_sq = 0;
    
    for(int i = 0; i < dim; i++) {
        int32_t diff = get_vector_element(v1, i) - get_vector_element(v2, i);
        sum_sq += (int64_t)diff * diff;
    }
    
    return (int32_t)sqrt((double)sum_sq);
}

// EDM近邻搜索
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

// EDM预测
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

// 分析事件率的自相关性
void analyze_autocorrelation(EventRates rates[], int num_rates) {
    printf("\n=== 事件率自相关性分析 ===\n");
    
    // 最大滞后期
    int max_lag = num_rates / 4;
    if (max_lag > 20) max_lag = 20;
    
    // 对每种事件类型计算自相关
    for (int type = 0; type < EVENT_TYPES; type++) {
        const char* event_names[] = {"新订单", "修改订单", "删除订单", "交易执行"};
        printf("\n%s事件的自相关系数:\n", event_names[type]);
        
        // 计算均值
        double mean = 0.0;
        for (int i = 0; i < num_rates; i++) {
            mean += rates[i].rates[type];
        }
        mean /= num_rates;
        
        // 计算方差
        double variance = 0.0;
        for (int i = 0; i < num_rates; i++) {
            double diff = rates[i].rates[type] - mean;
            variance += diff * diff;
        }
        variance /= num_rates;
        
        // 计算不同滞后期的自相关
        for (int lag = 1; lag <= max_lag; lag++) {
            double autocorr = 0.0;
            for (int i = 0; i < num_rates - lag; i++) {
                double diff1 = rates[i].rates[type] - mean;
                double diff2 = rates[i + lag].rates[type] - mean;
                autocorr += diff1 * diff2;
            }
            autocorr /= (num_rates - lag) * variance;
            
            printf("滞后%d: %.4f", lag, autocorr);
            
            // 简单的可视化
            printf(" |");
            int bars = (int)(fabs(autocorr) * 40);
            for (int b = 0; b < bars; b++) {
                printf(autocorr >= 0 ? "+" : "-");
            }
            printf("|\n");
        }
    }
}

// 分析事件类型之间的相关性
void analyze_cross_correlation(EventRates rates[], int num_rates) {
    printf("\n=== 事件类型间的相关性分析 ===\n");
    
    const char* event_names[] = {"新订单", "修改订单", "删除订单", "交易执行"};
    
    // 计算每种事件类型的均值
    double means[EVENT_TYPES] = {0};
    for (int type = 0; type < EVENT_TYPES; type++) {
        for (int i = 0; i < num_rates; i++) {
            means[type] += rates[i].rates[type];
        }
        means[type] /= num_rates;
    }
    
    // 计算每种事件类型的标准差
    double stddevs[EVENT_TYPES] = {0};
    for (int type = 0; type < EVENT_TYPES; type++) {
        for (int i = 0; i < num_rates; i++) {
            double diff = rates[i].rates[type] - means[type];
            stddevs[type] += diff * diff;
        }
        stddevs[type] = sqrt(stddevs[type] / num_rates);
    }
    
    // 计算事件类型之间的相关系数
    printf("\n相关系数矩阵:\n");
    printf("%-12s", "");
    for (int type2 = 0; type2 < EVENT_TYPES; type2++) {
        printf("%-12s", event_names[type2]);
    }
    printf("\n");
    
    for (int type1 = 0; type1 < EVENT_TYPES; type1++) {
        printf("%-12s", event_names[type1]);
        
        for (int type2 = 0; type2 < EVENT_TYPES; type2++) {
            double correlation = 0.0;
            
            for (int i = 0; i < num_rates; i++) {
                double diff1 = rates[i].rates[type1] - means[type1];
                double diff2 = rates[i].rates[type2] - means[type2];
                correlation += diff1 * diff2;
            }
            
            correlation /= num_rates * stddevs[type1] * stddevs[type2];
            
            printf("%-12.4f", correlation);
        }
        printf("\n");
    }
}

// 使用EDM分析事件率
void analyze_event_rates_with_edm(EventRates rates[], int num_rates) {
    printf("\n=== 使用EDM分析事件率 ===\n");
    
    // 如果数据点太少，无法进行EDM分析
    if (num_rates < EDM_EMBEDDING_DIM + EDM_PREDICTION_STEPS) {
        printf("数据点不足，无法进行EDM分析\n");
        return;
    }
    
    // 将事件率转换为向量序列
    vector_reg_t* rate_vectors = (vector_reg_t*)malloc(num_rates * sizeof(vector_reg_t));
    for (int i = 0; i < num_rates; i++) {
        event_rates_to_vector(rates, i, &rate_vectors[i]);
    }
    
    // 创建嵌入库
    vector_reg_t* embedded_library = (vector_reg_t*)malloc(num_rates * sizeof(vector_reg_t));
    for (int i = 0; i < num_rates; i++) {
        time_delay_embedding(&embedded_library[i], &rate_vectors[i], EDM_TIME_DELAY, EDM_EMBEDDING_DIM);
    }
    
    // 对每种事件类型进行EDM预测
    const char* event_names[] = {"新订单", "修改订单", "删除订单", "交易执行"};
    
    for (int type = 0; type < EVENT_TYPES; type++) {
        printf("\n分析%s事件的动态:\n", event_names[type]);
        
        // 选择最后一个时间点进行预测
        vector_reg_t current_state;
        time_delay_embedding(&current_state, &rate_vectors[num_rates-1], EDM_TIME_DELAY, EDM_EMBEDDING_DIM);
        
        // 使用EDM进行预测
        vector_reg_t prediction;
        edm_predict(&prediction, &current_state, embedded_library, num_rates, 
                   EDM_NUM_NEIGHBORS, EDM_EMBEDDING_DIM, EDM_PREDICTION_STEPS);
        
        // 提取该事件类型的预测率
        int32_t predicted_rate = get_vector_element(&prediction, type);
        double actual_rate = (double)predicted_rate / 1000.0; // 转换回实际率
        
        printf("%s事件的预测率: %.4f\n", event_names[type], actual_rate);
        
        // 计算预测的非线性特性
        printf("分析%s事件的非线性特性:\n", event_names[type]);
        
        // 计算实际序列的方差
        double mean = 0.0;
        for (int i = 0; i < num_rates; i++) {
            mean += rates[i].rates[type];
        }
        mean /= num_rates;
        
        double variance = 0.0;
        for (int i = 0; i < num_rates; i++) {
            double diff = rates[i].rates[type] - mean;
            variance += diff * diff;
        }
        variance /= num_rates;
        
        printf("均值: %.4f, 方差: %.4f\n", mean, variance);
        
        // 计算序列的非线性指标（简化版）
        double nonlinearity = 0.0;
        for (int i = 1; i < num_rates; i++) {
            double diff = rates[i].rates[type] - rates[i-1].rates[type];
            nonlinearity += fabs(diff);
        }
        nonlinearity /= (num_rates - 1);
        
        printf("非线性指标: %.4f\n", nonlinearity);
        
        // 判断是否符合泊松过程
        // 对于泊松过程，方差应该接近均值
        double poisson_ratio = variance / mean;
        printf("方差/均值比率: %.4f ", poisson_ratio);
        
        if (fabs(poisson_ratio - 1.0) < 0.2) {
            printf("(接近1，符合泊松过程特性)\n");
        } else if (poisson_ratio > 1.0) {
            printf("(大于1，表现出过度分散，可能是集群到达)\n");
        } else {
            printf("(小于1，表现出欠分散，可能是规则到达)\n");
        }
    }
    
    // 清理
    free(rate_vectors);
    free(embedded_library);
}

// 主函数
int main() {
    printf("=== 订单簿事件泊松到达率分析 ===\n\n");
    
    // 初始化执行单元
    init_execution_unit();
    
    // 生成模拟的订单簿事件
    OrderEvent events[MAX_EVENTS];
    generate_simulated_events(events, MAX_EVENTS);
    
    // 计算事件率
    EventRates rates[MAX_EVENTS / TIME_WINDOW + 1];
    int num_rates;
    calculate_event_rates(events, MAX_EVENTS, rates, &num_rates);
    
    // 打印事件率
    printf("\n=== 事件率统计 ===\n");
    printf("%-15s %-15s %-15s %-15s %-15s %-15s\n", 
           "时间窗口", "新订单率", "修改订单率", "删除订单率", "交易执行率", "总事件率");
    
    for (int i = 0; i < num_rates; i++) {
        double total_rate = 0;
        for (int j = 0; j < EVENT_TYPES; j++) {
            total_rate += rates[i].rates[j];
        }
        
        printf("%-15d %-15.4f %-15.4f %-15.4f %-15.4f %-15.4f\n", 
               rates[i].window_start,
               rates[i].rates[EVENT_NEW],
               rates[i].rates[EVENT_MODIFY],
               rates[i].rates[EVENT_DELETE],
               rates[i].rates[EVENT_EXECUTE],
               total_rate);
    }
    
    // 分析事件率的自相关性
    analyze_autocorrelation(rates, num_rates);
    
    // 分析事件类型之间的相关性
    analyze_cross_correlation(rates, num_rates);
    
    // 使用EDM分析事件率
    analyze_event_rates_with_edm(rates, num_rates);
    
    printf("\n=== 分析完成 ===\n");
    
    return 0;
} 