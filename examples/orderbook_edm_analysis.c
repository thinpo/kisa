/**
 * orderbook_edm_analysis.c - 使用经验动态建模(EDM)分析订单簿事件的泊松到达率
 * 
 * 这个程序模拟订单簿事件流，并使用EDM分析不同类型事件的到达率和模式：
 * 1. 新订单事件 (New)
 * 2. 修改订单事件 (Modify)
 * 3. 删除订单事件 (Delete)
 * 4. 交易执行事件 (Execute)
 * 
 * 增强功能:
 * - 命令行参数配置
 * - 从文件读取真实数据
 * - 输出可视化数据
 * - 多线程数据处理
 */

#include "../include/kisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <unistd.h>
#include <pthread.h>
#include <ctype.h>    // 添加头文件用于isdigit函数
#include <sys/stat.h> // 添加头文件用于mkdir函数

// 默认参数定义（可通过命令行覆盖）
#define DEFAULT_MAX_EVENTS 1000   // 最大事件数
#define DEFAULT_VECTOR_SIZE 8     // 向量大小
#define DEFAULT_EVENT_TYPES 4     // 事件类型数量
#define DEFAULT_TIME_WINDOW 100   // 时间窗口大小
#define DEFAULT_EMB_DIM 3         // EDM嵌入维度
#define DEFAULT_TIME_DELAY 1      // 时间延迟
#define DEFAULT_NUM_NEIGHBORS 3   // 近邻数量
#define DEFAULT_PRED_STEPS 5      // 预测步数
#define DEFAULT_THREADS 4         // 默认线程数
#define MAX_FILENAME_LENGTH 256   // 最大文件名长度

// 全局配置结构体
typedef struct {
    int max_events;
    int vector_size;
    int event_types;
    int time_window;
    int edm_embedding_dim;
    int edm_time_delay;
    int edm_num_neighbors;
    int edm_prediction_steps;
    int num_threads;
    char input_file[MAX_FILENAME_LENGTH];
    char output_dir[MAX_FILENAME_LENGTH];
    int use_input_file;
    int generate_plots;
} Config;

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
    int counts[DEFAULT_EVENT_TYPES];
    double rates[DEFAULT_EVENT_TYPES];
} EventRates;

// 距离数组结构体（移到全局以提高代码清晰度）
typedef struct {
    int index;
    int32_t distance;
} DistanceItem;

// 多线程计算的工作结构体
typedef struct {
    int thread_id;
    int start_idx;
    int end_idx;
    void* data;
    void* result;
} ThreadWork;

// 全局配置
Config config;

// 辅助函数：显示使用帮助
void show_usage(const char* program_name) {
    printf("使用: %s [选项]\n\n", program_name);
    printf("选项:\n");
    printf("  -h, --help                显示此帮助信息\n");
    printf("  -e, --events <数量>       设置最大事件数 (默认: %d)\n", DEFAULT_MAX_EVENTS);
    printf("  -w, --window <大小>       设置时间窗口大小 (默认: %d)\n", DEFAULT_TIME_WINDOW);
    printf("  -d, --dimension <维度>    设置EDM嵌入维度 (默认: %d)\n", DEFAULT_EMB_DIM);
    printf("  -t, --threads <数量>      设置线程数 (默认: %d)\n", DEFAULT_THREADS);
    printf("  -i, --input <文件名>      从文件读取订单簿事件数据\n");
    printf("  -o, --output <目录>       设置输出目录 (默认: .)\n");
    printf("  -p, --plot                生成可用于绘图的数据文件\n");
    printf("\n");
}

// 解析命令行参数
void parse_arguments(int argc, char* argv[], Config* config) {
    // 设置默认配置
    config->max_events = DEFAULT_MAX_EVENTS;
    config->vector_size = DEFAULT_VECTOR_SIZE;
    config->event_types = DEFAULT_EVENT_TYPES;
    config->time_window = DEFAULT_TIME_WINDOW;
    config->edm_embedding_dim = DEFAULT_EMB_DIM;
    config->edm_time_delay = DEFAULT_TIME_DELAY;
    config->edm_num_neighbors = DEFAULT_NUM_NEIGHBORS;
    config->edm_prediction_steps = DEFAULT_PRED_STEPS;
    config->num_threads = DEFAULT_THREADS;
    strncpy(config->input_file, "", MAX_FILENAME_LENGTH);
    strncpy(config->output_dir, ".", MAX_FILENAME_LENGTH);
    config->use_input_file = 0;
    config->generate_plots = 0;
    
    // 定义长选项
    static struct option long_options[] = {
        {"help",      no_argument,       0, 'h'},
        {"events",    required_argument, 0, 'e'},
        {"window",    required_argument, 0, 'w'},
        {"dimension", required_argument, 0, 'd'},
        {"threads",   required_argument, 0, 't'},
        {"input",     required_argument, 0, 'i'},
        {"output",    required_argument, 0, 'o'},
        {"plot",      no_argument,       0, 'p'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "he:w:d:t:i:o:p", long_options, &option_index)) != -1) {
        switch (c) {
            case 'h':
                show_usage(argv[0]);
                exit(0);
                break;
            case 'e':
                config->max_events = atoi(optarg);
                if (config->max_events <= 0) {
                    fprintf(stderr, "错误: 事件数必须为正数\n");
                    exit(1);
                }
                break;
            case 'w':
                config->time_window = atoi(optarg);
                if (config->time_window <= 0) {
                    fprintf(stderr, "错误: 时间窗口大小必须为正数\n");
                    exit(1);
                }
                break;
            case 'd':
                config->edm_embedding_dim = atoi(optarg);
                if (config->edm_embedding_dim <= 0 || config->edm_embedding_dim > config->vector_size) {
                    fprintf(stderr, "错误: 嵌入维度必须为正数且不大于向量大小(%d)\n", config->vector_size);
                    exit(1);
                }
                break;
            case 't':
                config->num_threads = atoi(optarg);
                if (config->num_threads <= 0) {
                    fprintf(stderr, "错误: 线程数必须为正数\n");
                    exit(1);
                }
                break;
            case 'i':
                strncpy(config->input_file, optarg, MAX_FILENAME_LENGTH - 1);
                config->use_input_file = 1;
                break;
            case 'o':
                strncpy(config->output_dir, optarg, MAX_FILENAME_LENGTH - 1);
                break;
            case 'p':
                config->generate_plots = 1;
                break;
            case '?':
                // getopt_long已经输出了错误消息
                exit(1);
                break;
            default:
                abort();
        }
    }
    
    printf("配置: 最大事件数=%d, 时间窗口=%d, 嵌入维度=%d, 线程数=%d\n", 
           config->max_events, config->time_window, config->edm_embedding_dim, config->num_threads);
    
    if (config->use_input_file) {
        printf("从文件读取数据: %s\n", config->input_file);
    } else {
        printf("使用模拟生成的数据\n");
    }
    
    printf("输出目录: %s\n", config->output_dir);
    printf("生成绘图数据: %s\n", config->generate_plots ? "是" : "否");
}

// 辅助函数：获取向量元素 (内联以提高性能)
static inline int32_t get_vector_element(const vector_reg_t* reg, int i) {
    if (i < 0 || i >= VECTOR_LENGTH) {
        fprintf(stderr, "Error: Vector index out of bounds: %d\n", i);
        return 0;
    }
    
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

// 辅助函数：设置向量元素 (内联以提高性能)
static inline void set_vector_element(vector_reg_t* reg, int i, int32_t value) {
    if (i < 0 || i >= VECTOR_LENGTH) {
        fprintf(stderr, "Error: Vector index out of bounds: %d\n", i);
        return;
    }
    
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
    if (v == NULL) {
        fprintf(stderr, "Error: Cannot initialize NULL vector\n");
        return;
    }
    
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

// 从CSV文件读取订单簿事件
int read_events_from_file(const char* filename, OrderEvent events[], int max_events) {
    if (events == NULL) {
        fprintf(stderr, "Error: Invalid events array\n");
        return 0;
    }
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 0;
    }
    
    // 读取CSV文件头（如果有）
    char line[1024];
    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Error: Empty file or read error\n");
        fclose(file);
        return 0;
    }
    
    // 检查是否是CSV头，如果是，则跳过
    if (strstr(line, "timestamp") != NULL || strstr(line, "time") != NULL) {
        // 这是一个标题行，继续读取数据
    } else {
        // 不是标题行，回到文件开头
        rewind(file);
    }
    
    // 读取事件数据
    int count = 0;
    while (count < max_events && fgets(line, sizeof(line), file) != NULL) {
        // 解析CSV行
        // 预期格式: timestamp,type,price,quantity
        char* token = strtok(line, ",");
        if (!token) continue;
        events[count].timestamp = atoi(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        // 解析事件类型（可以是数字或字符串）
        if (isdigit(token[0])) {
            events[count].type = (EventType)atoi(token);
        } else {
            if (strcmp(token, "NEW") == 0 || strcmp(token, "new") == 0) {
                events[count].type = EVENT_NEW;
            } else if (strcmp(token, "MODIFY") == 0 || strcmp(token, "modify") == 0) {
                events[count].type = EVENT_MODIFY;
            } else if (strcmp(token, "DELETE") == 0 || strcmp(token, "delete") == 0) {
                events[count].type = EVENT_DELETE;
            } else if (strcmp(token, "EXECUTE") == 0 || strcmp(token, "execute") == 0) {
                events[count].type = EVENT_EXECUTE;
            } else {
                // 默认为新订单
                events[count].type = EVENT_NEW;
            }
        }
        
        token = strtok(NULL, ",");
        if (!token) continue;
        events[count].price = atof(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        events[count].quantity = atoi(token);
        
        count++;
    }
    
    fclose(file);
    printf("从文件 %s 读取了 %d 个订单簿事件\n", filename, count);
    return count;
}

// 生成模拟的订单簿事件
void generate_simulated_events(OrderEvent events[], int num_events) {
    if (events == NULL) {
        fprintf(stderr, "Error: Invalid events array\n");
        return;
    }
    
    if (num_events <= 0 || num_events > config.max_events) {
        fprintf(stderr, "Error: Invalid number of events: %d (max: %d)\n", num_events, config.max_events);
        return;
    }
    
    srand(time(NULL));
    
    // 设置不同事件类型的基础泊松率
    double base_rates[DEFAULT_EVENT_TYPES] = {0.4, 0.3, 0.2, 0.1}; // 新订单、修改、删除、执行的基础率
    
    // 为每种事件类型生成周期性变化
    double amplitudes[DEFAULT_EVENT_TYPES] = {0.2, 0.15, 0.1, 0.05}; // 振幅
    double periods[DEFAULT_EVENT_TYPES] = {200, 150, 100, 50}; // 周期
    
    int current_time = 0;
    for (int i = 0; i < num_events; i++) {
        // 计算每种事件类型的当前概率（基础率+周期变化）
        double probs[DEFAULT_EVENT_TYPES];
        double total_prob = 0;
        
        for (int j = 0; j < DEFAULT_EVENT_TYPES; j++) {
            // 添加周期性变化
            probs[j] = base_rates[j] + amplitudes[j] * sin(2 * M_PI * current_time / periods[j]);
            // 确保概率非负
            if (probs[j] < 0.01) probs[j] = 0.01;
            total_prob += probs[j];
        }
        
        // 归一化概率
        for (int j = 0; j < DEFAULT_EVENT_TYPES; j++) {
            probs[j] /= total_prob;
        }
        
        // 随机选择事件类型
        double r = (double)rand() / RAND_MAX;
        double cumulative = 0;
        EventType selected_type = EVENT_NEW; // 默认
        
        for (int j = 0; j < DEFAULT_EVENT_TYPES; j++) {
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

// 计算时间窗口内的事件率（多线程版本）
void* calculate_event_rates_thread(void* arg) {
    ThreadWork* work = (ThreadWork*)arg;
    OrderEvent* events = (OrderEvent*)work->data;
    EventRates* rates = (EventRates*)work->result;
    int start_idx = work->start_idx;
    int end_idx = work->end_idx;
    
    // 这个函数假设事件率数组已经初始化好了
    for (int i = start_idx; i <= end_idx; i++) {
        int window_idx = (events[i].timestamp - rates[0].window_start) / config.time_window;
        if (window_idx >= 0 && window_idx < (config.max_events / config.time_window + 1)) {
            rates[window_idx].counts[events[i].type]++;
        }
    }
    
    return NULL;
}

// 计算时间窗口内的事件率
void calculate_event_rates(OrderEvent events[], int num_events, EventRates rates[], int *num_rates) {
    if (events == NULL || rates == NULL || num_rates == NULL) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    if (num_events == 0) {
        *num_rates = 0;
        return;
    }
    
    int start_time = events[0].timestamp;
    int end_time = events[num_events-1].timestamp;
    
    // 计算需要多少个时间窗口
    int num_windows = (end_time - start_time) / config.time_window + 1;
    *num_rates = num_windows;
    
    // 初始化事件率数组
    for (int i = 0; i < num_windows; i++) {
        rates[i].window_start = start_time + i * config.time_window;
        rates[i].window_end = rates[i].window_start + config.time_window - 1;
        
        for (int j = 0; j < config.event_types; j++) {
            rates[i].counts[j] = 0;
            rates[i].rates[j] = 0.0;
        }
    }
    
    // 多线程计算事件率
    if (config.num_threads > 1 && num_events > 1000) {
        pthread_t threads[config.num_threads];
        ThreadWork work[config.num_threads];
        
        // 划分工作
        int events_per_thread = num_events / config.num_threads;
        int start_idx = 0;
        
        for (int i = 0; i < config.num_threads; i++) {
            work[i].thread_id = i;
            work[i].start_idx = start_idx;
            work[i].end_idx = (i == config.num_threads - 1) ? (num_events - 1) : (start_idx + events_per_thread - 1);
            work[i].data = events;
            work[i].result = rates;
            
            start_idx += events_per_thread;
        }
        
        // 创建线程
        for (int i = 0; i < config.num_threads; i++) {
            if (pthread_create(&threads[i], NULL, calculate_event_rates_thread, &work[i]) != 0) {
                fprintf(stderr, "Error: Failed to create thread %d\n", i);
                // 回退到单线程模式
                for (int j = 0; j < i; j++) {
                    pthread_join(threads[j], NULL);
                }
                goto single_thread;
            }
        }
        
        // 等待所有线程完成
        for (int i = 0; i < config.num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
    } else {
    single_thread:
        // 单线程计算（用于小数据集或回退模式）
        for (int i = 0; i < num_events; i++) {
            int window_idx = (events[i].timestamp - start_time) / config.time_window;
            if (window_idx >= 0 && window_idx < num_windows) {
                rates[window_idx].counts[events[i].type]++;
            }
        }
    }
    
    // 计算事件率
    for (int i = 0; i < num_windows; i++) {
        for (int j = 0; j < config.event_types; j++) {
            rates[i].rates[j] = (double)rates[i].counts[j] / config.time_window;
        }
    }
    
    printf("计算了%d个时间窗口的事件率\n", num_windows);
}

// 将事件率转换为向量
void event_rates_to_vector(EventRates rates[], int window_idx, vector_reg_t* result) {
    if (rates == NULL || result == NULL) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
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
    for (int i = 0; i < config.event_types && i < VECTOR_LENGTH; i++) {
        int32_t rate_scaled = (int32_t)(rates[window_idx].rates[i] * 1000); // 缩放以转换为整数
        set_vector_element(result, i, rate_scaled);
    }
    
    // 后4个元素是各类事件的计数
    for (int i = 0; i < config.event_types && i + config.event_types < VECTOR_LENGTH; i++) {
        set_vector_element(result, i + config.event_types, rates[window_idx].counts[i]);
    }
}

// EDM时间延迟嵌入
void time_delay_embedding(vector_reg_t* result, vector_reg_t* input, int delay, int embedding_dim) {
    if (result == NULL || input == NULL) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    if (delay <= 0 || embedding_dim <= 0 || embedding_dim > VECTOR_LENGTH) {
        fprintf(stderr, "Error: Invalid delay or embedding dimension\n");
        return;
    }
    
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
    if (v1 == NULL || v2 == NULL) {
        fprintf(stderr, "Error: Invalid input vectors\n");
        return INT32_MAX;
    }
    
    if (dim <= 0 || dim > VECTOR_LENGTH) {
        fprintf(stderr, "Error: Invalid dimension: %d\n", dim);
        return INT32_MAX;
    }
    
    int64_t sum_sq = 0;
    
    for(int i = 0; i < dim; i++) {
        int32_t diff = get_vector_element(v1, i) - get_vector_element(v2, i);
        sum_sq += (int64_t)diff * diff;
    }
    
    return (int32_t)sqrt((double)sum_sq);
}

// 比较函数用于快速排序
static int compare_distances(const void* a, const void* b) {
    const DistanceItem* item1 = (const DistanceItem*)a;
    const DistanceItem* item2 = (const DistanceItem*)b;
    
    if (item1->distance < item2->distance) return -1;
    if (item1->distance > item2->distance) return 1;
    return 0;
}

// EDM近邻搜索 (使用快速排序替代冒泡排序)
void find_nearest_neighbors(int* neighbor_indices, vector_reg_t* target, 
                           vector_reg_t library[], int library_size, 
                           int num_neighbors, int embedding_dim) {
    if (neighbor_indices == NULL || target == NULL || library == NULL) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    if (library_size <= 0 || num_neighbors <= 0 || num_neighbors > library_size) {
        fprintf(stderr, "Error: Invalid library size or number of neighbors\n");
        return;
    }
    
    printf("执行近邻搜索 (库大小=%d, 近邻数=%d)\n", library_size, num_neighbors);
    
    // 分配距离数组
    DistanceItem* distances = (DistanceItem*)malloc(library_size * sizeof(DistanceItem));
    if (distances == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    
    // 计算目标向量与库中每个向量的距离
    for(int i = 0; i < library_size; i++) {
        distances[i].index = i;
        distances[i].distance = euclidean_distance(target, &library[i], embedding_dim);
    }
    
    // 使用快速排序替代冒泡排序
    qsort(distances, library_size, sizeof(DistanceItem), compare_distances);
    
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
    if (result == NULL || current_state == NULL || library == NULL) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    if (library_size <= 0 || num_neighbors <= 0 || num_neighbors > library_size ||
        prediction_steps <= 0 || embedding_dim <= 0) {
        fprintf(stderr, "Error: Invalid parameters for EDM prediction\n");
        return;
    }
    
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
    if (neighbor_indices == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    
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
void analyze_autocorrelation(EventRates rates[], int num_rates, int generate_plot) {
    if (rates == NULL || num_rates <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    printf("\n=== 事件率自相关性分析 ===\n");
    
    // 最大滞后期
    int max_lag = num_rates / 4;
    if (max_lag > 20) max_lag = 20;
    if (max_lag <= 0) {
        printf("数据点不足，无法进行自相关分析\n");
        return;
    }
    
    // 创建用于绘图的文件（如果需要）
    FILE* plot_file = NULL;
    if (generate_plot) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/autocorrelation_plot.dat", config.output_dir);
        plot_file = fopen(filename, "w");
        if (!plot_file) {
            fprintf(stderr, "Warning: Cannot create plot data file %s\n", filename);
        } else {
            fprintf(plot_file, "# 滞后\t新订单\t修改订单\t删除订单\t交易执行\n");
        }
    }
    
    // 对每种事件类型计算自相关
    for (int type = 0; type < config.event_types; type++) {
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
        
        if (variance < 1e-10) {
            printf("方差接近零，无法计算有意义的自相关\n");
            continue;
        }
        
        // 计算不同滞后期的自相关
        double* autocorr_values = (double*)malloc((max_lag + 1) * sizeof(double));
        if (!autocorr_values) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            if (plot_file) fclose(plot_file);
            return;
        }
        
        for (int lag = 0; lag <= max_lag; lag++) {
            double autocorr = 0.0;
            for (int i = 0; i < num_rates - lag; i++) {
                double diff1 = rates[i].rates[type] - mean;
                double diff2 = rates[i + lag].rates[type] - mean;
                autocorr += diff1 * diff2;
            }
            autocorr /= (num_rates - lag) * variance;
            autocorr_values[lag] = autocorr;
            
            if (lag > 0) {  // 不打印滞后0的值（总是1）
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
        
        // 将自相关数据写入绘图文件
        if (plot_file) {
            for (int lag = 0; lag <= max_lag; lag++) {
                if (type == 0) {
                    fprintf(plot_file, "%d", lag);
                }
                fprintf(plot_file, "\t%.6f", autocorr_values[lag]);
                if (type == config.event_types - 1) {
                    fprintf(plot_file, "\n");
                }
            }
        }
        
        free(autocorr_values);
    }
    
    if (plot_file) {
        fclose(plot_file);
        printf("\n自相关数据已保存到 %s/autocorrelation_plot.dat，可使用gnuplot或其他工具绘图\n", config.output_dir);
    }
}

// 分析事件类型之间的相关性
void analyze_cross_correlation(EventRates rates[], int num_rates, int generate_plot) {
    if (rates == NULL || num_rates <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    printf("\n=== 事件类型间的相关性分析 ===\n");
    
    const char* event_names[] = {"新订单", "修改订单", "删除订单", "交易执行"};
    
    // 计算每种事件类型的均值
    double* means = (double*)malloc(config.event_types * sizeof(double));
    if (!means) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    
    for (int type = 0; type < config.event_types; type++) {
        means[type] = 0.0;
        for (int i = 0; i < num_rates; i++) {
            means[type] += rates[i].rates[type];
        }
        means[type] /= num_rates;
    }
    
    // 计算每种事件类型的标准差
    double* stddevs = (double*)malloc(config.event_types * sizeof(double));
    if (!stddevs) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(means);
        return;
    }
    
    for (int type = 0; type < config.event_types; type++) {
        stddevs[type] = 0.0;
        for (int i = 0; i < num_rates; i++) {
            double diff = rates[i].rates[type] - means[type];
            stddevs[type] += diff * diff;
        }
        stddevs[type] = sqrt(stddevs[type] / num_rates);
    }
    
    // 计算事件类型之间的相关系数
    printf("\n相关系数矩阵:\n");
    printf("%-12s", "");
    for (int type2 = 0; type2 < config.event_types; type2++) {
        printf("%-12s", event_names[type2]);
    }
    printf("\n");
    
    // 创建用于绘图的文件（如果需要）
    FILE* plot_file = NULL;
    if (generate_plot) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/correlation_matrix.dat", config.output_dir);
        plot_file = fopen(filename, "w");
        if (!plot_file) {
            fprintf(stderr, "Warning: Cannot create plot data file %s\n", filename);
        } else {
            // 写入头部
            fprintf(plot_file, "# 相关矩阵\n# ");
            for (int type2 = 0; type2 < config.event_types; type2++) {
                fprintf(plot_file, "%s\t", event_names[type2]);
            }
            fprintf(plot_file, "\n");
        }
    }
    
    // 计算并存储相关矩阵
    double** correlation_matrix = (double**)malloc(config.event_types * sizeof(double*));
    if (!correlation_matrix) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(means);
        free(stddevs);
        if (plot_file) fclose(plot_file);
        return;
    }
    
    for (int type1 = 0; type1 < config.event_types; type1++) {
        correlation_matrix[type1] = (double*)malloc(config.event_types * sizeof(double));
        if (!correlation_matrix[type1]) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            for (int j = 0; j < type1; j++) {
                free(correlation_matrix[j]);
            }
            free(correlation_matrix);
            free(means);
            free(stddevs);
            if (plot_file) fclose(plot_file);
            return;
        }
        
        printf("%-12s", event_names[type1]);
        if (plot_file) fprintf(plot_file, "%s\t", event_names[type1]);
        
        for (int type2 = 0; type2 < config.event_types; type2++) {
            // 检查标准差是否接近零
            if (stddevs[type1] < 1e-10 || stddevs[type2] < 1e-10) {
                correlation_matrix[type1][type2] = 0.0;
                printf("%-12s", "N/A");
                if (plot_file) fprintf(plot_file, "NaN\t");
                continue;
            }
            
            double correlation = 0.0;
            
            for (int i = 0; i < num_rates; i++) {
                double diff1 = rates[i].rates[type1] - means[type1];
                double diff2 = rates[i].rates[type2] - means[type2];
                correlation += diff1 * diff2;
            }
            
            correlation /= num_rates * stddevs[type1] * stddevs[type2];
            correlation_matrix[type1][type2] = correlation;
            
            printf("%-12.4f", correlation);
            if (plot_file) fprintf(plot_file, "%.6f\t", correlation);
        }
        printf("\n");
        if (plot_file) fprintf(plot_file, "\n");
    }
    
    if (plot_file) {
        fclose(plot_file);
        printf("\n相关矩阵数据已保存到 %s/correlation_matrix.dat，可使用gnuplot或其他工具绘制热图\n", config.output_dir);
        
        // 创建一个简单的Gnuplot脚本
        char script_filename[MAX_FILENAME_LENGTH];
        snprintf(script_filename, sizeof(script_filename), "%s/plot_correlation.gp", config.output_dir);
        FILE* script_file = fopen(script_filename, "w");
        if (script_file) {
            fprintf(script_file, "# Gnuplot脚本 - 绘制相关矩阵热图\n");
            fprintf(script_file, "set terminal png size 800,600\n");
            fprintf(script_file, "set output '%s/correlation_heatmap.png'\n", config.output_dir);
            fprintf(script_file, "set title '事件类型间的相关性热图'\n");
            fprintf(script_file, "set xlabel '事件类型'\n");
            fprintf(script_file, "set ylabel '事件类型'\n");
            fprintf(script_file, "set xtics ('新订单' 0, '修改订单' 1, '删除订单' 2, '交易执行' 3)\n");
            fprintf(script_file, "set ytics ('新订单' 0, '修改订单' 1, '删除订单' 2, '交易执行' 3)\n");
            fprintf(script_file, "set cbrange [-1:1]\n");
            fprintf(script_file, "set palette defined (-1 'blue', 0 'white', 1 'red')\n");
            fprintf(script_file, "set view map\n");
            fprintf(script_file, "set size square\n");
            fprintf(script_file, "set key off\n");
            fprintf(script_file, "splot '%s/correlation_matrix.dat' using 1:2:3 with pm3d\n", config.output_dir);
            fclose(script_file);
            printf("已生成Gnuplot脚本 %s，可使用命令 'gnuplot %s' 生成热图\n", script_filename, script_filename);
        }
    }
    
    // 释放内存
    for (int i = 0; i < config.event_types; i++) {
        free(correlation_matrix[i]);
    }
    free(correlation_matrix);
    free(means);
    free(stddevs);
}

// 卡方检验判断是否符合泊松分布
double chi_square_test_poisson(int observed[], int n, double lambda) {
    if (observed == NULL || n <= 0 || lambda <= 0) {
        return -1.0;
    }
    
    // 计算期望频率
    double* expected = (double*)malloc(n * sizeof(double));
    if (expected == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return -1.0;
    }
    
    int total_observations = 0;
    for (int i = 0; i < n; i++) {
        total_observations += observed[i];
    }
    
    // 计算泊松分布的期望频率
    for (int i = 0; i < n; i++) {
        // P(X = k) = (lambda^k * e^-lambda) / k!
        double p = exp(-lambda);
        for (int k = 1; k <= i; k++) {
            p *= lambda / k;
        }
        expected[i] = p * total_observations;
    }
    
    // 计算卡方统计量
    double chi_square = 0.0;
    for (int i = 0; i < n; i++) {
        if (expected[i] >= 5.0) {  // 卡方要求期望频率 >= 5
            double diff = observed[i] - expected[i];
            chi_square += (diff * diff) / expected[i];
        }
    }
    
    free(expected);
    return chi_square;
}

// 使用EDM分析事件率并保存结果
void analyze_event_rates_with_edm(EventRates rates[], int num_rates, int generate_plot) {
    if (rates == NULL || num_rates <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    printf("\n=== 使用EDM分析事件率 ===\n");
    
    // 如果数据点太少，无法进行EDM分析
    if (num_rates < config.edm_embedding_dim + config.edm_prediction_steps) {
        printf("数据点不足，无法进行EDM分析\n");
        return;
    }
    
    // 创建用于保存结果的文件
    FILE* results_file = NULL;
    if (generate_plot) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/edm_analysis_results.txt", config.output_dir);
        results_file = fopen(filename, "w");
        if (!results_file) {
            fprintf(stderr, "Warning: Cannot create results file %s\n", filename);
        } else {
            fprintf(results_file, "==== EDM分析结果 ====\n\n");
            fprintf(results_file, "参数设置:\n");
            fprintf(results_file, "- 嵌入维度: %d\n", config.edm_embedding_dim);
            fprintf(results_file, "- 时间延迟: %d\n", config.edm_time_delay);
            fprintf(results_file, "- 近邻数量: %d\n", config.edm_num_neighbors);
            fprintf(results_file, "- 预测步数: %d\n\n", config.edm_prediction_steps);
        }
    }
    
    // 创建用于绘图的预测数据文件
    FILE* prediction_file = NULL;
    if (generate_plot) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/edm_predictions.dat", config.output_dir);
        prediction_file = fopen(filename, "w");
        if (!prediction_file) {
            fprintf(stderr, "Warning: Cannot create prediction data file %s\n", filename);
        } else {
            fprintf(prediction_file, "# 时间\t实际新订单率\t预测新订单率\t实际修改率\t预测修改率\t实际删除率\t预测删除率\t实际执行率\t预测执行率\n");
        }
    }
    
    // 将事件率转换为向量序列
    vector_reg_t* rate_vectors = (vector_reg_t*)malloc(num_rates * sizeof(vector_reg_t));
    if (rate_vectors == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        if (results_file) fclose(results_file);
        if (prediction_file) fclose(prediction_file);
        return;
    }
    
    for (int i = 0; i < num_rates; i++) {
        event_rates_to_vector(rates, i, &rate_vectors[i]);
    }
    
    // 创建嵌入库
    vector_reg_t* embedded_library = (vector_reg_t*)malloc(num_rates * sizeof(vector_reg_t));
    if (embedded_library == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(rate_vectors);
        if (results_file) fclose(results_file);
        if (prediction_file) fclose(prediction_file);
        return;
    }
    
    for (int i = 0; i < num_rates; i++) {
        time_delay_embedding(&embedded_library[i], &rate_vectors[i], config.edm_time_delay, config.edm_embedding_dim);
    }
    
    // 对每种事件类型进行EDM预测
    const char* event_names[] = {"新订单", "修改订单", "删除订单", "交易执行"};
    
    // 用于存储预测值的数组
    double* predicted_rates = (double*)malloc(config.event_types * sizeof(double));
    if (!predicted_rates) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(rate_vectors);
        free(embedded_library);
        if (results_file) fclose(results_file);
        if (prediction_file) fclose(prediction_file);
        return;
    }
    
    for (int type = 0; type < config.event_types; type++) {
        printf("\n分析%s事件的动态:\n", event_names[type]);
        if (results_file) fprintf(results_file, "\n== %s事件分析 ==\n", event_names[type]);
        
        // 选择最后一个时间点进行预测
        vector_reg_t current_state;
        time_delay_embedding(&current_state, &rate_vectors[num_rates-1], config.edm_time_delay, config.edm_embedding_dim);
        
        // 使用EDM进行预测
        vector_reg_t prediction;
        edm_predict(&prediction, &current_state, embedded_library, num_rates, 
                   config.edm_num_neighbors, config.edm_embedding_dim, config.edm_prediction_steps);
        
        // 提取该事件类型的预测率
        int32_t predicted_rate_i32 = get_vector_element(&prediction, type);
        double actual_rate = (double)predicted_rate_i32 / 1000.0; // 转换回实际率
        predicted_rates[type] = actual_rate;
        
        printf("%s事件的预测率: %.4f\n", event_names[type], actual_rate);
        if (results_file) fprintf(results_file, "预测率: %.4f\n", actual_rate);
        
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
        if (results_file) fprintf(results_file, "均值: %.4f, 方差: %.4f\n", mean, variance);
        
        // 计算序列的非线性指标（简化版）
        double nonlinearity = 0.0;
        for (int i = 1; i < num_rates; i++) {
            double diff = rates[i].rates[type] - rates[i-1].rates[type];
            nonlinearity += fabs(diff);
        }
        nonlinearity /= (num_rates - 1);
        
        printf("非线性指标: %.4f\n", nonlinearity);
        if (results_file) fprintf(results_file, "非线性指标: %.4f\n", nonlinearity);
        
        // 判断是否符合泊松过程
        // 对于泊松过程，方差应该接近均值
        double poisson_ratio = variance / mean;
        printf("方差/均值比率: %.4f ", poisson_ratio);
        if (results_file) fprintf(results_file, "方差/均值比率: %.4f ", poisson_ratio);
        
        if (fabs(poisson_ratio - 1.0) < 0.2) {
            printf("(接近1，符合泊松过程特性)\n");
            if (results_file) fprintf(results_file, "(接近1，符合泊松过程特性)\n");
        } else if (poisson_ratio > 1.0) {
            printf("(大于1，表现出过度分散，可能是集群到达)\n");
            if (results_file) fprintf(results_file, "(大于1，表现出过度分散，可能是集群到达)\n");
        } else {
            printf("(小于1，表现出欠分散，可能是规则到达)\n");
            if (results_file) fprintf(results_file, "(小于1，表现出欠分散，可能是规则到达)\n");
        }
        
        // 进行卡方检验
        if (num_rates >= 5) {  // 确保有足够的观测值
            // 创建观测频率数组 (简化为5个区间)
            int observed[5] = {0};
            for (int i = 0; i < num_rates; i++) {
                int count = (int)(rates[i].rates[type] * config.time_window);
                if (count >= 4) {
                    observed[4]++;
                } else {
                    observed[count]++;
                }
            }
            
            // 进行卡方检验
            double chi_square = chi_square_test_poisson(observed, 5, mean * config.time_window);
            if (chi_square >= 0) {
                printf("卡方检验值: %.4f (自由度=3, 95%%临界值=7.815)\n", chi_square);
                if (results_file) fprintf(results_file, "卡方检验值: %.4f (自由度=3, 95%%临界值=7.815)\n", chi_square);
                
                if (chi_square < 7.815) {
                    printf("结论: 在95%%置信水平下，无法拒绝数据符合泊松分布的假设\n");
                    if (results_file) fprintf(results_file, "结论: 在95%%置信水平下，无法拒绝数据符合泊松分布的假设\n");
                } else {
                    printf("结论: 在95%%置信水平下，拒绝数据符合泊松分布的假设\n");
                    if (results_file) fprintf(results_file, "结论: 在95%%置信水平下，拒绝数据符合泊松分布的假设\n");
                }
            }
        }
    }
    
    // 将预测结果写入数据文件
    if (prediction_file) {
        for (int i = 0; i < num_rates; i++) {
            fprintf(prediction_file, "%d", rates[i].window_start);
            
            for (int type = 0; type < config.event_types; type++) {
                fprintf(prediction_file, "\t%.6f", rates[i].rates[type]);
                
                // 对于最后几个点，添加预测值（其他点使用NaN）
                if (i >= num_rates - config.edm_prediction_steps) {
                    fprintf(prediction_file, "\t%.6f", predicted_rates[type]);
                } else {
                    fprintf(prediction_file, "\tNaN");
                }
            }
            fprintf(prediction_file, "\n");
        }
        
        fclose(prediction_file);
        printf("\nEDM预测数据已保存到 %s/edm_predictions.dat，可使用gnuplot或其他工具绘图\n", config.output_dir);
        
        // 创建一个简单的Gnuplot脚本
        char script_filename[MAX_FILENAME_LENGTH];
        snprintf(script_filename, sizeof(script_filename), "%s/plot_predictions.gp", config.output_dir);
        FILE* script_file = fopen(script_filename, "w");
        if (script_file) {
            fprintf(script_file, "# Gnuplot脚本 - 绘制EDM预测结果\n");
            fprintf(script_file, "set terminal png size 1200,800\n");
            fprintf(script_file, "set output '%s/edm_predictions.png'\n", config.output_dir);
            fprintf(script_file, "set title 'EDM事件率预测'\n");
            fprintf(script_file, "set xlabel '时间'\n");
            fprintf(script_file, "set ylabel '事件率'\n");
            fprintf(script_file, "set grid\n");
            fprintf(script_file, "set key outside\n");
            
            // 为每种事件类型绘制实际值和预测值
            fprintf(script_file, "plot '%s/edm_predictions.dat' using 1:2 with lines title '实际新订单率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:3 with points pt 7 title '预测新订单率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:4 with lines title '实际修改率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:5 with points pt 7 title '预测修改率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:6 with lines title '实际删除率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:7 with points pt 7 title '预测删除率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:8 with lines title '实际执行率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:9 with points pt 7 title '预测执行率'\n", config.output_dir);
            
            fclose(script_file);
            printf("已生成Gnuplot脚本 %s，可使用命令 'gnuplot %s' 生成预测图\n", script_filename, script_filename);
        }
    }
    
    if (results_file) {
        fclose(results_file);
        printf("\nEDM分析结果已保存到 %s/edm_analysis_results.txt\n", config.output_dir);
    }
    
    // 清理
    free(predicted_rates);
    free(rate_vectors);
    free(embedded_library);
}

// 创建输出目录（如果不存在）
void ensure_output_directory_exists(const char* dir_path) {
    // 检查目录是否存在
    if (access(dir_path, F_OK) != 0) {
        // 目录不存在，创建它
        printf("创建输出目录: %s\n", dir_path);
        
        // 使用mkdir创建目录（权限设置为755）
        if (mkdir(dir_path, 0755) != 0) {
            fprintf(stderr, "警告: 无法创建目录 %s\n", dir_path);
        }
    }
}

// 主函数
int main(int argc, char* argv[]) {
    printf("=== 订单簿事件泊松到达率分析 ===\n\n");
    
    // 解析命令行参数
    parse_arguments(argc, argv, &config);
    
    // 确保输出目录存在
    ensure_output_directory_exists(config.output_dir);
    
    // 初始化执行单元
    init_execution_unit();
    
    // 验证事件数量在范围内
    if (config.max_events <= 0) {
        fprintf(stderr, "Error: MAX_EVENTS must be positive\n");
        return 1;
    }
    
    // 分配内存
    OrderEvent* events = (OrderEvent*)malloc(config.max_events * sizeof(OrderEvent));
    if (!events) {
        fprintf(stderr, "Error: Memory allocation failed for events\n");
        return 1;
    }
    
    int num_events = 0;
    
    // 读取或生成事件数据
    if (config.use_input_file) {
        // 从文件读取事件数据
        num_events = read_events_from_file(config.input_file, events, config.max_events);
        if (num_events == 0) {
            fprintf(stderr, "Error: Failed to read events from file %s\n", config.input_file);
            free(events);
            return 1;
        }
    } else {
        // 使用模拟生成的事件数据
        num_events = config.max_events;
        generate_simulated_events(events, num_events);
    }
    
    // 计算事件率
    EventRates* rates = (EventRates*)malloc((config.max_events / config.time_window + 1) * sizeof(EventRates));
    if (!rates) {
        fprintf(stderr, "Error: Memory allocation failed for rates\n");
        free(events);
        return 1;
    }
    
    int num_rates;
    calculate_event_rates(events, num_events, rates, &num_rates);
    
    // 打印事件率
    printf("\n=== 事件率统计 ===\n");
    printf("%-15s %-15s %-15s %-15s %-15s %-15s\n", 
           "时间窗口", "新订单率", "修改订单率", "删除订单率", "交易执行率", "总事件率");
    
    // 创建用于绘图的事件率数据文件
    FILE* rates_file = NULL;
    if (config.generate_plots) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/event_rates.dat", config.output_dir);
        rates_file = fopen(filename, "w");
        if (rates_file) {
            fprintf(rates_file, "# 时间窗口 新订单率 修改订单率 删除订单率 交易执行率 总事件率\n");
        }
    }
    
    for (int i = 0; i < num_rates; i++) {
        double total_rate = 0;
        for (int j = 0; j < config.event_types; j++) {
            total_rate += rates[i].rates[j];
        }
        
        printf("%-15d %-15.4f %-15.4f %-15.4f %-15.4f %-15.4f\n", 
               rates[i].window_start,
               rates[i].rates[EVENT_NEW],
               rates[i].rates[EVENT_MODIFY],
               rates[i].rates[EVENT_DELETE],
               rates[i].rates[EVENT_EXECUTE],
               total_rate);
        
        // 将事件率数据写入文件
        if (rates_file) {
            fprintf(rates_file, "%d %.6f %.6f %.6f %.6f %.6f\n", 
                   rates[i].window_start,
                   rates[i].rates[EVENT_NEW],
                   rates[i].rates[EVENT_MODIFY],
                   rates[i].rates[EVENT_DELETE],
                   rates[i].rates[EVENT_EXECUTE],
                   total_rate);
        }
    }
    
    if (rates_file) {
        fclose(rates_file);
        printf("\n事件率数据已保存到 %s/event_rates.dat\n", config.output_dir);
        
        // 创建一个简单的Gnuplot脚本
        char script_filename[MAX_FILENAME_LENGTH];
        snprintf(script_filename, sizeof(script_filename), "%s/plot_rates.gp", config.output_dir);
        FILE* script_file = fopen(script_filename, "w");
        if (script_file) {
            fprintf(script_file, "# Gnuplot脚本 - 绘制事件率\n");
            fprintf(script_file, "set terminal png size 1200,800\n");
            fprintf(script_file, "set output '%s/event_rates.png'\n", config.output_dir);
            fprintf(script_file, "set title '订单簿事件率'\n");
            fprintf(script_file, "set xlabel '时间'\n");
            fprintf(script_file, "set ylabel '事件率 (事件/时间单位)'\n");
            fprintf(script_file, "set grid\n");
            fprintf(script_file, "set key outside\n");
            fprintf(script_file, "plot '%s/event_rates.dat' using 1:2 with lines title '新订单率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/event_rates.dat' using 1:3 with lines title '修改订单率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/event_rates.dat' using 1:4 with lines title '删除订单率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/event_rates.dat' using 1:5 with lines title '交易执行率', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/event_rates.dat' using 1:6 with lines title '总事件率'\n", config.output_dir);
            fclose(script_file);
            printf("已生成Gnuplot脚本 %s，可使用命令 'gnuplot %s' 生成事件率图\n", script_filename, script_filename);
        }
    }
    
    // 分析事件率的自相关性
    analyze_autocorrelation(rates, num_rates, config.generate_plots);
    
    // 分析事件类型之间的相关性
    analyze_cross_correlation(rates, num_rates, config.generate_plots);
    
    // 使用EDM分析事件率
    analyze_event_rates_with_edm(rates, num_rates, config.generate_plots);
    
    printf("\n=== 分析完成 ===\n");
    if (config.generate_plots) {
        printf("所有分析数据和绘图脚本已保存到目录: %s\n", config.output_dir);
    }
    
    // 释放内存
    free(events);
    free(rates);
    
    return 0;
} 