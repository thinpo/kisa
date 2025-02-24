/**
 * orderbook_edm_analysis.c - Using Empirical Dynamic Modeling (EDM) to analyze Poisson arrival rates of orderbook events
 * 
 * This program simulates orderbook event flows and uses EDM to analyze arrival rates and patterns of different event types:
 * 1. New order events (New)
 * 2. Modify order events (Modify)
 * 3. Delete order events (Delete)
 * 4. Trade execution events (Execute)
 * 
 * Enhanced features:
 * - Command line parameter configuration
 * - Reading real data from files
 * - Output visualization data
 * - Multi-threaded data processing
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
#include <ctype.h>    // Add header file for isdigit function
#include <sys/stat.h> // Add header file for mkdir function

// Default parameter definitions (can be overridden via command line)
#define DEFAULT_MAX_EVENTS 1000   // Maximum number of events
#define DEFAULT_VECTOR_SIZE 8     // Vector size
#define DEFAULT_EVENT_TYPES 4     // Number of event types
#define DEFAULT_TIME_WINDOW 100   // Time window size
#define DEFAULT_EMB_DIM 3         // EDM embedding dimension
#define DEFAULT_TIME_DELAY 1      // Time delay
#define DEFAULT_NUM_NEIGHBORS 3   // Number of neighbors
#define DEFAULT_PRED_STEPS 5      // Prediction steps
#define DEFAULT_THREADS 4         // Default number of threads
#define MAX_FILENAME_LENGTH 256   // Maximum filename length

// Global configuration structure
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

// Event type enumeration
typedef enum {
    EVENT_NEW = 0,
    EVENT_MODIFY = 1,
    EVENT_DELETE = 2,
    EVENT_EXECUTE = 3
} EventType;

// Event structure
typedef struct {
    EventType type;
    int timestamp;
    double price;
    int quantity;
} OrderEvent;

// Event rate structure
typedef struct {
    int window_start;
    int window_end;
    int counts[DEFAULT_EVENT_TYPES];
    double rates[DEFAULT_EVENT_TYPES];
} EventRates;

// Distance array structure (moved to global for code clarity)
typedef struct {
    int index;
    int32_t distance;
} DistanceItem;

// Multi-threaded calculation work structure
typedef struct {
    int thread_id;
    int start_idx;
    int end_idx;
    void* data;
    void* result;
} ThreadWork;

// Global configuration
Config config;

// Helper function: Display usage help
void show_usage(const char* program_name) {
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  -h, --help                 Display this help information\n");
    printf("  -e, --events <number>      Set maximum number of events (default: %d)\n", DEFAULT_MAX_EVENTS);
    printf("  -w, --window <size>        Set time window size (default: %d)\n", DEFAULT_TIME_WINDOW);
    printf("  -d, --dimension <dimension> Set EDM embedding dimension (default: %d)\n", DEFAULT_EMB_DIM);
    printf("  -t, --threads <number>     Set number of threads (default: %d)\n", DEFAULT_THREADS);
    printf("  -i, --input <filename>     Read orderbook event data from file\n");
    printf("  -o, --output <directory>   Set output directory (default: .)\n");
    printf("  -p, --plot                 Generate data file for plotting\n");
    printf("\n");
}

// Parse command line arguments
void parse_arguments(int argc, char* argv[], Config* config) {
    // Set default configuration
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
    
    // Define long options
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
                    fprintf(stderr, "Error: Events number must be positive\n");
                    exit(1);
                }
                break;
            case 'w':
                config->time_window = atoi(optarg);
                if (config->time_window <= 0) {
                    fprintf(stderr, "Error: Time window size must be positive\n");
                    exit(1);
                }
                break;
            case 'd':
                config->edm_embedding_dim = atoi(optarg);
                if (config->edm_embedding_dim <= 0 || config->edm_embedding_dim > config->vector_size) {
                    fprintf(stderr, "Error: Embedding dimension must be positive and not greater than vector size(%d)\n", config->vector_size);
                    exit(1);
                }
                break;
            case 't':
                config->num_threads = atoi(optarg);
                if (config->num_threads <= 0) {
                    fprintf(stderr, "Error: Thread number must be positive\n");
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
                // getopt_long already outputs error message
                exit(1);
                break;
            default:
                abort();
        }
    }
    
    printf("Configuration: Maximum events=%d, Time window=%d, Embedding dimension=%d, Threads=%d\n", 
           config->max_events, config->time_window, config->edm_embedding_dim, config->num_threads);
    
    if (config->use_input_file) {
        printf("Reading data from file: %s\n", config->input_file);
    } else {
        printf("Using simulated data\n");
    }
    
    printf("Output directory: %s\n", config->output_dir);
    printf("Generate plotting data: %s\n", config->generate_plots ? "Yes" : "No");
}

// Helper function: Get vector element (inline for performance)
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

// Helper function: Set vector element (inline for performance)
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

// Helper function: Print vector content
void print_vector(const char* name, const vector_reg_t* v) {
    printf("%s: [", name);
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        printf("%d%s", get_vector_element(v, i), i < VECTOR_LENGTH-1 ? ", " : "");
    }
    printf("]\n");
}

// Helper function: Initialize vector
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

// Read orderbook events from CSV file
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
    
    // Read CSV file header (if any)
    char line[1024];
    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Error: Empty file or read error\n");
        fclose(file);
        return 0;
    }
    
    // Check if it's a CSV header, if so, skip
    if (strstr(line, "timestamp") != NULL || strstr(line, "time") != NULL) {
        // This is a title line, continue reading data
    } else {
        // Not a title line, go back to file start
        rewind(file);
    }
    
    // Read event data
    int count = 0;
    while (count < max_events && fgets(line, sizeof(line), file) != NULL) {
        // Parse CSV line
        // Expected format: timestamp,type,price,quantity
        char* token = strtok(line, ",");
        if (!token) continue;
        events[count].timestamp = atoi(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        // Parse event type (can be number or string)
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
                // Default to new order
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
    printf("Read %d orderbook events from file %s\n", count, filename);
    return count;
}

// Generate simulated orderbook events
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
    
    // Set base Poisson rates for different event types
    double base_rates[DEFAULT_EVENT_TYPES] = {0.4, 0.3, 0.2, 0.1}; // Base rates for new, modify, delete, execute
    
    // Generate periodic changes for each event type
    double amplitudes[DEFAULT_EVENT_TYPES] = {0.2, 0.15, 0.1, 0.05}; // Amplitudes
    double periods[DEFAULT_EVENT_TYPES] = {200, 150, 100, 50}; // Periods
    
    int current_time = 0;
    for (int i = 0; i < num_events; i++) {
        // Calculate current probability for each event type (base rate + periodic change)
        double probs[DEFAULT_EVENT_TYPES];
        double total_prob = 0;
        
        for (int j = 0; j < DEFAULT_EVENT_TYPES; j++) {
            // Add periodic change
            probs[j] = base_rates[j] + amplitudes[j] * sin(2 * M_PI * current_time / periods[j]);
            // Ensure probability is non-negative
            if (probs[j] < 0.01) probs[j] = 0.01;
            total_prob += probs[j];
        }
        
        // Normalize probabilities
        for (int j = 0; j < DEFAULT_EVENT_TYPES; j++) {
            probs[j] /= total_prob;
        }
        
        // Randomly select event type
        double r = (double)rand() / RAND_MAX;
        double cumulative = 0;
        EventType selected_type = EVENT_NEW; // Default
        
        for (int j = 0; j < DEFAULT_EVENT_TYPES; j++) {
            cumulative += probs[j];
            if (r <= cumulative) {
                selected_type = (EventType)j;
                break;
            }
        }
        
        // Generate event
        events[i].type = selected_type;
        events[i].timestamp = current_time;
        events[i].price = 100.0 + (rand() % 1000) / 100.0; // Price between 100-110
        events[i].quantity = 1 + rand() % 100; // Quantity between 1-100
        
        // Update time (using Poisson process to simulate event intervals)
        double lambda = 0.5; // Average event rate
        double u = (double)rand() / RAND_MAX;
        int time_increment = (int)(-log(u) / lambda); // Poisson interval
        if (time_increment < 1) time_increment = 1;
        current_time += time_increment;
    }
    
    printf("Generated %d simulated orderbook events\n", num_events);
}

// Calculate event rates within time window (multi-threaded version)
void* calculate_event_rates_thread(void* arg) {
    ThreadWork* work = (ThreadWork*)arg;
    OrderEvent* events = (OrderEvent*)work->data;
    EventRates* rates = (EventRates*)work->result;
    int start_idx = work->start_idx;
    int end_idx = work->end_idx;
    
    // This function assumes event rates array is already initialized
    for (int i = start_idx; i <= end_idx; i++) {
        int window_idx = (events[i].timestamp - rates[0].window_start) / config.time_window;
        if (window_idx >= 0 && window_idx < (config.max_events / config.time_window + 1)) {
            rates[window_idx].counts[events[i].type]++;
        }
    }
    
    return NULL;
}

// Calculate event rates within time window
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
    
    // Calculate how many time windows are needed
    int num_windows = (end_time - start_time) / config.time_window + 1;
    *num_rates = num_windows;
    
    // Initialize event rates array
    for (int i = 0; i < num_windows; i++) {
        rates[i].window_start = start_time + i * config.time_window;
        rates[i].window_end = rates[i].window_start + config.time_window - 1;
        
        for (int j = 0; j < config.event_types; j++) {
            rates[i].counts[j] = 0;
            rates[i].rates[j] = 0.0;
        }
    }
    
    // Multi-threaded calculation of event rates
    if (config.num_threads > 1 && num_events > 1000) {
        pthread_t threads[config.num_threads];
        ThreadWork work[config.num_threads];
        
        // Divide work
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
        
        // Create threads
        for (int i = 0; i < config.num_threads; i++) {
            if (pthread_create(&threads[i], NULL, calculate_event_rates_thread, &work[i]) != 0) {
                fprintf(stderr, "Error: Failed to create thread %d\n", i);
                // Fall back to single-thread mode
                for (int j = 0; j < i; j++) {
                    pthread_join(threads[j], NULL);
                }
                goto single_thread;
            }
        }
        
        // Wait for all threads to complete
        for (int i = 0; i < config.num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
    } else {
    single_thread:
        // Single-thread calculation (for small data set or fallback mode)
        for (int i = 0; i < num_events; i++) {
            int window_idx = (events[i].timestamp - start_time) / config.time_window;
            if (window_idx >= 0 && window_idx < num_windows) {
                rates[window_idx].counts[events[i].type]++;
            }
        }
    }
    
    // Calculate event rates
    for (int i = 0; i < num_windows; i++) {
        for (int j = 0; j < config.event_types; j++) {
            rates[i].rates[j] = (double)rates[i].counts[j] / config.time_window;
        }
    }
    
    printf("Calculated event rates for %d time windows\n", num_windows);
}

// Convert event rates to vector
void event_rates_to_vector(EventRates rates[], int window_idx, vector_reg_t* result) {
    if (rates == NULL || result == NULL) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    // Initialize result vector to zero
#ifdef __aarch64__
    result->low = vdupq_n_s32(0);
    result->high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = 0;
    }
#endif
    
    // Convert event rates to vector elements
    // First 4 elements are rates for each event type
    for (int i = 0; i < config.event_types && i < VECTOR_LENGTH; i++) {
        int32_t rate_scaled = (int32_t)(rates[window_idx].rates[i] * 1000); // Scale to convert to integer
        set_vector_element(result, i, rate_scaled);
    }
    
    // Last 4 elements are counts for each event type
    for (int i = 0; i < config.event_types && i + config.event_types < VECTOR_LENGTH; i++) {
        set_vector_element(result, i + config.event_types, rates[window_idx].counts[i]);
    }
}

// EDM time delay embedding
void time_delay_embedding(vector_reg_t* result, vector_reg_t* input, int delay, int embedding_dim) {
    if (result == NULL || input == NULL) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    if (delay <= 0 || embedding_dim <= 0 || embedding_dim > VECTOR_LENGTH) {
        fprintf(stderr, "Error: Invalid delay or embedding dimension\n");
        return;
    }
    
    printf("Executing time delay embedding (delay=%d, embedding dimension=%d)\n", delay, embedding_dim);
    
    // Initialize result vector to zero
#ifdef __aarch64__
    result->low = vdupq_n_s32(0);
    result->high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = 0;
    }
#endif
    
    // For each element of the vector, create time delay embedding
    for(int i = 0; i < embedding_dim; i++) {
        // Calculate current time point
        int time_point = i * delay;
        
        // If time point is within vector range
        if(time_point < VECTOR_LENGTH) {
            int32_t value = get_vector_element(input, time_point);
            set_vector_element(result, i, value);
        }
    }
    
    print_vector("Time delay embedding result", result);
}

// Calculate Euclidean distance
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

// Comparison function for quick sort
static int compare_distances(const void* a, const void* b) {
    const DistanceItem* item1 = (const DistanceItem*)a;
    const DistanceItem* item2 = (const DistanceItem*)b;
    
    if (item1->distance < item2->distance) return -1;
    if (item1->distance > item2->distance) return 1;
    return 0;
}

// EDM nearest neighbor search (using quick sort instead of bubble sort)
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
    
    printf("Executing nearest neighbor search (library size=%d, number of neighbors=%d)\n", library_size, num_neighbors);
    
    // Allocate distance array
    DistanceItem* distances = (DistanceItem*)malloc(library_size * sizeof(DistanceItem));
    if (distances == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    
    // Calculate distance between target vector and each vector in library
    for(int i = 0; i < library_size; i++) {
        distances[i].index = i;
        distances[i].distance = euclidean_distance(target, &library[i], embedding_dim);
    }
    
    // Use quick sort instead of bubble sort
    qsort(distances, library_size, sizeof(DistanceItem), compare_distances);
    
    // Get nearest neighbor indices
    for(int i = 0; i < num_neighbors && i < library_size; i++) {
        neighbor_indices[i] = distances[i].index;
        printf("Nearest neighbor %d: Index %d, Distance %d\n", i+1, neighbor_indices[i], distances[i].distance);
    }
    
    free(distances);
}

// EDM prediction
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
    
    printf("Executing EDM prediction (prediction steps=%d)\n", prediction_steps);
    
    // Initialize result vector to zero
#ifdef __aarch64__
    result->low = vdupq_n_s32(0);
    result->high = vdupq_n_s32(0);
#else
    for(int i = 0; i < VECTOR_LENGTH; i++) {
        (*result)[i] = 0;
    }
#endif
    
    // Find nearest neighbors
    int* neighbor_indices = (int*)malloc(num_neighbors * sizeof(int));
    if (neighbor_indices == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }
    
    find_nearest_neighbors(neighbor_indices, current_state, library, library_size, num_neighbors, embedding_dim);
    
    // Based on nearest neighbors, perform weighted average prediction
    int32_t total_weight = 0;
    
    for(int i = 0; i < num_neighbors; i++) {
        int neighbor_idx = neighbor_indices[i];
        
        // Calculate weight (simplified as inverse of distance)
        int32_t distance = euclidean_distance(current_state, &library[neighbor_idx], embedding_dim);
        int32_t weight = distance == 0 ? 1000 : 1000 / distance; // Avoid division by zero
        total_weight += weight;
        
        // For each prediction step, add future value to result
        for(int step = 1; step <= prediction_steps; step++) {
            // Ensure we don't go out of range of library
            if(neighbor_idx + step < library_size) {
                for(int j = 0; j < VECTOR_LENGTH; j++) {
                    int32_t future_val = get_vector_element(&library[neighbor_idx + step], j);
                    int32_t current_val = get_vector_element(result, j);
                    set_vector_element(result, j, current_val + future_val * weight);
                }
            }
        }
    }
    
    // Normalize result
    if(total_weight > 0) {
        for(int j = 0; j < VECTOR_LENGTH; j++) {
            int32_t val = get_vector_element(result, j);
            set_vector_element(result, j, val / total_weight);
        }
    }
    
    free(neighbor_indices);
    print_vector("EDM prediction result", result);
}

// Analyze autocorrelation of event rates
void analyze_autocorrelation(EventRates rates[], int num_rates, int generate_plot) {
    if (rates == NULL || num_rates <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    printf("\n=== Event rates autocorrelation analysis ===\n");
    
    // Maximum lag period
    int max_lag = num_rates / 4;
    if (max_lag > 20) max_lag = 20;
    if (max_lag <= 0) {
        printf("Insufficient data points to perform autocorrelation analysis\n");
        return;
    }
    
    // Create file for plotting (if needed)
    FILE* plot_file = NULL;
    if (generate_plot) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/autocorrelation_plot.dat", config.output_dir);
        plot_file = fopen(filename, "w");
        if (!plot_file) {
            fprintf(stderr, "Warning: Cannot create plot data file %s\n", filename);
        } else {
            fprintf(plot_file, "# Lag\tNew order\tModify order\tDelete order\tExecute\n");
        }
    }
    
    // Calculate autocorrelation for each event type
    for (int type = 0; type < config.event_types; type++) {
        const char* event_names[] = {"New order", "Modify order", "Delete order", "Execute"};
        printf("\nAutocorrelation coefficient for %s event:\n", event_names[type]);
        
        // Calculate mean
        double mean = 0.0;
        for (int i = 0; i < num_rates; i++) {
            mean += rates[i].rates[type];
        }
        mean /= num_rates;
        
        // Calculate variance
        double variance = 0.0;
        for (int i = 0; i < num_rates; i++) {
            double diff = rates[i].rates[type] - mean;
            variance += diff * diff;
        }
        variance /= num_rates;
        
        if (variance < 1e-10) {
            printf("Variance near zero, unable to calculate meaningful autocorrelation\n");
            continue;
        }
        
        // Calculate autocorrelation for different lag periods
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
            
            if (lag > 0) {  // Don't print lag 0 value (always 1)
                printf("Lag%d: %.4f", lag, autocorr);
                
                // Simple visualization
                printf(" |");
                int bars = (int)(fabs(autocorr) * 40);
                for (int b = 0; b < bars; b++) {
                    printf(autocorr >= 0 ? "+" : "-");
                }
                printf("|\n");
            }
        }
        
        // Write autocorrelation data to plotting file
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
        printf("\nAutocorrelation data saved to %s/autocorrelation_plot.dat, can use gnuplot or other tool to plot\n", config.output_dir);
    }
}

// Analyze correlation between event types
void analyze_cross_correlation(EventRates rates[], int num_rates, int generate_plot) {
    if (rates == NULL || num_rates <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    printf("\n=== Event type correlation analysis ===\n");
    
    const char* event_names[] = {"New order", "Modify order", "Delete order", "Execute"};
    
    // Calculate mean for each event type
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
    
    // Calculate standard deviation for each event type
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
    
    // Calculate correlation coefficients between event types
    printf("\nCorrelation matrix:\n");
    printf("%-12s", "");
    for (int type2 = 0; type2 < config.event_types; type2++) {
        printf("%-12s", event_names[type2]);
    }
    printf("\n");
    
    // Create file for plotting (if needed)
    FILE* plot_file = NULL;
    if (generate_plot) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/correlation_matrix.dat", config.output_dir);
        plot_file = fopen(filename, "w");
        if (!plot_file) {
            fprintf(stderr, "Warning: Cannot create plot data file %s\n", filename);
        } else {
            // Write header
            fprintf(plot_file, "# Correlation matrix\n# ");
            for (int type2 = 0; type2 < config.event_types; type2++) {
                fprintf(plot_file, "%s\t", event_names[type2]);
            }
            fprintf(plot_file, "\n");
        }
    }
    
    // Calculate and store correlation matrix
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
            // Check if standard deviation is near zero
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
        printf("\nCorrelation matrix data saved to %s/correlation_matrix.dat, can use gnuplot or other tool to plot heatmap\n", config.output_dir);
        
        // Create a simple Gnuplot script
        char script_filename[MAX_FILENAME_LENGTH];
        snprintf(script_filename, sizeof(script_filename), "%s/plot_correlation.gp", config.output_dir);
        FILE* script_file = fopen(script_filename, "w");
        if (script_file) {
            fprintf(script_file, "# Gnuplot script - Plot correlation matrix heatmap\n");
            fprintf(script_file, "set terminal png size 800,600\n");
            fprintf(script_file, "set output '%s/correlation_heatmap.png'\n", config.output_dir);
            fprintf(script_file, "set title 'Event type correlation heatmap'\n");
            fprintf(script_file, "set xlabel 'Event type'\n");
            fprintf(script_file, "set ylabel 'Event type'\n");
            fprintf(script_file, "set xtics ('New order' 0, 'Modify order' 1, 'Delete order' 2, 'Execute' 3)\n");
            fprintf(script_file, "set ytics ('New order' 0, 'Modify order' 1, 'Delete order' 2, 'Execute' 3)\n");
            fprintf(script_file, "set cbrange [-1:1]\n");
            fprintf(script_file, "set palette defined (-1 'blue', 0 'white', 1 'red')\n");
            fprintf(script_file, "set view map\n");
            fprintf(script_file, "set size square\n");
            fprintf(script_file, "set key off\n");
            fprintf(script_file, "splot '%s/correlation_matrix.dat' using 1:2:3 with pm3d\n", config.output_dir);
            fclose(script_file);
            printf("Generated Gnuplot script %s, can use command 'gnuplot %s' to generate heatmap\n", script_filename, script_filename);
        }
    }
    
    // Release memory
    for (int i = 0; i < config.event_types; i++) {
        free(correlation_matrix[i]);
    }
    free(correlation_matrix);
    free(means);
    free(stddevs);
}

// Chi-square test to determine if Poisson distribution is valid
double chi_square_test_poisson(int observed[], int n, double lambda) {
    if (observed == NULL || n <= 0 || lambda <= 0) {
        return -1.0;
    }
    
    // Calculate expected frequencies
    double* expected = (double*)malloc(n * sizeof(double));
    if (expected == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return -1.0;
    }
    
    int total_observations = 0;
    for (int i = 0; i < n; i++) {
        total_observations += observed[i];
    }
    
    // Calculate expected frequencies for Poisson distribution
    for (int i = 0; i < n; i++) {
        // P(X = k) = (lambda^k * e^-lambda) / k!
        double p = exp(-lambda);
        for (int k = 1; k <= i; k++) {
            p *= lambda / k;
        }
        expected[i] = p * total_observations;
    }
    
    // Calculate chi-square statistic
    double chi_square = 0.0;
    for (int i = 0; i < n; i++) {
        if (expected[i] >= 5.0) {  // Chi-square requires expected frequency >= 5
            double diff = observed[i] - expected[i];
            chi_square += (diff * diff) / expected[i];
        }
    }
    
    free(expected);
    return chi_square;
}

// Use EDM to analyze event rates and save results
void analyze_event_rates_with_edm(EventRates rates[], int num_rates, int generate_plot) {
    if (rates == NULL || num_rates <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }
    
    printf("\n=== Use EDM to analyze event rates ===\n");
    
    // If data points are too few, unable to perform EDM analysis
    if (num_rates < config.edm_embedding_dim + config.edm_prediction_steps) {
        printf("Insufficient data points to perform EDM analysis\n");
        return;
    }
    
    // Create file for saving results
    FILE* results_file = NULL;
    if (generate_plot) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/edm_analysis_results.txt", config.output_dir);
        results_file = fopen(filename, "w");
        if (!results_file) {
            fprintf(stderr, "Warning: Cannot create results file %s\n", filename);
        } else {
            fprintf(results_file, "==== EDM analysis results ====\n\n");
            fprintf(results_file, "Parameter settings:\n");
            fprintf(results_file, "- Embedding dimension: %d\n", config.edm_embedding_dim);
            fprintf(results_file, "- Time delay: %d\n", config.edm_time_delay);
            fprintf(results_file, "- Number of neighbors: %d\n", config.edm_num_neighbors);
            fprintf(results_file, "- Prediction steps: %d\n\n", config.edm_prediction_steps);
        }
    }
    
    // Create file for plotting prediction data
    FILE* prediction_file = NULL;
    if (generate_plot) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/edm_predictions.dat", config.output_dir);
        prediction_file = fopen(filename, "w");
        if (!prediction_file) {
            fprintf(stderr, "Warning: Cannot create prediction data file %s\n", filename);
        } else {
            fprintf(prediction_file, "# Time\tActual new order rate\tPredicted new order rate\tActual modify rate\tPredicted modify rate\tActual delete rate\tPredicted delete rate\tActual execute rate\tPredicted execute rate\n");
        }
    }
    
    // Convert event rates to vector sequence
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
    
    // Create embedding library
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
    
    // Perform EDM prediction for each event type
    const char* event_names[] = {"New order", "Modify order", "Delete order", "Execute"};
    
    // Array for storing predicted values
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
        printf("\nAnalyze dynamic of %s event:\n", event_names[type]);
        if (results_file) fprintf(results_file, "\n== %s event analysis ==\n", event_names[type]);
        
        // Select last time point for prediction
        vector_reg_t current_state;
        time_delay_embedding(&current_state, &rate_vectors[num_rates-1], config.edm_time_delay, config.edm_embedding_dim);
        
        // Use EDM for prediction
        vector_reg_t prediction;
        edm_predict(&prediction, &current_state, embedded_library, num_rates, 
                   config.edm_num_neighbors, config.edm_embedding_dim, config.edm_prediction_steps);
        
        // Extract predicted rate for this event type
        int32_t predicted_rate_i32 = get_vector_element(&prediction, type);
        double actual_rate = (double)predicted_rate_i32 / 1000.0; // Convert back to actual rate
        predicted_rates[type] = actual_rate;
        
        printf("%s event predicted rate: %.4f\n", event_names[type], actual_rate);
        if (results_file) fprintf(results_file, "Predicted rate: %.4f\n", actual_rate);
        
        // Analyze nonlinear characteristics of sequence
        printf("Analyze nonlinear characteristics of %s event:\n", event_names[type]);
        
        // Calculate variance of actual sequence
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
        
        printf("Mean: %.4f, Variance: %.4f\n", mean, variance);
        if (results_file) fprintf(results_file, "Mean: %.4f, Variance: %.4f\n", mean, variance);
        
        // Calculate nonlinear index of sequence (simplified version)
        double nonlinearity = 0.0;
        for (int i = 1; i < num_rates; i++) {
            double diff = rates[i].rates[type] - rates[i-1].rates[type];
            nonlinearity += fabs(diff);
        }
        nonlinearity /= (num_rates - 1);
        
        printf("Nonlinear index: %.4f\n", nonlinearity);
        if (results_file) fprintf(results_file, "Nonlinear index: %.4f\n", nonlinearity);
        
        // Check if Poisson process is valid
        // For Poisson process, variance should be close to mean
        double poisson_ratio = variance / mean;
        printf("Variance/mean ratio: %.4f ", poisson_ratio);
        if (results_file) fprintf(results_file, "Variance/mean ratio: %.4f ", poisson_ratio);
        
        if (fabs(poisson_ratio - 1.0) < 0.2) {
            printf("(close to 1, valid Poisson process characteristic)\n");
            if (results_file) fprintf(results_file, "(close to 1, valid Poisson process characteristic)\n");
        } else if (poisson_ratio > 1.0) {
            printf("(greater than 1, indicating overdispersion, possibly cluster arrival)\n");
            if (results_file) fprintf(results_file, "(greater than 1, indicating overdispersion, possibly cluster arrival)\n");
        } else {
            printf("(less than 1, indicating underdispersion, possibly regular arrival)\n");
            if (results_file) fprintf(results_file, "(less than 1, indicating underdispersion, possibly regular arrival)\n");
        }
        
        // Perform chi-square test
        if (num_rates >= 5) {  // Ensure enough observations
            // Create observed frequency array (simplified as 5 intervals)
            int observed[5] = {0};
            for (int i = 0; i < num_rates; i++) {
                int count = (int)(rates[i].rates[type] * config.time_window);
                if (count >= 4) {
                    observed[4]++;
                } else {
                    observed[count]++;
                }
            }
            
            // Perform chi-square test
            double chi_square = chi_square_test_poisson(observed, 5, mean * config.time_window);
            if (chi_square >= 0) {
                printf("Chi-square test value: %.4f (degrees of freedom=3, 95%% critical value=7.815)\n", chi_square);
                if (results_file) fprintf(results_file, "Chi-square test value: %.4f (degrees of freedom=3, 95%% critical value=7.815)\n", chi_square);
                
                if (chi_square < 7.815) {
                    printf("Conclusion: At 95%% confidence level, unable to reject data conforming to Poisson distribution assumption\n");
                    if (results_file) fprintf(results_file, "Conclusion: At 95%% confidence level, unable to reject data conforming to Poisson distribution assumption\n");
                } else {
                    printf("Conclusion: At 95%% confidence level, reject data conforming to Poisson distribution assumption\n");
                    if (results_file) fprintf(results_file, "Conclusion: At 95%% confidence level, reject data conforming to Poisson distribution assumption\n");
                }
            }
        }
    }
    
    // Write prediction results to data file
    if (prediction_file) {
        for (int i = 0; i < num_rates; i++) {
            fprintf(prediction_file, "%d", rates[i].window_start);
            
            for (int type = 0; type < config.event_types; type++) {
                fprintf(prediction_file, "\t%.6f", rates[i].rates[type]);
                
                // For last few points, add predicted value (others use NaN)
                if (i >= num_rates - config.edm_prediction_steps) {
                    fprintf(prediction_file, "\t%.6f", predicted_rates[type]);
                } else {
                    fprintf(prediction_file, "\tNaN");
                }
            }
            fprintf(prediction_file, "\n");
        }
        
        fclose(prediction_file);
        printf("\nEDM prediction data saved to %s/edm_predictions.dat, can use gnuplot or other tool to plot\n", config.output_dir);
        
        // Create a simple Gnuplot script
        char script_filename[MAX_FILENAME_LENGTH];
        snprintf(script_filename, sizeof(script_filename), "%s/plot_predictions.gp", config.output_dir);
        FILE* script_file = fopen(script_filename, "w");
        if (script_file) {
            fprintf(script_file, "# Gnuplot script - Plot EDM prediction results\n");
            fprintf(script_file, "set terminal png size 1200,800\n");
            fprintf(script_file, "set output '%s/edm_predictions.png'\n", config.output_dir);
            fprintf(script_file, "set title 'EDM event rate prediction'\n");
            fprintf(script_file, "set xlabel 'Time'\n");
            fprintf(script_file, "set ylabel 'Event rate'\n");
            fprintf(script_file, "set grid\n");
            fprintf(script_file, "set key outside\n");
            
            // Plot actual and predicted values for each event type
            fprintf(script_file, "plot '%s/edm_predictions.dat' using 1:2 with lines title 'Actual new order rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:3 with points pt 7 title 'Predicted new order rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:4 with lines title 'Actual modify rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:5 with points pt 7 title 'Predicted modify rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:6 with lines title 'Actual delete rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:7 with points pt 7 title 'Predicted delete rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:8 with lines title 'Actual execute rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/edm_predictions.dat' using 1:9 with points pt 7 title 'Predicted execute rate'\n", config.output_dir);
            
            fclose(script_file);
            printf("Generated Gnuplot script %s, can use command 'gnuplot %s' to generate prediction plot\n", script_filename, script_filename);
        }
    }
    
    if (results_file) {
        fclose(results_file);
        printf("\nEDM analysis results saved to %s/edm_analysis_results.txt\n", config.output_dir);
    }
    
    // Clean up
    free(predicted_rates);
    free(rate_vectors);
    free(embedded_library);
}

// Create output directory (if it doesn't exist)
void ensure_output_directory_exists(const char* dir_path) {
    // Check if directory exists
    if (access(dir_path, F_OK) != 0) {
        // Directory doesn't exist, create it
        printf("Creating output directory: %s\n", dir_path);
        
        // Use mkdir to create directory (permissions set to 755)
        if (mkdir(dir_path, 0755) != 0) {
            fprintf(stderr, "Warning: Unable to create directory %s\n", dir_path);
        }
    }
}

// Main function
int main(int argc, char* argv[]) {
    printf("=== Orderbook event Poisson arrival rate analysis ===\n\n");
    
    // Parse command line arguments
    parse_arguments(argc, argv, &config);
    
    // Ensure output directory exists
    ensure_output_directory_exists(config.output_dir);
    
    // Initialize execution unit
    init_execution_unit();
    
    // Verify event number within range
    if (config.max_events <= 0) {
        fprintf(stderr, "Error: MAX_EVENTS must be positive\n");
        return 1;
    }
    
    // Allocate memory
    OrderEvent* events = (OrderEvent*)malloc(config.max_events * sizeof(OrderEvent));
    if (!events) {
        fprintf(stderr, "Error: Memory allocation failed for events\n");
        return 1;
    }
    
    int num_events = 0;
    
    // Read or generate event data
    if (config.use_input_file) {
        // Read event data from file
        num_events = read_events_from_file(config.input_file, events, config.max_events);
        if (num_events == 0) {
            fprintf(stderr, "Error: Failed to read events from file %s\n", config.input_file);
            free(events);
            return 1;
        }
    } else {
        // Use simulated event data
        num_events = config.max_events;
        generate_simulated_events(events, num_events);
    }
    
    // Calculate event rates
    EventRates* rates = (EventRates*)malloc((config.max_events / config.time_window + 1) * sizeof(EventRates));
    if (!rates) {
        fprintf(stderr, "Error: Memory allocation failed for rates\n");
        free(events);
        return 1;
    }
    
    int num_rates;
    calculate_event_rates(events, num_events, rates, &num_rates);
    
    // Print event rates
    printf("\n=== Event rate statistics ===\n");
    printf("%-15s %-15s %-15s %-15s %-15s %-15s\n", 
           "Time window", "New order rate", "Modify order rate", "Delete order rate", "Execute rate", "Total event rate");
    
    // Create file for plotting event rate data
    FILE* rates_file = NULL;
    if (config.generate_plots) {
        char filename[MAX_FILENAME_LENGTH];
        snprintf(filename, sizeof(filename), "%s/event_rates.dat", config.output_dir);
        rates_file = fopen(filename, "w");
        if (rates_file) {
            fprintf(rates_file, "# Time window  New order rate  Modify order rate  Delete order rate  Execute rate  Total event rate\n");
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
        
        // Write event rate data to file
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
        printf("\nEvent rate data saved to %s/event_rates.dat\n", config.output_dir);
        
        // Create a simple Gnuplot script
        char script_filename[MAX_FILENAME_LENGTH];
        snprintf(script_filename, sizeof(script_filename), "%s/plot_rates.gp", config.output_dir);
        FILE* script_file = fopen(script_filename, "w");
        if (script_file) {
            fprintf(script_file, "# Gnuplot script - Plot event rates\n");
            fprintf(script_file, "set terminal png size 1200,800\n");
            fprintf(script_file, "set output '%s/event_rates.png'\n", config.output_dir);
            fprintf(script_file, "set title 'Orderbook event rates'\n");
            fprintf(script_file, "set xlabel 'Time'\n");
            fprintf(script_file, "set ylabel 'Event rate (events/time unit)'\n");
            fprintf(script_file, "set grid\n");
            fprintf(script_file, "set key outside\n");
            fprintf(script_file, "plot '%s/event_rates.dat' using 1:2 with lines title 'New order rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/event_rates.dat' using 1:3 with lines title 'Modify order rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/event_rates.dat' using 1:4 with lines title 'Delete order rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/event_rates.dat' using 1:5 with lines title 'Execute rate', \\\n", config.output_dir);
            fprintf(script_file, "     '%s/event_rates.dat' using 1:6 with lines title 'Total event rate'\n", config.output_dir);
            fclose(script_file);
            printf("Generated Gnuplot script %s, can use command 'gnuplot %s' to generate event rate plot\n", script_filename, script_filename);
        }
    }
    
    // Analyze autocorrelation of event rates
    analyze_autocorrelation(rates, num_rates, config.generate_plots);
    
    // Analyze correlation between event types
    analyze_cross_correlation(rates, num_rates, config.generate_plots);
    
    // Use EDM to analyze event rates
    analyze_event_rates_with_edm(rates, num_rates, config.generate_plots);
    
    printf("\n=== Analysis completed ===\n");
    if (config.generate_plots) {
        printf("All analysis data and plotting scripts saved to directory: %s\n", config.output_dir);
    }
    
    // Release memory
    free(events);
    free(rates);
    
    return 0;
} 