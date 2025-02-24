/**
 * orderbook_edm_analysis.c - Using Empirical Dynamic Modeling (EDM) to analyze Poisson arrival rates of orderbook events
 * 
 * This program simulates orderbook event flows and uses EDM to analyze arrival rates and patterns of different event types:
 * 1. New order events (New)
 * 2. Modify order events (Modify)
 * 3. Delete order events (Delete)
 * 4. Trade execution events (Execute)
 * 
 * The program uses EDM to predict future event rates and analyze dynamic relationships between events.
 */

#include "../../include/kisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Parameter definitions
#define MAX_EVENTS 1000          // Maximum number of events
#define VECTOR_SIZE 8            // Vector size
#define EVENT_TYPES 4            // Number of event types
#define TIME_WINDOW 100          // Time window size
#define EDM_EMBEDDING_DIM 3      // EDM embedding dimension
#define EDM_TIME_DELAY 1         // Time delay
#define EDM_NUM_NEIGHBORS 3      // Number of neighbors
#define EDM_PREDICTION_STEPS 5   // Prediction steps

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
    int counts[EVENT_TYPES];
    double rates[EVENT_TYPES];
} EventRates;

// Helper function: Get vector element
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

// Helper function: Set vector element
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

// Generate simulated orderbook events
void generate_simulated_events(OrderEvent events[], int num_events) {
    srand(time(NULL));
    
    // Set base Poisson rates for different event types
    double base_rates[EVENT_TYPES] = {0.4, 0.3, 0.2, 0.1}; // Base rates for new, modify, delete, execute
    
    // Set amplitudes and periods for periodic changes
    double amplitudes[EVENT_TYPES] = {0.2, 0.15, 0.1, 0.05}; // Amplitudes
    double periods[EVENT_TYPES] = {200, 150, 100, 50}; // Periods
    
    int current_time = 0;
    for (int i = 0; i < num_events; i++) {
        // Calculate current probabilities for each event type (base rate + periodic change)
        double probs[EVENT_TYPES];
        double total_prob = 0;
        
        for (int j = 0; j < EVENT_TYPES; j++) {
            // Add periodic change
            probs[j] = base_rates[j] + amplitudes[j] * sin(2 * M_PI * current_time / periods[j]);
            // Ensure non-negative probabilities
            if (probs[j] < 0.01) probs[j] = 0.01;
            total_prob += probs[j];
        }
        
        // Normalize probabilities
        for (int j = 0; j < EVENT_TYPES; j++) {
            probs[j] /= total_prob;
        }
        
        // Randomly select event type
        double r = (double)rand() / RAND_MAX;
        double cumulative = 0;
        EventType selected_type = EVENT_NEW; // Default
        
        for (int j = 0; j < EVENT_TYPES; j++) {
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

// Calculate event rates within a time window
void calculate_event_rates(OrderEvent events[], int num_events, EventRates rates[], int *num_rates) {
    if (num_events == 0) {
        *num_rates = 0;
        return;
    }
    
    int start_time = events[0].timestamp;
    int end_time = events[num_events-1].timestamp;
    
    // Calculate how many time windows are needed
    int num_windows = (end_time - start_time) / TIME_WINDOW + 1;
    *num_rates = num_windows;
    
    // Initialize event rates array
    for (int i = 0; i < num_windows; i++) {
        rates[i].window_start = start_time + i * TIME_WINDOW;
        rates[i].window_end = rates[i].window_start + TIME_WINDOW - 1;
        
        for (int j = 0; j < EVENT_TYPES; j++) {
            rates[i].counts[j] = 0;
            rates[i].rates[j] = 0.0;
        }
    }
    
    // Count events in each time window
    for (int i = 0; i < num_events; i++) {
        int window_idx = (events[i].timestamp - start_time) / TIME_WINDOW;
        if (window_idx >= 0 && window_idx < num_windows) {
            rates[window_idx].counts[events[i].type]++;
        }
    }
    
    // Calculate event rates
    for (int i = 0; i < num_windows; i++) {
        for (int j = 0; j < EVENT_TYPES; j++) {
            rates[i].rates[j] = (double)rates[i].counts[j] / TIME_WINDOW;
        }
    }
    
    printf("Calculated event rates for %d time windows\n", num_windows);
}

// Convert event rates to vector
void event_rates_to_vector(EventRates rates[], int window_idx, vector_reg_t* result) {
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
    // First 4 elements are rates of different event types
    for (int i = 0; i < EVENT_TYPES && i < VECTOR_LENGTH; i++) {
        int32_t rate_scaled = (int32_t)(rates[window_idx].rates[i] * 1000); // Scale to convert to integer
        set_vector_element(result, i, rate_scaled);
    }
    
    // Next 4 elements are counts of different event types
    for (int i = 0; i < EVENT_TYPES && i + EVENT_TYPES < VECTOR_LENGTH; i++) {
        set_vector_element(result, i + EVENT_TYPES, rates[window_idx].counts[i]);
    }
}

// EDM time delay embedding
void time_delay_embedding(vector_reg_t* result, vector_reg_t* input, int delay, int embedding_dim) {
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
    
    // For each element of the vector, create delay embedding
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
    int64_t sum_sq = 0;
    
    for(int i = 0; i < dim; i++) {
        int32_t diff = get_vector_element(v1, i) - get_vector_element(v2, i);
        sum_sq += (int64_t)diff * diff;
    }
    
    return (int32_t)sqrt((double)sum_sq);
}

// EDM nearest neighbor search
void find_nearest_neighbors(int* neighbor_indices, vector_reg_t* target, 
                           vector_reg_t library[], int library_size, 
                           int num_neighbors, int embedding_dim) {
    printf("Executing nearest neighbor search (library size=%d, number of neighbors=%d)\n", library_size, num_neighbors);
    
    // Distance array
    typedef struct {
        int index;
        int32_t distance;
    } DistanceItem;
    
    DistanceItem* distances = (DistanceItem*)malloc(library_size * sizeof(DistanceItem));
    
    // Calculate distance between target vector and each vector in library
    for(int i = 0; i < library_size; i++) {
        distances[i].index = i;
        distances[i].distance = euclidean_distance(target, &library[i], embedding_dim);
    }
    
    // Simple bubble sort to find nearest neighbors
    for(int i = 0; i < library_size - 1; i++) {
        for(int j = 0; j < library_size - i - 1; j++) {
            if(distances[j].distance > distances[j + 1].distance) {
                DistanceItem temp = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = temp;
            }
        }
    }
    
    // Get nearest neighbor indices
    for(int i = 0; i < num_neighbors && i < library_size; i++) {
        neighbor_indices[i] = distances[i].index;
        printf("Neighbor %d: Index %d, Distance %d\n", i+1, neighbor_indices[i], distances[i].distance);
    }
    
    free(distances);
}

// EDM prediction
void edm_predict(vector_reg_t* result, vector_reg_t* current_state, 
                vector_reg_t library[], int library_size, 
                int num_neighbors, int embedding_dim, int prediction_steps) {
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
    find_nearest_neighbors(neighbor_indices, current_state, library, library_size, num_neighbors, embedding_dim);
    
    // Based on weighted average of nearest neighbors
    int32_t total_weight = 0;
    
    for(int i = 0; i < num_neighbors; i++) {
        int neighbor_idx = neighbor_indices[i];
        
        // Calculate weight (simplified as inverse of distance)
        int32_t distance = euclidean_distance(current_state, &library[neighbor_idx], embedding_dim);
        int32_t weight = distance == 0 ? 1000 : 1000 / distance; // Avoid division by zero
        total_weight += weight;
        
        // For each prediction step, add future value to result
        for(int step = 1; step <= prediction_steps; step++) {
            // Ensure we don't go out of library range
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
void analyze_autocorrelation(EventRates rates[], int num_rates) {
    printf("\n=== Event rates autocorrelation analysis ===\n");
    
    // Maximum lag period
    int max_lag = num_rates / 4;
    if (max_lag > 20) max_lag = 20;
    
    // Calculate autocorrelation for each event type
    for (int type = 0; type < EVENT_TYPES; type++) {
        const char* event_names[] = {"New order", "Modify order", "Delete order", "Trade execution"};
        printf("\nAutocorrelation coefficient of %s events:\n", event_names[type]);
        
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
        
        // Calculate autocorrelation for different lag periods
        for (int lag = 1; lag <= max_lag; lag++) {
            double autocorr = 0.0;
            for (int i = 0; i < num_rates - lag; i++) {
                double diff1 = rates[i].rates[type] - mean;
                double diff2 = rates[i + lag].rates[type] - mean;
                autocorr += diff1 * diff2;
            }
            autocorr /= (num_rates - lag) * variance;
            
            printf("Lag %d: %.4f", lag, autocorr);
            
            // Simple visualization
            printf(" |");
            int bars = (int)(fabs(autocorr) * 40);
            for (int b = 0; b < bars; b++) {
                printf(autocorr >= 0 ? "+" : "-");
            }
            printf("|\n");
        }
    }
}

// Analyze correlation between event types
void analyze_cross_correlation(EventRates rates[], int num_rates) {
    printf("\n=== Event type correlation analysis ===\n");
    
    const char* event_names[] = {"New order", "Modify order", "Delete order", "Trade execution"};
    
    // Calculate mean for each event type
    double means[EVENT_TYPES] = {0};
    for (int type = 0; type < EVENT_TYPES; type++) {
        for (int i = 0; i < num_rates; i++) {
            means[type] += rates[i].rates[type];
        }
        means[type] /= num_rates;
    }
    
    // Calculate standard deviation for each event type
    double stddevs[EVENT_TYPES] = {0};
    for (int type = 0; type < EVENT_TYPES; type++) {
        for (int i = 0; i < num_rates; i++) {
            double diff = rates[i].rates[type] - means[type];
            stddevs[type] += diff * diff;
        }
        stddevs[type] = sqrt(stddevs[type] / num_rates);
    }
    
    // Calculate correlation coefficient between event types
    printf("\nCorrelation coefficient matrix:\n");
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

// Analyze event rates using EDM
void analyze_event_rates_with_edm(EventRates rates[], int num_rates) {
    printf("\n=== Analyzing event rates using EDM ===\n");
    
    // If data points are too few, cannot perform EDM analysis
    if (num_rates < EDM_EMBEDDING_DIM + EDM_PREDICTION_STEPS) {
        printf("Insufficient data points for EDM analysis\n");
        return;
    }
    
    // Convert event rates to vector sequence
    vector_reg_t* rate_vectors = (vector_reg_t*)malloc(num_rates * sizeof(vector_reg_t));
    for (int i = 0; i < num_rates; i++) {
        event_rates_to_vector(rates, i, &rate_vectors[i]);
    }
    
    // Create embedding library
    vector_reg_t* embedded_library = (vector_reg_t*)malloc(num_rates * sizeof(vector_reg_t));
    for (int i = 0; i < num_rates; i++) {
        time_delay_embedding(&embedded_library[i], &rate_vectors[i], EDM_TIME_DELAY, EDM_EMBEDDING_DIM);
    }
    
    // Perform EDM prediction for each event type
    const char* event_names[] = {"New order", "Modify order", "Delete order", "Trade execution"};
    
    for (int type = 0; type < EVENT_TYPES; type++) {
        printf("\nAnalyzing dynamic of %s events:\n", event_names[type]);
        
        // Select last time point for prediction
        vector_reg_t current_state;
        time_delay_embedding(&current_state, &rate_vectors[num_rates-1], EDM_TIME_DELAY, EDM_EMBEDDING_DIM);
        
        // Use EDM for prediction
        vector_reg_t prediction;
        edm_predict(&prediction, &current_state, embedded_library, num_rates, 
                   EDM_NUM_NEIGHBORS, EDM_EMBEDDING_DIM, EDM_PREDICTION_STEPS);
        
        // Extract predicted rate for this event type
        int32_t predicted_rate = get_vector_element(&prediction, type);
        double actual_rate = (double)predicted_rate / 1000.0; // Convert back to actual rate
        
        printf("%s event predicted rate: %.4f\n", event_names[type], actual_rate);
        
        // Analyze nonlinear characteristics of the sequence
        printf("Analyzing nonlinear characteristics of %s events:\n", event_names[type]);
        
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
        
        // Calculate nonlinear index of the sequence (simplified version)
        double nonlinearity = 0.0;
        for (int i = 1; i < num_rates; i++) {
            double diff = rates[i].rates[type] - rates[i-1].rates[type];
            nonlinearity += fabs(diff);
        }
        nonlinearity /= (num_rates - 1);
        
        printf("Nonlinear index: %.4f\n", nonlinearity);
        
        // Check if it fits Poisson process
        // For Poisson process, variance should be close to mean
        double poisson_ratio = variance / mean;
        printf("Variance/mean ratio: %.4f ", poisson_ratio);
        
        if (fabs(poisson_ratio - 1.0) < 0.2) {
            printf("(close to 1, fits Poisson process characteristics)\n");
        } else if (poisson_ratio > 1.0) {
            printf("(greater than 1, indicates overdispersion, possibly clustered arrival)\n");
        } else {
            printf("(less than 1, indicates underdispersion, possibly regular arrival)\n");
        }
    }
    
    // Clean up
    free(rate_vectors);
    free(embedded_library);
}

// Main function
int main() {
    printf("=== Orderbook event Poisson arrival rate analysis ===\n\n");
    
    // Initialize execution unit
    init_execution_unit();
    
    // Generate simulated orderbook events
    OrderEvent events[MAX_EVENTS];
    generate_simulated_events(events, MAX_EVENTS);
    
    // Calculate event rates
    EventRates rates[MAX_EVENTS / TIME_WINDOW + 1];
    int num_rates;
    calculate_event_rates(events, MAX_EVENTS, rates, &num_rates);
    
    // Print event rates
    printf("\n=== Event rates statistics ===\n");
    printf("%-15s %-15s %-15s %-15s %-15s %-15s\n", 
           "Time window", "New order rate", "Modify order rate", "Delete order rate", "Trade execution rate", "Total event rate");
    
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
    
    // Analyze autocorrelation of event rates
    analyze_autocorrelation(rates, num_rates);
    
    // Analyze correlation between event types
    analyze_cross_correlation(rates, num_rates);
    
    // Analyze event rates using EDM
    analyze_event_rates_with_edm(rates, num_rates);
    
    printf("\n=== Analysis completed ===\n");
    
    return 0;
} 