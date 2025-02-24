#include "../../include/kisa.h"
#include <string.h>
#include <math.h>

// Only compile AVX-512 code on x86_64 platforms with AVX-512 support
#if defined(__x86_64__) && defined(HAVE_AVX512)

#include <immintrin.h>

// Helper function for bit reversal
static uint32_t bit_reverse(uint32_t x, int bits) {
    uint32_t result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Complex number structure for AVX-512
typedef struct {
    __m512 real;
    __m512 imag;
} complex_avx512_t;

// Vector operations using AVX-512
void vector_fft(vector_reg_t* result, vector_reg_t* input) {
    // Copy input data for now (TODO: implement AVX-512 FFT)
    memcpy(result, input, sizeof(vector_reg_t));
}

void vector_ifft(vector_reg_t* result, vector_reg_t* input) {
    // Copy input data for now (TODO: implement AVX-512 IFFT)
    memcpy(result, input, sizeof(vector_reg_t));
}

void vector_sort(vector_reg_t* result, vector_reg_t* input) {
    // Simple bubble sort implementation for now (TODO: implement AVX-512 sort)
    memcpy(result, input, sizeof(vector_reg_t));
    #ifdef __aarch64__
    int32x4_t low = result->low;
    int32x4_t high = result->high;
    for (int i = 0; i < VECTOR_LENGTH - 1; i++) {
        for (int j = 0; j < VECTOR_LENGTH - i - 1; j++) {
            int32_t val_j = j < 4 ? vgetq_lane_s32(low, j) : vgetq_lane_s32(high, j - 4);
            int32_t val_next = (j + 1) < 4 ? vgetq_lane_s32(low, j + 1) : vgetq_lane_s32(high, j - 3);
            if (val_j > val_next) {
                // Swap values
                if (j < 4) {
                    if (j + 1 < 4) {
                        low = vsetq_lane_s32(val_next, low, j);
                        low = vsetq_lane_s32(val_j, low, j + 1);
                    } else {
                        low = vsetq_lane_s32(val_next, low, j);
                        high = vsetq_lane_s32(val_j, high, 0);
                    }
                } else {
                    high = vsetq_lane_s32(val_next, high, j - 4);
                    high = vsetq_lane_s32(val_j, high, j - 3);
                }
            }
        }
    }
    result->low = low;
    result->high = high;
    #else
    for (int i = 0; i < VECTOR_LENGTH - 1; i++) {
        for (int j = 0; j < VECTOR_LENGTH - i - 1; j++) {
            if ((*result)[j] > (*result)[j + 1]) {
                int32_t temp = (*result)[j];
                (*result)[j] = (*result)[j + 1];
                (*result)[j + 1] = temp;
            }
        }
    }
    #endif
}

int32_t vector_reduce(vector_reg_t* input, reduction_op_t op) {
    #ifdef __aarch64__
    int32x4_t low = input->low;
    int32x4_t high = input->high;
    int32_t result = vgetq_lane_s32(low, 0);
    
    switch (op) {
        case RED_SUM: {
            int32x4_t sum = vaddq_s32(low, high);
            int32x2_t sum2 = vadd_s32(vget_low_s32(sum), vget_high_s32(sum));
            result = vget_lane_s32(vpadd_s32(sum2, sum2), 0);
            break;
        }
        case RED_PROD: {
            for (int i = 1; i < 4; i++) {
                result *= vgetq_lane_s32(low, i);
            }
            for (int i = 0; i < 4; i++) {
                result *= vgetq_lane_s32(high, i);
            }
            break;
        }
        case RED_MAX: {
            int32x4_t max = vmaxq_s32(low, high);
            int32x2_t max2 = vmax_s32(vget_low_s32(max), vget_high_s32(max));
            result = vget_lane_s32(vpmax_s32(max2, max2), 0);
            break;
        }
        case RED_MIN: {
            int32x4_t min = vminq_s32(low, high);
            int32x2_t min2 = vmin_s32(vget_low_s32(min), vget_high_s32(min));
            result = vget_lane_s32(vpmin_s32(min2, min2), 0);
            break;
        }
    }
    return result;
    #else
    int32_t result = (*input)[0];
    for (int i = 1; i < VECTOR_LENGTH; i++) {
        switch (op) {
            case RED_SUM:
                result += (*input)[i];
                break;
            case RED_PROD:
                result *= (*input)[i];
                break;
            case RED_MAX:
                if ((*input)[i] > result) result = (*input)[i];
                break;
            case RED_MIN:
                if ((*input)[i] < result) result = (*input)[i];
                break;
        }
    }
    return result;
    #endif
}

void vector_scan(vector_reg_t* result, vector_reg_t* input, reduction_op_t op) {
    #ifdef __aarch64__
    int32x4_t low = input->low;
    int32x4_t high = input->high;
    int32x4_t scan_low = low;
    int32x4_t scan_high = high;
    
    switch (op) {
        case RED_SUM: {
            for (int i = 1; i < 4; i++) {
                int32_t prev = vgetq_lane_s32(scan_low, i - 1);
                int32_t curr = vgetq_lane_s32(low, i);
                scan_low = vsetq_lane_s32(prev + curr, scan_low, i);
            }
            int32_t last_low = vgetq_lane_s32(scan_low, 3);
            for (int i = 0; i < 4; i++) {
                int32_t curr = vgetq_lane_s32(high, i);
                scan_high = vsetq_lane_s32(last_low + curr, scan_high, i);
                last_low += curr;
            }
            break;
        }
        case RED_PROD: {
            for (int i = 1; i < 4; i++) {
                int32_t prev = vgetq_lane_s32(scan_low, i - 1);
                int32_t curr = vgetq_lane_s32(low, i);
                scan_low = vsetq_lane_s32(prev * curr, scan_low, i);
            }
            int32_t last_low = vgetq_lane_s32(scan_low, 3);
            for (int i = 0; i < 4; i++) {
                int32_t curr = vgetq_lane_s32(high, i);
                scan_high = vsetq_lane_s32(last_low * curr, scan_high, i);
                last_low *= curr;
            }
            break;
        }
        case RED_MAX: {
            for (int i = 1; i < 4; i++) {
                int32_t prev = vgetq_lane_s32(scan_low, i - 1);
                int32_t curr = vgetq_lane_s32(low, i);
                scan_low = vsetq_lane_s32(prev > curr ? prev : curr, scan_low, i);
            }
            int32_t last_low = vgetq_lane_s32(scan_low, 3);
            for (int i = 0; i < 4; i++) {
                int32_t curr = vgetq_lane_s32(high, i);
                scan_high = vsetq_lane_s32(last_low > curr ? last_low : curr, scan_high, i);
                last_low = last_low > curr ? last_low : curr;
            }
            break;
        }
        case RED_MIN: {
            for (int i = 1; i < 4; i++) {
                int32_t prev = vgetq_lane_s32(scan_low, i - 1);
                int32_t curr = vgetq_lane_s32(low, i);
                scan_low = vsetq_lane_s32(prev < curr ? prev : curr, scan_low, i);
            }
            int32_t last_low = vgetq_lane_s32(scan_low, 3);
            for (int i = 0; i < 4; i++) {
                int32_t curr = vgetq_lane_s32(high, i);
                scan_high = vsetq_lane_s32(last_low < curr ? last_low : curr, scan_high, i);
                last_low = last_low < curr ? last_low : curr;
            }
            break;
        }
    }
    result->low = scan_low;
    result->high = scan_high;
    #else
    (*result)[0] = (*input)[0];
    for (int i = 1; i < VECTOR_LENGTH; i++) {
        switch (op) {
            case RED_SUM:
                (*result)[i] = (*result)[i-1] + (*input)[i];
                break;
            case RED_PROD:
                (*result)[i] = (*result)[i-1] * (*input)[i];
                break;
            case RED_MAX:
                (*result)[i] = (*result)[i-1] > (*input)[i] ? (*result)[i-1] : (*input)[i];
                break;
            case RED_MIN:
                (*result)[i] = (*result)[i-1] < (*input)[i] ? (*result)[i-1] : (*input)[i];
                break;
        }
    }
    #endif
}

#else // !(__x86_64__ && HAVE_AVX512)

// Helper functions for NEON lane access
#ifdef __aarch64__
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

void vector_fft(vector_reg_t* result, vector_reg_t* input) {
    memcpy(result, input, sizeof(vector_reg_t));
}

void vector_ifft(vector_reg_t* result, vector_reg_t* input) {
    memcpy(result, input, sizeof(vector_reg_t));
}

void vector_sort(vector_reg_t* result, vector_reg_t* input) {
    memcpy(result, input, sizeof(vector_reg_t));
    #ifdef __aarch64__
    int32x4_t low = result->low;
    int32x4_t high = result->high;
    for (int i = 0; i < VECTOR_LENGTH - 1; i++) {
        for (int j = 0; j < VECTOR_LENGTH - i - 1; j++) {
            int32_t val_j = j < 4 ? get_lane_s32(low, j) : get_lane_s32(high, j - 4);
            int32_t val_next = (j + 1) < 4 ? get_lane_s32(low, j + 1) : get_lane_s32(high, j - 3);
            if (val_j > val_next) {
                // Swap values
                if (j < 4) {
                    if (j + 1 < 4) {
                        low = set_lane_s32(val_next, low, j);
                        low = set_lane_s32(val_j, low, j + 1);
                    } else {
                        low = set_lane_s32(val_next, low, j);
                        high = set_lane_s32(val_j, high, 0);
                    }
                } else {
                    high = set_lane_s32(val_next, high, j - 4);
                    high = set_lane_s32(val_j, high, j - 3);
                }
            }
        }
    }
    result->low = low;
    result->high = high;
    #else
    for (int i = 0; i < VECTOR_LENGTH - 1; i++) {
        for (int j = 0; j < VECTOR_LENGTH - i - 1; j++) {
            if ((*result)[j] > (*result)[j + 1]) {
                int32_t temp = (*result)[j];
                (*result)[j] = (*result)[j + 1];
                (*result)[j + 1] = temp;
            }
        }
    }
    #endif
}

int32_t vector_reduce(vector_reg_t* input, reduction_op_t op) {
    #ifdef __aarch64__
    int32x4_t low = input->low;
    int32x4_t high = input->high;
    int32_t result = get_lane_s32(low, 0);
    
    switch (op) {
        case RED_SUM: {
            int32x4_t sum = vaddq_s32(low, high);
            int32x2_t sum2 = vadd_s32(vget_low_s32(sum), vget_high_s32(sum));
            result = vget_lane_s32(vpadd_s32(sum2, sum2), 0);
            break;
        }
        case RED_PROD: {
            for (int i = 1; i < 4; i++) {
                result *= get_lane_s32(low, i);
            }
            for (int i = 0; i < 4; i++) {
                result *= get_lane_s32(high, i);
            }
            break;
        }
        case RED_MAX: {
            int32x4_t max = vmaxq_s32(low, high);
            int32x2_t max2 = vmax_s32(vget_low_s32(max), vget_high_s32(max));
            result = vget_lane_s32(vpmax_s32(max2, max2), 0);
            break;
        }
        case RED_MIN: {
            int32x4_t min = vminq_s32(low, high);
            int32x2_t min2 = vmin_s32(vget_low_s32(min), vget_high_s32(min));
            result = vget_lane_s32(vpmin_s32(min2, min2), 0);
            break;
        }
    }
    return result;
    #else
    int32_t result = (*input)[0];
    for (int i = 1; i < VECTOR_LENGTH; i++) {
        switch (op) {
            case RED_SUM:
                result += (*input)[i];
                break;
            case RED_PROD:
                result *= (*input)[i];
                break;
            case RED_MAX:
                if ((*input)[i] > result) result = (*input)[i];
                break;
            case RED_MIN:
                if ((*input)[i] < result) result = (*input)[i];
                break;
        }
    }
    return result;
    #endif
}

void vector_scan(vector_reg_t* result, vector_reg_t* input, reduction_op_t op) {
    #ifdef __aarch64__
    int32x4_t low = input->low;
    int32x4_t high = input->high;
    int32x4_t scan_low = low;
    int32x4_t scan_high = high;
    
    switch (op) {
        case RED_SUM: {
            for (int i = 1; i < 4; i++) {
                int32_t prev = get_lane_s32(scan_low, i - 1);
                int32_t curr = get_lane_s32(low, i);
                scan_low = set_lane_s32(prev + curr, scan_low, i);
            }
            int32_t last_low = get_lane_s32(scan_low, 3);
            for (int i = 0; i < 4; i++) {
                int32_t curr = get_lane_s32(high, i);
                scan_high = set_lane_s32(last_low + curr, scan_high, i);
                last_low += curr;
            }
            break;
        }
        case RED_PROD: {
            for (int i = 1; i < 4; i++) {
                int32_t prev = get_lane_s32(scan_low, i - 1);
                int32_t curr = get_lane_s32(low, i);
                scan_low = set_lane_s32(prev * curr, scan_low, i);
            }
            int32_t last_low = get_lane_s32(scan_low, 3);
            for (int i = 0; i < 4; i++) {
                int32_t curr = get_lane_s32(high, i);
                scan_high = set_lane_s32(last_low * curr, scan_high, i);
                last_low *= curr;
            }
            break;
        }
        case RED_MAX: {
            for (int i = 1; i < 4; i++) {
                int32_t prev = get_lane_s32(scan_low, i - 1);
                int32_t curr = get_lane_s32(low, i);
                scan_low = set_lane_s32(prev > curr ? prev : curr, scan_low, i);
            }
            int32_t last_low = get_lane_s32(scan_low, 3);
            for (int i = 0; i < 4; i++) {
                int32_t curr = get_lane_s32(high, i);
                scan_high = set_lane_s32(last_low > curr ? last_low : curr, scan_high, i);
                last_low = last_low > curr ? last_low : curr;
            }
            break;
        }
        case RED_MIN: {
            for (int i = 1; i < 4; i++) {
                int32_t prev = get_lane_s32(scan_low, i - 1);
                int32_t curr = get_lane_s32(low, i);
                scan_low = set_lane_s32(prev < curr ? prev : curr, scan_low, i);
            }
            int32_t last_low = get_lane_s32(scan_low, 3);
            for (int i = 0; i < 4; i++) {
                int32_t curr = get_lane_s32(high, i);
                scan_high = set_lane_s32(last_low < curr ? last_low : curr, scan_high, i);
                last_low = last_low < curr ? last_low : curr;
            }
            break;
        }
    }
    result->low = scan_low;
    result->high = scan_high;
    #else
    (*result)[0] = (*input)[0];
    for (int i = 1; i < VECTOR_LENGTH; i++) {
        switch (op) {
            case RED_SUM:
                (*result)[i] = (*result)[i-1] + (*input)[i];
                break;
            case RED_PROD:
                (*result)[i] = (*result)[i-1] * (*input)[i];
                break;
            case RED_MAX:
                (*result)[i] = (*result)[i-1] > (*input)[i] ? (*result)[i-1] : (*input)[i];
                break;
            case RED_MIN:
                (*result)[i] = (*result)[i-1] < (*input)[i] ? (*result)[i-1] : (*input)[i];
                break;
        }
    }
    #endif
}

#endif // __x86_64__ && HAVE_AVX512 