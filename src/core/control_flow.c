#include "../../include/kisa.h"
#include <stdint.h>
#include <stdbool.h>

// 条件码寄存器状态
typedef struct {
    bool zero;      // 零标志
    bool negative;  // 负标志
    bool overflow;  // 溢出标志
    bool carry;     // 进位标志
} condition_flags_t;

// 全局条件码寄存器
static condition_flags_t condition_flags = {0};

// 更新条件码
void update_flags(int32_t result, bool check_carry) {
    condition_flags.zero = (result == 0);
    condition_flags.negative = (result < 0);
    condition_flags.overflow = false;  // 需要根据具体操作更新
    if (check_carry) {
        condition_flags.carry = (result > INT32_MAX || result < INT32_MIN);
    }
}

// 条件分支指令
bool branch_if_equal(int32_t target_pc) {
    return condition_flags.zero;
}

bool branch_if_not_equal(int32_t target_pc) {
    return !condition_flags.zero;
}

bool branch_if_greater(int32_t target_pc) {
    return !condition_flags.zero && !condition_flags.negative;
}

bool branch_if_less(int32_t target_pc) {
    return condition_flags.negative;
}

// 向量掩码生成
void vector_mask_from_flags(vector_reg_t* mask) {
    #ifdef __aarch64__
    // NEON implementation
    int32x4_t flag_value = vdupq_n_s32(condition_flags.zero ? -1 : 0);
    mask->low = flag_value;
    mask->high = flag_value;
    #else
    // Scalar fallback
    int32_t flag_value = condition_flags.zero ? -1 : 0;
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        (*mask)[i] = flag_value;
    }
    #endif
}

// 向量条件执行
void vector_execute_if(vector_reg_t* result, vector_reg_t* mask, vector_reg_t* a, void (*op)(vector_reg_t*, vector_reg_t*)) {
    vector_reg_t temp;
    op(&temp, a);  // 执行操作
    vector_select(result, mask, &temp, result);  // 根据掩码选择结果
} 