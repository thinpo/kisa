#ifndef KISA_H
#define KISA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// 向量长度定义
#define VECTOR_LENGTH 8
#define MAX_REGISTERS 256

// 向量寄存器类型定义
#ifdef __aarch64__
typedef struct {
    int32x4_t low;
    int32x4_t high;
} vector_reg_t;
#else
typedef int32_t vector_reg_t[VECTOR_LENGTH];
#endif

// 规约操作类型
typedef enum {
    RED_SUM,
    RED_PROD,
    RED_MAX,
    RED_MIN
} reduction_op_t;

// 操作码定义
typedef enum {
    // 标量运算
    OP_ADD = 0x01,
    OP_SUB = 0x02,
    OP_MUL = 0x03,
    OP_DIV = 0x04,
    
    // 向量运算
    OP_VADD = 0x11,
    OP_VSUB = 0x12,
    OP_VMUL = 0x13,
    OP_VDIV = 0x14,
    OP_VFFT = 0x15,
    OP_VIFFT = 0x16,
    OP_VSORT = 0x17,
    
    // 规约操作
    OP_VREDSUM = 0x21,
    OP_VREDPROD = 0x22,
    OP_VREDMAX = 0x23,
    OP_VREDMIN = 0x24,
    
    // 扫描操作
    OP_VSCANSUM = 0x31,
    OP_VSCANPROD = 0x32,
    OP_VSCANMAX = 0x33,
    OP_VSCANMIN = 0x34,
    
    // 控制流
    OP_BEQ = 0x41,
    OP_BNE = 0x42,
    OP_BGT = 0x43,
    OP_BLT = 0x44,
    OP_VCMP = 0x45,
    OP_VSEL = 0x46,
    
    // 内存操作
    OP_LOAD = 0x51,
    OP_STORE = 0x52,
    OP_VLOAD = 0x53,
    OP_VSTORE = 0x54
} opcode_t;

// 指令格式
typedef struct {
    uint8_t opcode;    // 操作码 (8位)
    uint8_t rd;        // 目标寄存器 (8位)
    uint8_t rs1;       // 源寄存器1 (8位)
    uint8_t rs2;       // 源寄存器2 (8位)
} instruction_t;

// 函数声明
// 寄存器访问
void set_scalar_reg(uint8_t reg_num, int32_t value);
int32_t get_scalar_reg(uint8_t reg_num);
void set_vector_reg(uint8_t reg_num, const vector_reg_t* value);
void get_vector_reg(uint8_t reg_num, vector_reg_t* value);

// 控制流操作
void update_flags(int32_t result, bool check_carry);
bool branch_if_equal(int32_t target_pc);
bool branch_if_not_equal(int32_t target_pc);
bool branch_if_greater(int32_t target_pc);
bool branch_if_less(int32_t target_pc);

// 向量操作
void vector_add(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b);
void vector_sub(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b);
void vector_mul(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b);
void vector_div(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b);
void vector_fft(vector_reg_t* result, vector_reg_t* input);
void vector_ifft(vector_reg_t* result, vector_reg_t* input);
void vector_sort(vector_reg_t* result, vector_reg_t* input);
int32_t vector_reduce(vector_reg_t* input, reduction_op_t op);
void vector_scan(vector_reg_t* result, vector_reg_t* input, reduction_op_t op);
void vector_compare(vector_reg_t* result, vector_reg_t* a, vector_reg_t* b);
void vector_select(vector_reg_t* result, vector_reg_t* mask, vector_reg_t* a, vector_reg_t* b);

// 指令解码和执行
instruction_t decode_instruction(uint32_t encoded_instruction);
bool validate_instruction(instruction_t* inst);
bool is_vector_operation(uint8_t opcode);
bool execute_instruction(instruction_t inst);
void init_execution_unit(void);

#endif // KISA_H 