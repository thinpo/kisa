/**
 * kisa.h - K-ISA (K-Inspired Instruction Set Architecture) main header file
 * 
 * This header defines the basic types and structures for K-ISA,
 * including vector registers and execution context.
 */

#ifndef KISA_H
#define KISA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// Vector length definitions
#define VECTOR_LENGTH 8
#define MAX_REGISTERS 256

// Vector register type definition
#ifdef __aarch64__
typedef struct {
    int32x4_t low;
    int32x4_t high;
} vector_reg_t;
#else
typedef int32_t vector_reg_t[VECTOR_LENGTH];
#endif

// Reduction operation types
typedef enum {
    RED_SUM,
    RED_PROD,
    RED_MAX,
    RED_MIN
} reduction_op_t;

// Opcode definitions
typedef enum {
    // Scalar operations
    OP_ADD = 0x01,
    OP_SUB = 0x02,
    OP_MUL = 0x03,
    OP_DIV = 0x04,
    
    // Vector operations
    OP_VADD = 0x11,
    OP_VSUB = 0x12,
    OP_VMUL = 0x13,
    OP_VDIV = 0x14,
    OP_VFFT = 0x15,
    OP_VIFFT = 0x16,
    OP_VSORT = 0x17,
    
    // Reduction operations
    OP_VREDSUM = 0x21,
    OP_VREDPROD = 0x22,
    OP_VREDMAX = 0x23,
    OP_VREDMIN = 0x24,
    
    // Scan operations
    OP_VSCANSUM = 0x31,
    OP_VSCANPROD = 0x32,
    OP_VSCANMAX = 0x33,
    OP_VSCANMIN = 0x34,
    
    // Control flow
    OP_BEQ = 0x41,
    OP_BNE = 0x42,
    OP_BGT = 0x43,
    OP_BLT = 0x44,
    OP_VCMP = 0x45,
    OP_VSEL = 0x46,
    
    // Memory operations
    OP_LOAD = 0x51,
    OP_STORE = 0x52,
    OP_VLOAD = 0x53,
    OP_VSTORE = 0x54
} opcode_t;

// Instruction format
typedef struct {
    uint8_t opcode;    // Operation code (8 bits)
    uint8_t rd;        // Destination register (8 bits)
    uint8_t rs1;       // Source register 1 (8 bits)
    uint8_t rs2;       // Source register 2 (8 bits)
} instruction_t;

// Function declarations
// Register access
void set_scalar_reg(uint8_t reg_num, int32_t value);
int32_t get_scalar_reg(uint8_t reg_num);
void set_vector_reg(uint8_t reg_num, const vector_reg_t* value);
void get_vector_reg(uint8_t reg_num, vector_reg_t* value);

// Control flow operations
void update_flags(int32_t result, bool check_carry);
bool branch_if_equal(int32_t target_pc);
bool branch_if_not_equal(int32_t target_pc);
bool branch_if_greater(int32_t target_pc);
bool branch_if_less(int32_t target_pc);

// Vector operations
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

// Instruction decoding and execution
instruction_t decode_instruction(uint32_t encoded_instruction);
bool validate_instruction(instruction_t* inst);
bool is_vector_operation(uint8_t opcode);
bool execute_instruction(instruction_t inst);
void init_execution_unit(void);

void print_register(const char* name, const vector_reg_t* reg);

#endif // KISA_H 