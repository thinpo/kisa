#include "../../include/kisa.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#define MAX_REGISTERS 256
#define MEMORY_SIZE 1024

// 寄存器文件
static vector_reg_t vector_registers[MAX_REGISTERS];
static int32_t scalar_registers[MAX_REGISTERS];

// 内存
static int32_t memory[MEMORY_SIZE];

// 程序计数器
static uint32_t program_counter = 0;

// 寄存器访问函数
void set_scalar_reg(uint8_t reg_num, int32_t value) {
    scalar_registers[reg_num] = value;
}

int32_t get_scalar_reg(uint8_t reg_num) {
    return scalar_registers[reg_num];
}

void set_vector_reg(uint8_t reg_num, const vector_reg_t* value) {
    if (value != NULL) {
#ifdef __aarch64__
        vector_registers[reg_num].low = value->low;
        vector_registers[reg_num].high = value->high;
#else
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            vector_registers[reg_num][i] = (*value)[i];
        }
#endif
    }
}

void get_vector_reg(uint8_t reg_num, vector_reg_t* value) {
    if (value != NULL) {
#ifdef __aarch64__
        value->low = vector_registers[reg_num].low;
        value->high = vector_registers[reg_num].high;
#else
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            (*value)[i] = vector_registers[reg_num][i];
        }
#endif
    }
}

// 初始化执行单元
void init_execution_unit() {
    // 清零所有寄存器和内存
    memset(scalar_registers, 0, sizeof(scalar_registers));
    memset(memory, 0, sizeof(memory));
    
    for (int i = 0; i < MAX_REGISTERS; i++) {
#ifdef __aarch64__
        vector_registers[i].low = vdupq_n_s32(0);
        vector_registers[i].high = vdupq_n_s32(0);
#else
        for (int j = 0; j < VECTOR_LENGTH; j++) {
            vector_registers[i][j] = 0;
        }
#endif
    }
    program_counter = 0;
}

// 执行内存操作
void execute_memory_op(uint8_t opcode, uint8_t rd, uint8_t rs1, uint8_t rs2) {
    uint32_t addr = scalar_registers[rs1];
    printf("Memory operation: opcode=0x%02x, rd=%u, rs1=%u, rs2=%u, addr=%u\n", 
           opcode, rd, rs1, rs2, addr);
    
    if (addr >= MEMORY_SIZE) {
        printf("Memory access out of bounds: addr=%u\n", addr);
        return;  // 内存访问越界
    }
    
    switch(opcode) {
        case OP_LOAD:
            scalar_registers[rd] = memory[addr];
            printf("LOAD: memory[%u] = %d -> reg[%u]\n", 
                   addr, memory[addr], rd);
            break;
            
        case OP_STORE:
            memory[addr] = scalar_registers[rs2];
            printf("STORE: reg[%u] = %d -> memory[%u]\n", 
                   rs2, scalar_registers[rs2], addr);
            break;
            
        case OP_VLOAD:
            if (addr + VECTOR_LENGTH > MEMORY_SIZE) {
                printf("Vector load out of bounds: addr=%u\n", addr);
                return;  // 向量访问越界
            }
#ifdef __aarch64__
            vector_registers[rd].low = vld1q_s32(&memory[addr]);
            vector_registers[rd].high = vld1q_s32(&memory[addr + 4]);
#else
            memcpy(&vector_registers[rd], &memory[addr], sizeof(vector_reg_t));
#endif
            printf("VLOAD: memory[%u:%u] -> vreg[%u]\n", 
                   addr, addr + VECTOR_LENGTH - 1, rd);
            break;
            
        case OP_VSTORE:
            if (addr + VECTOR_LENGTH > MEMORY_SIZE) {
                printf("Vector store out of bounds: addr=%u\n", addr);
                return;  // 向量访问越界
            }
#ifdef __aarch64__
            vst1q_s32(&memory[addr], vector_registers[rs2].low);
            vst1q_s32(&memory[addr + 4], vector_registers[rs2].high);
#else
            memcpy(&memory[addr], &vector_registers[rs2], sizeof(vector_reg_t));
#endif
            printf("VSTORE: vreg[%u] -> memory[%u:%u]\n", 
                   rs2, addr, addr + VECTOR_LENGTH - 1);
            break;
    }
}

// 执行标量运算
void execute_scalar_op(uint8_t opcode, uint8_t rd, uint8_t rs1, uint8_t rs2) {
    int32_t result = 0;
    switch(opcode) {
        case OP_ADD:
            result = scalar_registers[rs1] + scalar_registers[rs2];
            break;
        case OP_SUB:
            result = scalar_registers[rs1] - scalar_registers[rs2];
            break;
        case OP_MUL:
            result = scalar_registers[rs1] * scalar_registers[rs2];
            break;
        case OP_DIV:
            if (scalar_registers[rs2] != 0) {
                result = scalar_registers[rs1] / scalar_registers[rs2];
            }
            break;
    }
    scalar_registers[rd] = result;
    update_flags(result, true);
}

// 执行向量运算
void execute_vector_op(uint8_t opcode, uint8_t rd, uint8_t rs1, uint8_t rs2) {
    switch(opcode) {
        case OP_VADD:
            vector_add(&vector_registers[rd], &vector_registers[rs1], &vector_registers[rs2]);
            break;
        case OP_VSUB:
            vector_sub(&vector_registers[rd], &vector_registers[rs1], &vector_registers[rs2]);
            break;
        case OP_VMUL:
            vector_mul(&vector_registers[rd], &vector_registers[rs1], &vector_registers[rs2]);
            break;
        case OP_VDIV:
            vector_div(&vector_registers[rd], &vector_registers[rs1], &vector_registers[rs2]);
            break;
        case OP_VFFT:
            vector_fft(&vector_registers[rd], &vector_registers[rs1]);
            break;
        case OP_VIFFT:
            vector_ifft(&vector_registers[rd], &vector_registers[rs1]);
            break;
        case OP_VSORT:
            vector_sort(&vector_registers[rd], &vector_registers[rs1]);
            break;
    }
}

// 执行规约操作
void execute_reduction_op(uint8_t opcode, uint8_t rd, uint8_t rs1) {
    reduction_op_t red_op;
    switch(opcode) {
        case OP_VREDSUM:
            red_op = RED_SUM;
            break;
        case OP_VREDPROD:
            red_op = RED_PROD;
            break;
        case OP_VREDMAX:
            red_op = RED_MAX;
            break;
        case OP_VREDMIN:
            red_op = RED_MIN;
            break;
        default:
            return;
    }
    scalar_registers[rd] = vector_reduce(&vector_registers[rs1], red_op);
}

// 执行扫描操作
void execute_scan_op(uint8_t opcode, uint8_t rd, uint8_t rs1) {
    reduction_op_t scan_op;
    switch(opcode) {
        case OP_VSCANSUM:
            scan_op = RED_SUM;
            break;
        case OP_VSCANPROD:
            scan_op = RED_PROD;
            break;
        case OP_VSCANMAX:
            scan_op = RED_MAX;
            break;
        case OP_VSCANMIN:
            scan_op = RED_MIN;
            break;
        default:
            return;
    }
    vector_scan(&vector_registers[rd], &vector_registers[rs1], scan_op);
}

// 执行控制流操作
void execute_control_flow(uint8_t opcode, uint8_t rd, uint8_t rs1, uint8_t rs2) {
    switch(opcode) {
        case OP_BEQ:
            if (branch_if_equal(scalar_registers[rs1])) {
                program_counter = scalar_registers[rs2];
            }
            break;
        case OP_BNE:
            if (branch_if_not_equal(scalar_registers[rs1])) {
                program_counter = scalar_registers[rs2];
            }
            break;
        case OP_BGT:
            if (branch_if_greater(scalar_registers[rs1])) {
                program_counter = scalar_registers[rs2];
            }
            break;
        case OP_BLT:
            if (branch_if_less(scalar_registers[rs1])) {
                program_counter = scalar_registers[rs2];
            }
            break;
        case OP_VCMP:
            vector_compare(&vector_registers[rd], &vector_registers[rs1], &vector_registers[rs2]);
            break;
        case OP_VSEL:
            vector_select(&vector_registers[rd], &vector_registers[rs1], &vector_registers[rs2], &vector_registers[rd]);
            break;
    }
}

// 执行指令
bool execute_instruction(instruction_t inst) {
    printf("[DEBUG] Executing instruction: opcode=0x%02x, rd=%u, rs1=%u, rs2=%u\n",
           inst.opcode, inst.rd, inst.rs1, inst.rs2);
    fflush(stdout);
    
    // 首先检查是否是内存操作
    if (inst.opcode >= 0x51 && inst.opcode <= 0x54) {
        printf("[DEBUG] Memory operation detected\n");
        fflush(stdout);
        
        // 打印内存操作前的状态
        if (inst.opcode == OP_STORE) {
            printf("[DEBUG] Before STORE: reg[%u]=%d, addr=%u\n",
                   inst.rs2, scalar_registers[inst.rs2], scalar_registers[inst.rs1]);
        } else if (inst.opcode == OP_LOAD) {
            printf("[DEBUG] Before LOAD: memory[%u]=%d\n",
                   scalar_registers[inst.rs1], memory[scalar_registers[inst.rs1]]);
        }
        fflush(stdout);
        
        execute_memory_op(inst.opcode, inst.rd, inst.rs1, inst.rs2);
        
        // 打印内存操作后的状态
        if (inst.opcode == OP_STORE) {
            printf("[DEBUG] After STORE: memory[%u]=%d\n",
                   scalar_registers[inst.rs1], memory[scalar_registers[inst.rs1]]);
        } else if (inst.opcode == OP_LOAD) {
            printf("[DEBUG] After LOAD: reg[%u]=%d\n",
                   inst.rd, scalar_registers[inst.rd]);
        }
        fflush(stdout);
        
        return true;
    }
    
    // 然后处理其他操作
    switch(inst.opcode & 0xF0) {
        case 0x00:  // 标量运算
            printf("[DEBUG] Scalar operation\n");
            execute_scalar_op(inst.opcode, inst.rd, inst.rs1, inst.rs2);
            break;
        case 0x10:  // 向量运算
            printf("[DEBUG] Vector operation\n");
            execute_vector_op(inst.opcode, inst.rd, inst.rs1, inst.rs2);
            break;
        case 0x20:  // 规约操作
            printf("[DEBUG] Reduction operation\n");
            execute_reduction_op(inst.opcode, inst.rd, inst.rs1);
            break;
        case 0x30:  // 扫描操作
            printf("[DEBUG] Scan operation\n");
            execute_scan_op(inst.opcode, inst.rd, inst.rs1);
            break;
        case 0x40:  // 控制流
            printf("[DEBUG] Control flow operation\n");
            execute_control_flow(inst.opcode, inst.rd, inst.rs1, inst.rs2);
            break;
        default:
            printf("[DEBUG] Unknown operation type: 0x%02x\n", inst.opcode & 0xF0);
            return false;
    }
    fflush(stdout);
    return true;
} 