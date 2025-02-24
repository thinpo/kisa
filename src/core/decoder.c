#include "../../include/kisa.h"
#include <stdint.h>

// 解码32位指令
instruction_t decode_instruction(uint32_t encoded_instruction) {
    instruction_t decoded;
    
    // 提取各个字段
    decoded.opcode = (encoded_instruction >> 24) & 0xFF;
    decoded.rd = (encoded_instruction >> 16) & 0xFF;
    decoded.rs1 = (encoded_instruction >> 8) & 0xFF;
    decoded.rs2 = encoded_instruction & 0xFF;
    
    return decoded;
}

// 获取操作类型
bool is_vector_operation(uint8_t opcode) {
    return (opcode >= 0x11 && opcode <= 0x17) ||  // 向量运算
           (opcode >= 0x21 && opcode <= 0x24) ||  // 规约操作
           (opcode >= 0x31 && opcode <= 0x34) ||  // 扫描操作
           (opcode >= 0x45 && opcode <= 0x46);    // 向量控制流
}

// 获取操作数数量
int get_operand_count(uint8_t opcode) {
    switch(opcode) {
        case OP_VFFT:
        case OP_VIFFT:
        case OP_VSORT:
        case OP_VREDSUM:
        case OP_VREDPROD:
        case OP_VREDMAX:
        case OP_VREDMIN:
        case OP_VSCANSUM:
        case OP_VSCANPROD:
        case OP_VSCANMAX:
        case OP_VSCANMIN:
            return 1;  // 单操作数
            
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_VADD:
        case OP_VSUB:
        case OP_VMUL:
        case OP_VDIV:
        case OP_VCMP:
            return 2;  // 双操作数
            
        case OP_VSEL:
            return 3;  // 三操作数
            
        default:
            return 0;  // 未知操作码
    }
}

// 验证指令格式
bool validate_instruction(instruction_t* inst) {
    // 检查操作数数量是否正确
    int operand_count = get_operand_count(inst->opcode);
    if (operand_count == 0) {
        return false;  // 未知操作码
    }
    
    // 对于单操作数指令，rs2应为0
    if (operand_count == 1 && inst->rs2 != 0) {
        return false;
    }
    
    return true;
} 