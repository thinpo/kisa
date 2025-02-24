#include "../../include/kisa.h"
#include <stdint.h>
#include <stdbool.h>

uint32_t encode_instruction(const instruction_t* inst) {
    uint32_t encoded = 0;
    
    // Encode opcode (8 bits)
    encoded |= (uint32_t)inst->opcode;
    
    // Encode registers (8 bits each)
    encoded |= ((uint32_t)inst->rd & 0xFF) << 8;
    encoded |= ((uint32_t)inst->rs1 & 0xFF) << 16;
    encoded |= ((uint32_t)inst->rs2 & 0xFF) << 24;
    
    return encoded;
}

instruction_t decode_instruction(uint32_t word) {
    instruction_t inst = {0};
    
    // Decode opcode
    inst.opcode = word & 0xFF;
    
    // Decode registers
    inst.rd = (word >> 8) & 0xFF;
    inst.rs1 = (word >> 16) & 0xFF;
    inst.rs2 = (word >> 24) & 0xFF;
    
    return inst;
} 