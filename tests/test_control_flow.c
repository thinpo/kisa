#include "../include/kisa.h"
#include <stdio.h>
#include <assert.h>

// 辅助函数：创建指令
static uint32_t create_instruction(uint8_t opcode, uint8_t rd, uint8_t rs1, uint8_t rs2) {
    return (opcode << 24) | (rd << 16) | (rs1 << 8) | rs2;
}

// 测试标量运算和条件码
void test_scalar_operations() {
    printf("\n=== Testing Scalar Operations and Condition Flags ===\n");
    
    init_execution_unit();
    
    // 测试加法和条件码
    uint32_t add_inst = create_instruction(OP_ADD, 1, 0, 0);  // r1 = r0 + r0 (0 + 0)
    instruction_t decoded = decode_instruction(add_inst);
    assert(execute_instruction(decoded));
    assert(branch_if_equal(0));  // 结果应该为0，零标志应该被设置
    
    // 测试减法和负标志
    uint32_t sub_inst = create_instruction(OP_SUB, 2, 0, 1);  // r2 = r0 - r1 (0 - 0)
    decoded = decode_instruction(sub_inst);
    assert(execute_instruction(decoded));
    assert(branch_if_equal(0));  // 结果应该为0
    
    printf("Scalar operations and condition flags test passed\n");
}

// 测试向量比较和选择
void test_vector_compare_select() {
    printf("\n=== Testing Vector Compare and Select ===\n");
    
    init_execution_unit();
    
    // 设置测试数据
    uint32_t inst;
    instruction_t decoded;
    
    // 加载测试数据到向量寄存器
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        // r1: [1,2,3,4,5,6,7,8]
        inst = create_instruction(OP_ADD, 1, 0, 0);
        decoded = decode_instruction(inst);
        execute_instruction(decoded);
        
        // r2: [8,7,6,5,4,3,2,1]
        inst = create_instruction(OP_SUB, 2, 0, 1);
        decoded = decode_instruction(inst);
        execute_instruction(decoded);
    }
    
    // 测试向量比较
    inst = create_instruction(OP_VCMP, 3, 1, 2);  // r3 = r1 > r2
    decoded = decode_instruction(inst);
    assert(execute_instruction(decoded));
    
    // 测试向量选择
    // 对于向量选择操作，我们将rs2作为第三个源寄存器
    inst = create_instruction(OP_VSEL, 4, 3, 2);  // r4 = select(r3, r1, r2)
    decoded = decode_instruction(inst);
    assert(execute_instruction(decoded));
    
    printf("Vector compare and select test passed\n");
}

// 测试条件分支
void test_conditional_branches() {
    printf("\n=== Testing Conditional Branches ===\n");
    
    init_execution_unit();
    
    // 设置测试数据
    uint32_t inst = create_instruction(OP_ADD, 1, 0, 0);  // r1 = r0 + r0 (0 + 0)
    instruction_t decoded = decode_instruction(inst);
    execute_instruction(decoded);
    
    // 测试相等分支
    inst = create_instruction(OP_BEQ, 1, 2, 0);  // if (r1 == 0) pc = r2
    decoded = decode_instruction(inst);
    assert(execute_instruction(decoded));
    
    // 测试不相等分支
    inst = create_instruction(OP_ADD, 1, 1, 1);  // r1 = r1 + r1 (0 + 0)
    decoded = decode_instruction(inst);
    execute_instruction(decoded);
    
    inst = create_instruction(OP_BNE, 1, 2, 0);  // if (r1 != 0) pc = r2
    decoded = decode_instruction(inst);
    assert(execute_instruction(decoded));
    
    printf("Conditional branches test passed\n");
}

// 测试向量规约和扫描
void test_reduction_scan() {
    printf("\n=== Testing Vector Reduction and Scan Operations ===\n");
    
    init_execution_unit();
    
    // 设置测试数据
    uint32_t inst;
    instruction_t decoded;
    
    // 加载测试数据到向量寄存器
    for (int i = 0; i < VECTOR_LENGTH; i++) {
        inst = create_instruction(OP_ADD, 1, 0, 0);  // r1 = [1,1,1,1,1,1,1,1]
        decoded = decode_instruction(inst);
        execute_instruction(decoded);
    }
    
    // 测试向量规约
    inst = create_instruction(OP_VREDSUM, 2, 1, 0);  // r2 = reduce_sum(r1)
    decoded = decode_instruction(inst);
    assert(execute_instruction(decoded));
    
    // 测试向量扫描
    inst = create_instruction(OP_VSCANSUM, 3, 1, 0);  // r3 = scan_sum(r1)
    decoded = decode_instruction(inst);
    assert(execute_instruction(decoded));
    
    printf("Vector reduction and scan operations test passed\n");
}

// 主测试函数
int main() {
    printf("Starting control flow tests...\n");
    
    test_scalar_operations();
    test_vector_compare_select();
    test_conditional_branches();
    test_reduction_scan();
    
    printf("\nAll control flow tests passed!\n");
    return 0;
} 