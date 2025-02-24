import numpy as np
from typing import Optional
from .registers import RegisterFile
from .instructions import Instruction, InstructionType, OpCode

class ExecutionUnit:
    """Execution unit for K-ISA processor"""
    
    def __init__(self):
        self.registers = RegisterFile()
        self.memory = np.zeros(1024 * 1024, dtype=np.int32)  # 4MB memory
        
    def execute(self, instruction: Instruction) -> None:
        """Execute a single instruction"""
        if instruction.type == InstructionType.SCALAR_ARITHMETIC:
            self._execute_scalar_arithmetic(instruction)
        elif instruction.type == InstructionType.VECTOR_ARITHMETIC:
            self._execute_vector_arithmetic(instruction)
        elif instruction.type == InstructionType.MEMORY:
            self._execute_memory(instruction)
        elif instruction.type == InstructionType.CONTROL_FLOW:
            self._execute_control_flow(instruction)
        elif instruction.type == InstructionType.ARRAY_OPS:
            self._execute_array_ops(instruction)
        elif instruction.type == InstructionType.DATA_MOVEMENT:
            self._execute_data_movement(instruction)
            
    def _execute_scalar_arithmetic(self, inst: Instruction) -> None:
        """Execute scalar arithmetic instruction"""
        if inst.rd is None or inst.rs1 is None or inst.rs2 is None:
            raise ValueError("Invalid scalar arithmetic instruction")
            
        a = self.registers.get_scalar(inst.rs1)
        b = self.registers.get_scalar(inst.rs2)
        result = np.int32(0)
        
        if inst.opcode == OpCode.ADD:
            result = a + b
        elif inst.opcode == OpCode.SUB:
            result = a - b
        elif inst.opcode == OpCode.MUL:
            result = a * b
        elif inst.opcode == OpCode.DIV:
            if b == 0:
                raise ValueError("Division by zero")
            result = a // b
            
        self.registers.set_scalar(inst.rd, result)
        self._update_status_flags(result)
        
    def _execute_vector_arithmetic(self, inst: Instruction) -> None:
        """Execute vector arithmetic instruction"""
        if inst.vd is None or inst.vs1 is None or inst.vs2 is None:
            raise ValueError("Invalid vector arithmetic instruction")
            
        a = self.registers.get_vector(inst.vs1)
        b = self.registers.get_vector(inst.vs2)
        result = np.zeros(8, dtype=np.int32)
        
        if inst.opcode == OpCode.VADD:
            result = a + b
        elif inst.opcode == OpCode.VSUB:
            result = a - b
        elif inst.opcode == OpCode.VMUL:
            result = a * b
        elif inst.opcode == OpCode.VDIV:
            if np.any(b == 0):
                raise ValueError("Division by zero in vector")
            result = a // b
            
        self.registers.set_vector(inst.vd, result)
        
    def _execute_memory(self, inst: Instruction) -> None:
        """Execute memory access instruction"""
        if inst.opcode == OpCode.LD:
            if inst.rd is None or inst.rs1 is None:
                raise ValueError("Invalid load instruction")
            addr = self.registers.get_scalar(inst.rs1)
            if inst.offset is not None:
                addr += inst.offset
            value = self.memory[addr // 4]  # 4-byte aligned
            self.registers.set_scalar(inst.rd, value)
            
        elif inst.opcode == OpCode.ST:
            if inst.rs1 is None or inst.rd is None:
                raise ValueError("Invalid store instruction")
            addr = self.registers.get_scalar(inst.rd)
            if inst.offset is not None:
                addr += inst.offset
            value = self.registers.get_scalar(inst.rs1)
            self.memory[addr // 4] = value
            
        elif inst.opcode == OpCode.VLD:
            if inst.vd is None or inst.rs1 is None:
                raise ValueError("Invalid vector load instruction")
            addr = self.registers.get_scalar(inst.rs1)
            if inst.offset is not None:
                addr += inst.offset
            values = self.memory[addr // 4:addr // 4 + 8]
            self.registers.set_vector(inst.vd, values)
            
        elif inst.opcode == OpCode.VST:
            if inst.vs1 is None or inst.rd is None:
                raise ValueError("Invalid vector store instruction")
            addr = self.registers.get_scalar(inst.rd)
            if inst.offset is not None:
                addr += inst.offset
            values = self.registers.get_vector(inst.vs1)
            self.memory[addr // 4:addr // 4 + 8] = values
            
    def _execute_array_ops(self, inst: Instruction) -> None:
        """Execute array operation instruction"""
        if inst.rd is None or inst.vs1 is None:
            raise ValueError("Invalid array operation instruction")
            
        vec = self.registers.get_vector(inst.vs1)
        result = np.int32(0)
        
        if inst.opcode == OpCode.SUM:
            result = np.sum(vec)
        elif inst.opcode == OpCode.PROD:
            result = np.prod(vec)
        elif inst.opcode == OpCode.MAX:
            result = np.max(vec)
        elif inst.opcode == OpCode.MIN:
            result = np.min(vec)
            
        self.registers.set_scalar(inst.rd, result)
        self._update_status_flags(result)
        
    def _execute_data_movement(self, inst: Instruction) -> None:
        """Execute data movement instruction"""
        if inst.opcode == OpCode.MOV:
            if inst.rd is None or inst.rs1 is None:
                raise ValueError("Invalid move instruction")
            value = self.registers.get_scalar(inst.rs1)
            self.registers.set_scalar(inst.rd, value)
            
        elif inst.opcode == OpCode.VMOV:
            if inst.vd is None or inst.vs1 is None:
                raise ValueError("Invalid vector move instruction")
            value = self.registers.get_vector(inst.vs1)
            self.registers.set_vector(inst.vd, value)
            
        elif inst.opcode == OpCode.MOVI:
            if inst.rd is None or inst.immediate is None:
                raise ValueError("Invalid immediate move instruction")
            self.registers.set_scalar(inst.rd, inst.immediate)
            
    def _execute_control_flow(self, inst: Instruction) -> None:
        """Execute control flow instruction"""
        if inst.opcode == OpCode.JMP:
            if inst.immediate is None:
                raise ValueError("Invalid jump instruction")
            self.registers.pc = np.int32(inst.immediate)
            
        elif inst.opcode in [OpCode.BEQ, OpCode.BNE]:
            if inst.rs1 is None or inst.rs2 is None or inst.immediate is None:
                raise ValueError("Invalid branch instruction")
            a = self.registers.get_scalar(inst.rs1)
            b = self.registers.get_scalar(inst.rs2)
            
            if (inst.opcode == OpCode.BEQ and a == b) or \
               (inst.opcode == OpCode.BNE and a != b):
                self.registers.pc = np.int32(inst.immediate)
                
        elif inst.opcode == OpCode.CALL:
            if inst.immediate is None:
                raise ValueError("Invalid call instruction")
            # Save return address
            self.registers.set_scalar(14, self.registers.pc + 4)  # R14 is link register
            self.registers.pc = np.int32(inst.immediate)
            
        elif inst.opcode == OpCode.RET:
            # Return to saved address
            self.registers.pc = self.registers.get_scalar(14)
            
    def _update_status_flags(self, result: np.int32) -> None:
        """Update status register flags based on result"""
        self.registers.set_status_flags(
            zero=(result == 0),
            negative=(result < 0),
            overflow=False  # TODO: Implement overflow detection
        ) 