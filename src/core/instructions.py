from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import numpy as np

class InstructionType(Enum):
    """Instruction types supported by K-ISA"""
    SCALAR_ARITHMETIC = auto()
    VECTOR_ARITHMETIC = auto()
    MEMORY = auto()
    CONTROL_FLOW = auto()
    ARRAY_OPS = auto()
    DATA_MOVEMENT = auto()

class OpCode(Enum):
    """Operation codes for K-ISA instructions"""
    # Scalar Arithmetic
    ADD = 0x01
    SUB = 0x02
    MUL = 0x03
    DIV = 0x04
    
    # Vector Arithmetic
    VADD = 0x11
    VSUB = 0x12
    VMUL = 0x13
    VDIV = 0x14
    
    # Memory Access
    LD = 0x21
    ST = 0x22
    VLD = 0x23
    VST = 0x24
    
    # Control Flow
    BEQ = 0x31
    BNE = 0x32
    JMP = 0x33
    CALL = 0x34
    RET = 0x35
    
    # Array Operations
    SUM = 0x41
    PROD = 0x42
    MAX = 0x43
    MIN = 0x44
    
    # Data Movement
    MOV = 0x51
    VMOV = 0x52
    MOVI = 0x53

@dataclass
class Instruction:
    """Base class for all K-ISA instructions"""
    opcode: OpCode
    type: InstructionType
    
    # Registers
    rd: Optional[int] = None  # Destination register
    rs1: Optional[int] = None  # Source register 1
    rs2: Optional[int] = None  # Source register 2
    
    # Vector registers
    vd: Optional[int] = None  # Vector destination register
    vs1: Optional[int] = None  # Vector source register 1
    vs2: Optional[int] = None  # Vector source register 2
    
    # Immediate value and offset
    immediate: Optional[int] = None
    offset: Optional[int] = None
    
    def encode(self) -> np.uint32:
        """Encode instruction into 32-bit word"""
        encoded = np.uint32(0)
        
        # Encode opcode (bits 0-5)
        encoded |= np.uint32(self.opcode.value) & 0x3F
        
        # Encode registers based on instruction type
        if self.type == InstructionType.SCALAR_ARITHMETIC:
            if self.rd is not None:
                encoded |= (self.rd & 0xF) << 6
            if self.rs1 is not None:
                encoded |= (self.rs1 & 0xF) << 10
            if self.rs2 is not None:
                encoded |= (self.rs2 & 0xF) << 14
                
        elif self.type == InstructionType.VECTOR_ARITHMETIC:
            if self.vd is not None:
                encoded |= (self.vd & 0x7) << 6
            if self.vs1 is not None:
                encoded |= (self.vs1 & 0x7) << 9
            if self.vs2 is not None:
                encoded |= (self.vs2 & 0x7) << 12
                
        # Encode immediate value if present (bits 14-31)
        if self.immediate is not None:
            encoded |= (self.immediate & 0x3FFFF) << 14
            
        return encoded
    
    @staticmethod
    def decode(word: np.uint32) -> 'Instruction':
        """Decode 32-bit word into instruction"""
        # Extract opcode
        opcode_value = word & 0x3F
        opcode = OpCode(opcode_value)
        
        # Determine instruction type based on opcode
        if opcode_value <= 0x10:
            inst_type = InstructionType.SCALAR_ARITHMETIC
        elif opcode_value <= 0x20:
            inst_type = InstructionType.VECTOR_ARITHMETIC
        elif opcode_value <= 0x30:
            inst_type = InstructionType.MEMORY
        elif opcode_value <= 0x40:
            inst_type = InstructionType.CONTROL_FLOW
        elif opcode_value <= 0x50:
            inst_type = InstructionType.ARRAY_OPS
        else:
            inst_type = InstructionType.DATA_MOVEMENT
            
        # Create instruction object
        inst = Instruction(opcode=opcode, type=inst_type)
        
        # Decode registers based on instruction type
        if inst_type == InstructionType.SCALAR_ARITHMETIC:
            inst.rd = (word >> 6) & 0xF
            inst.rs1 = (word >> 10) & 0xF
            inst.rs2 = (word >> 14) & 0xF
            
        elif inst_type == InstructionType.VECTOR_ARITHMETIC:
            inst.vd = (word >> 6) & 0x7
            inst.vs1 = (word >> 9) & 0x7
            inst.vs2 = (word >> 12) & 0x7
            
        # Decode immediate value if present
        if inst_type in [InstructionType.DATA_MOVEMENT, InstructionType.MEMORY]:
            inst.immediate = (word >> 14) & 0x3FFFF
            
        return inst 