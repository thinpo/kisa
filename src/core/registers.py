import numpy as np
from typing import Dict, Union, List
from dataclasses import dataclass

@dataclass
class RegisterFile:
    """Register file containing both scalar and vector registers"""
    
    def __init__(self):
        # Initialize scalar registers (R0-R15)
        self.scalar_registers: Dict[int, np.int32] = {
            i: np.int32(0) for i in range(16)
        }
        
        # Initialize vector registers (V0-V7)
        self.vector_registers: Dict[int, np.ndarray] = {
            i: np.zeros(8, dtype=np.int32) for i in range(8)
        }
        
        # Special purpose registers
        self.pc: np.int32 = np.int32(0)  # Program Counter
        self.sp: np.int32 = np.int32(0)  # Stack Pointer
        self.sr: np.int32 = np.int32(0)  # Status Register
        
    def get_scalar(self, reg_num: int) -> np.int32:
        """Get value from scalar register"""
        if reg_num < 0 or reg_num > 15:
            raise ValueError(f"Invalid scalar register number: {reg_num}")
        return self.scalar_registers[reg_num]
    
    def set_scalar(self, reg_num: int, value: Union[int, np.int32]) -> None:
        """Set value in scalar register"""
        if reg_num < 0 or reg_num > 15:
            raise ValueError(f"Invalid scalar register number: {reg_num}")
        if reg_num == 0:  # R0 is always 0
            return
        self.scalar_registers[reg_num] = np.int32(value)
    
    def get_vector(self, reg_num: int) -> np.ndarray:
        """Get value from vector register"""
        if reg_num < 0 or reg_num > 7:
            raise ValueError(f"Invalid vector register number: {reg_num}")
        return self.vector_registers[reg_num]
    
    def set_vector(self, reg_num: int, value: Union[List[int], np.ndarray]) -> None:
        """Set value in vector register"""
        if reg_num < 0 or reg_num > 7:
            raise ValueError(f"Invalid vector register number: {reg_num}")
        if isinstance(value, list):
            value = np.array(value, dtype=np.int32)
        if value.shape != (8,):
            raise ValueError("Vector must have exactly 8 elements")
        self.vector_registers[reg_num] = value.astype(np.int32)
    
    def get_status_flags(self) -> Dict[str, bool]:
        """Get status register flags"""
        return {
            'zero': bool(self.sr & 1),
            'negative': bool(self.sr & 2),
            'overflow': bool(self.sr & 4),
        }
    
    def set_status_flags(self, zero: bool = False, negative: bool = False, overflow: bool = False) -> None:
        """Set status register flags"""
        self.sr = np.int32(
            (1 if zero else 0) |
            (2 if negative else 0) |
            (4 if overflow else 0)
        ) 