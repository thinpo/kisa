# K-ISA v0.1.0 Release Notes

We are excited to announce the first official release of K-ISA (K-Inspired Instruction Set Architecture)!

## Overview

K-ISA is a framework for efficient vector operations, particularly designed for computational finance and machine learning applications. This release includes a complete implementation of the core vector operations library, along with two specialized applications: a simplified LLM calculation framework and an orderbook analysis tool using Empirical Dynamic Modeling (EDM).

## Key Features

### Core Library
- Optimized vector operations for ARM64 (with NEON support) and x86_64 (with AVX-512 support)
- High-performance FFT/IFFT implementation with error control
- Efficient sorting, reduction, and scan operations
- Comprehensive test suite for validation

### LLM Calculation Framework
- Simplified implementation of transformer-based language model forward propagation
- Multi-head attention mechanism
- Layer normalization with numerical stability
- Positional encoding using sine/cosine functions
- Advanced FFT-based sequence processing
- Integration with Empirical Dynamic Modeling (EDM)

### Orderbook EDM Analysis
- Event rate calculation for different orderbook event types
- Statistical analysis with Poisson distribution testing
- Autocorrelation and cross-correlation analysis
- Empirical Dynamic Modeling for nonlinear pattern detection
- Visualization data generation for further exploration

## Changes in This Release

- Moved orderbook_edm_analysis.c from src/apps to examples directory
- Fixed include paths in llm_calc.c
- Updated CMakeLists.txt to reflect file structure changes
- Added comprehensive documentation in docs/ directory
- Translated comments in kisa.h to English
- Removed backup files
- Added build system improvements

## Documentation

Comprehensive documentation has been added to the docs/ directory:
- General project overview and build instructions
- Detailed documentation for the LLM calculation framework
- Detailed documentation for the orderbook EDM analysis tool

## Performance

The K-ISA framework demonstrates significant performance improvements over equivalent Python/NumPy implementations:
- Basic arithmetic operations: 77x-264x speedup
- Complex operations (FFT/IFFT): 246x-342x speedup
- Reduction operations: 649x-775x speedup
- Scan operations: 172x-605x speedup

## Building and Running

```bash
# Build the project
mkdir -p build && cd build
cmake ..
make

# Run the LLM calculation example
./llm_calc

# Run the orderbook analysis example
./orderbook_edm_analysis
```

## Future Plans

- Support for longer sequences in the LLM framework
- More sophisticated attention mechanisms
- Integration with actual text processing
- Support for different model architectures
- Performance optimizations for specific hardware
- Visualization of attention patterns
- Support for variable-sized time windows in orderbook analysis
- Integration with real-time market data feeds
- Advanced visualization capabilities

## License

K-ISA is licensed under the MIT License. 