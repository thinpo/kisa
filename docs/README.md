# K-ISA Project Documentation

## Overview

The K-ISA (K-Inspired Instruction Set Architecture) project is a framework for efficient vector operations, particularly designed for computational finance and machine learning applications. This documentation covers the main applications built using the K-ISA framework.

## Applications

The K-ISA project includes the following applications:

1. [LLM Calculation (llm_calc)](llm_calc.md) - A simplified implementation of Large Language Model forward propagation calculations.
2. [Orderbook EDM Analysis (orderbook_edm_analysis)](orderbook_edm_analysis.md) - A tool for analyzing financial market orderbook data using Empirical Dynamic Modeling techniques.

## Building the Project

The K-ISA project uses CMake as its build system. To build all applications:

```bash
mkdir -p build && cd build
cmake ..
make
```

This will build the K-ISA library and all applications.

## Core Library

The K-ISA core library provides efficient vector operations for both ARM64 (with NEON support) and x86_64 (with AVX-512 support) architectures. Key features include:

- Vector arithmetic operations (add, subtract, multiply, divide)
- Vector reduction operations (sum, max, min)
- Vector comparison and selection
- FFT and IFFT operations
- Matrix-vector multiplication
- Control flow operations

## Testing

The project includes several test programs to verify the correctness and performance of the K-ISA library:

- `test_vector_ops` - Tests basic vector operations
- `test_control_flow` - Tests control flow operations
- `kisa_test` - Comprehensive test suite for all K-ISA functionality
- `perf_test` - Performance benchmarks for K-ISA operations

To run all tests:

```bash
cd build
ctest
```

## Architecture Support

The K-ISA framework is designed to work efficiently on:

- ARM64 platforms with NEON support
- x86_64 platforms with AVX-512 support

The implementation automatically detects the available instruction set and uses the most efficient implementation.

## Directory Structure

```
kisa/
├── include/           # Public header files
├── src/
│   ├── core/          # Core K-ISA implementation
│   └── apps/          # Application programs
├── examples/          # Example programs
├── tests/             # Test programs
└── docs/              # Documentation
```

## License

The K-ISA project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions to the K-ISA project are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests to ensure they pass
5. Submit a pull request

## Contact

For questions or support, please open an issue on the project's GitHub repository. 