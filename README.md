# K-ISA (K-Inspired Instruction Set Architecture)

This is an implementation of K-ISA, a processor instruction set architecture inspired by the K programming language's emphasis on array operations and minimalism.

## Features

- 32-bit instruction word length
- SIMD (Single Instruction, Multiple Data) support via NEON on ARM64 and AVX-512 on x86_64
- Scalar and vector register operations
- High-precision FFT/IFFT implementation with error control
- Efficient array processing capabilities
- Support for multiple data types (integers, fixed-point, vectors)
- Optimized vector operations (FFT, IFFT, Sort, Reduction, Scan)

## Project Structure

```
.
├── src/           # Source code
│   └── core/      # Core implementation files
├── tests/         # Test files
├── include/       # Header files
└── docs/          # Documentation
```

## Building and Running

1. Create a build directory:
```bash
mkdir build && cd build
```

2. Configure and build:
```bash
cmake ..
make
```

3. Run tests:
```bash
./test_vector_ops  # Run vector operation tests
./kisa_test       # Run general KISA tests
python3 ../tests/perf_test.py  # Run Python performance tests
```

## Performance Comparison

Below is a performance comparison between K-ISA (using NEON on ARM64) and Python/NumPy for vector operations (vector length = 8):

### Basic Arithmetic Operations (ns/op)
| Operation | K-ISA (NEON) | Python/NumPy | Speedup |
|-----------|--------------|--------------|---------|
| Add       | 1.00        | 183.30       | 183x    |
| Sub       | 1.00        | 184.40       | 184x    |
| Mul       | 1.00        | 264.20       | 264x    |
| Div       | 5.20        | 400.80       | 77x     |

### Complex Operations (ns/op)
| Operation | K-ISA (NEON) | Python/NumPy | Speedup |
|-----------|--------------|--------------|---------|
| FFT       | 67.40       | 23038.80     | 342x    |
| IFFT      | 68.00       | 16714.70     | 246x    |
| Sort      | 23.90       | 256.30       | 11x     |

### Reduction Operations (ns/op)
| Operation | K-ISA (NEON) | Python/NumPy | Speedup |
|-----------|--------------|--------------|---------|
| Sum       | 1.30        | 844.20       | 649x    |
| Product   | 1.30        | 880.40       | 677x    |
| Max       | 1.10        | 776.80       | 706x    |
| Min       | 1.00        | 775.20       | 775x    |

### Scan Operations (ns/op)
| Operation | K-ISA (NEON) | Python/NumPy | Speedup |
|-----------|--------------|--------------|---------|
| Sum       | 1.30        | 787.10       | 605x    |
| Max       | 1.00        | 173.70       | 174x    |
| Min       | 1.00        | 171.50       | 172x    |

## Implementation Status

- [x] Register implementation
- [x] Basic arithmetic operations
- [x] Memory operations
- [x] Vector operations
  - [x] High-precision FFT/IFFT with error control
  - [x] Efficient sorting network
  - [x] Optimized reduction operations
  - [x] Fast scan operations
- [x] NEON support for ARM64
- [x] AVX-512 support for x86_64
- [x] Control flow instructions
- [x] Instruction decoder
- [x] Execution unit

## Technical Details

### Vector Operations
- Optimized for small vector operations (length = 8)
- Uses NEON intrinsics on ARM64 platforms
- Uses AVX-512 intrinsics on x86_64 platforms
- Implements efficient bitonic sort for vector sorting

### FFT/IFFT Implementation
- High-precision fixed-point arithmetic
- 64-bit intermediate computations
- Error-controlled implementation (max error ≤ 3)
- Optimized butterfly operations
- Efficient bit-reversal algorithm

### Performance Optimizations
- SIMD vectorization for basic operations
- Cache-friendly memory access patterns
- Minimized branching in critical paths
- Efficient register allocation
- Optimized reduction and scan operations

### Error Control
- Fixed-point arithmetic with 32.32 format
- Rounding for better precision
- Error bounds for FFT/IFFT operations
- Validated against reference implementations

## License

MIT License 