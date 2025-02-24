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
- Empirical Dynamic Modeling (EDM) for time series analysis and prediction
- Financial market microstructure analysis tools
- Simplified Large Language Model (LLM) computation framework

## Project Structure

```
.
├── src/                       # C implementation source code
│   ├── core/                  # Core implementation files
│   │   ├── vector_ops.c       # Vector operations implementation
│   │   ├── execution.c        # Execution unit implementation
│   │   ├── control_flow.c     # Control flow instructions
│   │   └── instruction.c      # Instruction decoder
│   ├── edm/                   # Empirical Dynamic Modeling
│   │   ├── embedding.c        # Time-delay embedding implementation
│   │   └── prediction.c       # Prediction algorithms
│   └── market/                # Financial market analysis
│       ├── orderbook.c        # Order book simulation
│       └── event_analysis.c   # Event analysis tools
├── python/                    # Python implementation
│   ├── kisa/                  # Python package
│   │   ├── __init__.py        # Package initialization
│   │   ├── vector_ops.py      # Vector operations in Python
│   │   ├── edm.py             # EDM implementation in Python
│   │   └── market.py          # Market analysis in Python
│   ├── examples/              # Python examples
│   │   ├── llm_calc.py        # LLM calculation in Python
│   │   └── orderbook_analysis.py # Order book analysis in Python
│   └── tests/                 # Python tests
│       └── perf_test.py       # Performance tests
├── include/                   # C header files
│   ├── core/                  # Core header files
│   ├── edm/                   # EDM header files
│   └── market/                # Market analysis header files
├── examples/                  # C example applications
│   ├── llm_calc.c             # Simplified LLM calculation
│   └── orderbook_edm_analysis.c # Order book event analysis
├── tests/                     # C test files
│   ├── test_vector_ops.c      # Vector operations tests
│   └── kisa_test.c            # General KISA tests
└── docs/                      # Documentation
    ├── architecture.md        # Architecture documentation
    ├── vector_ops.md          # Vector operations documentation
    ├── edm.md                 # EDM documentation
    ├── c_implementation.md    # C implementation details
    └── python_implementation.md # Python implementation details
```

## Building and Running

### C Implementation

1. Create a build directory:
```bash
mkdir build && cd build
```

2. Configure and build:
```bash
cmake ..
make
```

3. Run C tests:
```bash
./test_vector_ops  # Run vector operation tests
./kisa_test        # Run general KISA tests
```

4. Run C example applications:
```bash
# Run the simplified LLM calculation
./llm_calc

# Run the order book analysis with default settings
./orderbook_edm_analysis

# Run with command-line options
./orderbook_edm_analysis --events 5000 --window 50 --threads 8 --plot

# Run with input from real data and generate plots
./orderbook_edm_analysis --input real_orderbook_data.csv --output results --plot
```

### Python Implementation

1. Install the Python package:
```bash
cd python
pip install -e .
```

2. Run Python tests:
```bash
python tests/perf_test.py  # Run Python performance tests
```

3. Run Python examples:
```bash
python examples/llm_calc.py
python examples/orderbook_analysis.py
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

### C Implementation
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
- [x] Empirical Dynamic Modeling (EDM)
- [x] Financial market microstructure analysis
- [x] Simplified LLM computation framework

### Python Implementation
- [x] Basic vector operations
- [x] FFT/IFFT operations (using NumPy)
- [x] Sorting operations
- [x] Reduction operations
- [x] Scan operations
- [x] EDM implementation
- [x] Market microstructure analysis
- [x] LLM computation framework

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

## C Implementation

The C implementation provides a high-performance, low-level implementation of the K-ISA architecture, optimized for speed and efficiency.

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

## Python Implementation

The Python implementation provides a more accessible, high-level interface to the K-ISA functionality, making it easier to prototype and experiment with algorithms.

### Vector Operations
- NumPy-based implementation for vector operations
- Clean, Pythonic API for ease of use
- Integration with the Python scientific ecosystem
- Support for larger vector sizes than the C implementation

### Performance Considerations
- While slower than the C implementation, the Python version prioritizes:
  - Readability and maintainability
  - Ease of experimentation
  - Integration with Python data science tools
  - Visualization capabilities

### Python-Specific Features
- Jupyter notebook support for interactive exploration
- Matplotlib integration for visualization
- Pandas integration for data manipulation
- Scikit-learn compatibility for machine learning applications

## Empirical Dynamic Modeling (EDM)

K-ISA now includes a powerful implementation of Empirical Dynamic Modeling (EDM), a non-parametric method for analyzing and predicting complex nonlinear dynamical systems.

### EDM Features
- Time-delay embedding for state space reconstruction
- Nearest neighbor search for similar states
- Weighted prediction based on historical patterns
- Support for multivariate time series analysis
- Optimized for small to medium datasets

### EDM Technical Implementation
- Efficient Euclidean distance calculation using SIMD
- Optimized k-nearest neighbor search
- Weighted prediction with customizable parameters
- Time-delay embedding with variable dimensions
- Memory-efficient library construction

### EDM Applications
- Time series prediction
- Nonlinearity detection
- Causality analysis
- System dynamics characterization
- Anomaly detection in complex systems

## Financial Market Microstructure Analysis

K-ISA includes specialized tools for analyzing financial market microstructure, with a focus on order book dynamics and event analysis.

### Order Book Analysis Features
- Event arrival rate analysis (new orders, modifications, deletions, executions)
- Poisson process testing for event arrivals
- Autocorrelation analysis for temporal patterns
- Cross-correlation analysis between event types
- EDM-based prediction of future event rates

### Market Microstructure Technical Implementation
- Efficient event simulation with customizable parameters
- Time-window based rate calculation
- Statistical analysis tools for event patterns
- Visualization of correlation structures
- EDM integration for nonlinear pattern detection

### Market Analysis Applications
- High-frequency trading strategy development
- Market making optimization
- Liquidity analysis
- Market impact modeling
- Anomaly and manipulation detection
- Risk management in electronic markets

## Simplified LLM Computation Framework

K-ISA includes a simplified implementation of Large Language Model (LLM) forward propagation, demonstrating how vector operations can be used to build complex AI systems.

### LLM Architecture
- Multi-layer Transformer architecture
- Configurable embedding dimension (default: 8)
- Multi-head attention mechanism (default: 2 heads)
- Layer normalization with numerical stability enhancements
- Positional encoding using sine/cosine functions
- Advanced FFT-based sequence processing
- Integration with Empirical Dynamic Modeling (EDM)

### LLM Components
- **Matrix-Vector Multiplication**: Efficient implementation of linear transformations
- **Multi-Head Attention**: 
  - Parallel processing of attention heads
  - Query, key, and value projections
  - Attention score calculation and normalization
  - Head concatenation and final projection
- **Layer Normalization**: 
  - Mean and variance calculation
  - Normalization with numerical stability
  - Scale and shift operations
- **Positional Encoding**: 
  - Sinusoidal position embeddings
  - Position-aware sequence processing
- **FFT Processing**: 
  - Frequency domain convolution
  - Frequency feature extraction
  - Adaptive signal enhancement based on spectral characteristics
- **EDM Integration**:
  - Time-delay embedding of transformer outputs
  - Prediction of sequence dynamics
  - Non-linear pattern detection

### LLM Technical Implementation
- Optimized vector operations using K-ISA primitives
- Memory-efficient implementation suitable for embedded systems
- Fixed-point arithmetic for numerical stability
- Configurable model parameters (layers, heads, dimensions)
- Detailed logging and visualization of intermediate states
- Modular design for easy extension and modification

### LLM Performance
- Processing of 8-dimensional vectors through a 2-layer transformer
- Efficient computation even on resource-constrained hardware
- Demonstration of key transformer operations in a simplified setting
- Integration of frequency domain processing with attention mechanisms
- Combination of traditional deep learning with dynamical systems analysis

## Example Applications

### C Implementation

#### Simplified LLM Calculation (`examples/llm_calc.c`)
The repository includes a simplified implementation of Large Language Model (LLM) forward propagation, demonstrating:
- Matrix-vector multiplication (simulating linear layers)
- Vector activation functions (ReLU)
- Multi-head attention mechanism
- Advanced sequence processing using FFT
- Layer normalization
- Positional encoding
- Integration with Empirical Dynamic Modeling

To run the LLM calculation example:
```bash
# From the project root directory
gcc -o llm_calc examples/llm_calc.c src/core/vector_ops.c src/core/execution.c src/core/control_flow.c src/core/instruction.c -I. -lm
./llm_calc
```

#### Order Book Event Analysis (`examples/orderbook_edm_analysis.c`)
The repository includes an enhanced program for analyzing order book events using EDM:

**Features:**
- Simulates or loads real order book event streams
- Analyzes arrival rates for different event types
- Tests for Poisson process characteristics
- Detects temporal patterns and correlations
- Predicts future event rates using EDM
- Characterizes nonlinear dynamics in market events

**Advanced Capabilities:**
- **Command-line Configuration**: Customizable parameters via command-line arguments
  ```
  --events      Set number of events to analyze (default: 1000)
  --window      Set time window size for rate calculation (default: 100)
  --dimension   Set EDM embedding dimension (default: 3) 
  --threads     Set number of threads for parallel processing (default: 4)
  --input       Read order book events from CSV file
  --output      Set output directory for results (default: current dir)
  --plot        Generate data files for visualization
  ```
- **File I/O**: 
  - Read real order book data from CSV files
  - Flexible CSV format parsing with automatic header detection
  - Support for different event type formats (numeric or text)
- **Visualization Data Generation**: 
  - Generate data files for plotting event rates, correlations, and predictions
  - Create Gnuplot scripts for automated visualization
  - Output correlation matrices and heatmaps
  - Export time series and forecast data
- **Multi-threading**: 
  - Parallel processing of large datasets
  - Configurable thread count for optimal performance
  - Automatic fallback to single-threaded mode for small datasets

To run the order book analysis example:
```bash
# Basic usage
gcc -o orderbook_edm_analysis examples/orderbook_edm_analysis.c src/core/vector_ops.c src/core/execution.c -I. -lm -pthread
./orderbook_edm_analysis

# With command-line options
./orderbook_edm_analysis --events 5000 --window 50 --threads 8 --plot

# Using real data
./orderbook_edm_analysis --input market_data.csv --output results --plot
```

**Output Examples:**
- Event rate statistics and temporal patterns
- Autocorrelation plots for each event type
- Cross-correlation matrices between event types
- EDM-based predictions of future event rates
- Chi-square test results for Poisson process validation
- Nonlinearity indicators and system characterization

### Python Implementation

#### LLM Calculation (`python/examples/llm_calc.py`)
The Python version of the LLM calculation provides:
- A more accessible implementation for experimentation
- Visualization of attention patterns and layer activations
- Easy modification of model parameters
- Integration with popular Python ML libraries

To run the Python LLM calculation example:
```bash
# From the project root directory
python python/examples/llm_calc.py
```

#### Order Book Analysis (`python/examples/orderbook_analysis.py`)
The Python version of the order book analysis provides:
- Interactive visualization of order book events
- Statistical analysis tools built on SciPy and StatsModels
- Easy parameter tuning for simulation
- Integration with pandas for data manipulation

To run the Python order book analysis example:
```bash
# From the project root directory
python python/examples/orderbook_analysis.py
```

## License

MIT License 