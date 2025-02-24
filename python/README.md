# K-ISA Python Package

This is the Python implementation of K-ISA (Knowledge-based Instruction Set Architecture), a library for vector operations, empirical dynamic modeling (EDM), and market analysis.

## Installation

You can install the package directly from the source:

```bash
cd python
pip install -e .
```

Or install with development dependencies:

```bash
cd python
pip install -e ".[dev]"
```

## Features

The K-ISA Python package provides the following modules:

### Vector Operations

The `kisa.vector_ops` module provides efficient vector operations:

```python
import numpy as np
from kisa.vector_ops import vector_add, vector_mul, vector_fft

# Create vectors
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Perform operations
result_add = vector_add(a, b)  # [6, 8, 10, 12]
result_mul = vector_mul(a, b)  # [5, 12, 21, 32]
result_fft = vector_fft(a)     # FFT of vector a
```

### Empirical Dynamic Modeling (EDM)

The `kisa.edm` module provides tools for nonlinear time series analysis:

```python
import numpy as np
from kisa.edm import time_delay_embedding, simplex_projection

# Create a time series
t = np.linspace(0, 10 * np.pi, 500)
time_series = np.sin(t)

# Perform time-delay embedding
E = 3  # embedding dimension
tau = 5  # time delay
embedded = time_delay_embedding(time_series, E, tau)

# Perform simplex projection
predictions, skill = simplex_projection(time_series, E, tau, 1, 3)
print(f"Prediction skill: {skill}")
```

### Market Analysis

The `kisa.market` module provides tools for analyzing financial market data:

```python
from kisa.market import generate_simulated_events, calculate_event_rates, analyze_event_rates_with_edm

# Generate simulated orderbook events
events = generate_simulated_events(1000)

# Calculate event rates in 60-second windows
rates = calculate_event_rates(events, 60)

# Analyze event rates using EDM
edm_results = analyze_event_rates_with_edm(rates, 3, 1, 1, 3)
```

## Examples

The package includes example scripts in the `examples` directory:

- `llm_calc.py`: Demonstrates using vector operations for a simplified transformer-based language model.
- `orderbook_analysis.py`: Demonstrates analyzing orderbook events and applying EDM techniques.

To run the examples:

```bash
cd python
python examples/llm_calc.py
python examples/orderbook_analysis.py --max-events 1000 --window-size 60 --plot
```

## Testing

Run the tests using pytest:

```bash
cd python
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 