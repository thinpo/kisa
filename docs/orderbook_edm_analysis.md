# Orderbook Empirical Dynamic Modeling Analysis Documentation

## Overview

The `orderbook_edm_analysis` program is a specialized tool for analyzing financial market orderbook data using Empirical Dynamic Modeling (EDM) techniques. This program processes orderbook events, calculates event rates over time windows, and applies statistical analysis to determine if the event patterns follow a Poisson distribution. Additionally, it performs autocorrelation analysis and generates visualization data for further exploration.

## Features

The program offers several key features for orderbook data analysis:

1. **Event Rate Calculation**:
   - Processes multiple types of orderbook events (new orders, modifications, deletions, executions)
   - Calculates event rates over configurable time windows
   - Generates time series of event rates for further analysis

2. **Statistical Analysis**:
   - Chi-square goodness-of-fit test for Poisson distribution
   - Variance/mean ratio calculation
   - Nonlinearity index computation
   - Confidence interval determination

3. **Empirical Dynamic Modeling**:
   - Time-delay embedding of event rate time series
   - Nearest neighbor prediction algorithm
   - Nonlinear pattern detection in event rates

4. **Correlation Analysis**:
   - Autocorrelation calculation for each event type
   - Cross-correlation between different event types
   - Correlation matrix generation

5. **Visualization Support**:
   - Generates data files for plotting event rates
   - Creates Gnuplot scripts for visualization
   - Produces correlation matrix visualization data

## Command-Line Options

The program supports the following command-line options:

| Option | Description | Default |
|--------|-------------|---------|
| `--max-events` | Maximum number of events to process | 1000 |
| `--window-size` | Size of time window in seconds | 60 |
| `--edm-dimension` | Embedding dimension for EDM | 3 |
| `--threads` | Number of threads to use | 1 |
| `--input` | Input file path (CSV format) | (generates simulated data) |
| `--output-dir` | Output directory for results | Current directory |
| `--plot` | Generate plotting data and scripts | Disabled |

## Usage

### Building the Program

The program can be built using CMake:

```bash
mkdir -p build && cd build
cmake ..
make
```

### Running the Program

Basic usage:

```bash
./orderbook_edm_analysis
```

With custom options:

```bash
./orderbook_edm_analysis --max-events 5000 --window-size 30 --edm-dimension 5 --threads 4 --input orderbook_data.csv --output-dir ./results --plot
```

### Input Data Format

When using a custom input file, the data should be in CSV format with the following columns:

1. Timestamp (in microseconds)
2. Event type (1 = New order, 2 = Modify order, 3 = Delete order, 4 = Execute)
3. Order ID
4. Price
5. Quantity

Example:
```
1623456789000000,1,12345,100.25,10
1623456789050000,2,12345,100.50,5
1623456789100000,4,12345,100.50,5
1623456789150000,3,12345,0,0
```

If no input file is provided, the program generates simulated orderbook data.

## Output Files

The program generates several output files:

1. **edm_analysis_results.txt**: Contains detailed statistical analysis results
2. **event_rates.dat**: Time series of event rates for each event type
3. **autocorrelation_plot.dat**: Autocorrelation coefficients for each event type
4. **correlation_matrix.dat**: Correlation matrix between different event types
5. **edm_predictions.dat**: EDM predictions vs. actual values

When the `--plot` option is enabled, the program also generates Gnuplot scripts:

1. **plot_rates.gp**: Script for plotting event rates over time
2. **plot_correlation.gp**: Script for visualizing the correlation matrix
3. **plot_predictions.gp**: Script for comparing EDM predictions with actual values

## Technical Implementation

### Event Processing

The program processes orderbook events in the following steps:

1. Read events from input file or generate simulated events
2. Sort events by timestamp
3. Divide events into time windows based on the specified window size
4. Count events of each type within each window
5. Calculate event rates (events per window)

### Statistical Analysis

For each event type, the program performs the following statistical analyses:

1. **Poisson Distribution Test**:
   - Calculate mean (λ) and variance of event rates
   - Compute variance/mean ratio (should be close to 1 for Poisson)
   - Perform chi-square goodness-of-fit test
   - Determine if event rates follow a Poisson distribution at 95% confidence

2. **Nonlinearity Detection**:
   - Calculate nonlinear index based on higher-order moments
   - Compare observed distribution with theoretical Poisson distribution

### Empirical Dynamic Modeling

The EDM implementation includes:

1. **Time-Delay Embedding**:
   - Convert time series into state space representation
   - Create embedding vectors with dimension E (specified by `--edm-dimension`)
   - Build library of historical states

2. **Nearest Neighbor Prediction**:
   - For each prediction point, find k nearest neighbors in state space
   - Weight neighbors by distance (exponential decay)
   - Generate prediction based on weighted average of neighbors' future states

3. **Prediction Evaluation**:
   - Calculate prediction error (RMSE)
   - Compare with null model (mean prediction)
   - Compute prediction skill (improvement over null model)

### Correlation Analysis

The correlation analysis includes:

1. **Autocorrelation**:
   - Calculate autocorrelation function for each event type
   - Determine correlation decay time
   - Identify potential periodicities

2. **Cross-Correlation**:
   - Calculate correlation between different event types
   - Generate correlation matrix
   - Identify relationships between event types

## Performance Considerations

The program is optimized for:

- Multi-threaded processing (configurable with `--threads`)
- Efficient memory usage for large datasets
- Vectorized operations for statistical calculations
- Optimized nearest neighbor search algorithm

## Example Results

### Statistical Analysis Output

The program generates detailed statistical analysis for each event type:

```
=== Event Type: New order ===
Number of events: 250
Mean rate (λ): 12.50
Variance: 11.84
Variance/Mean ratio: 0.95
Nonlinear index: 0.03
Chi-square test statistic: 18.24
Critical value (95%): 28.87
Conclusion: At 95% confidence level, the data conforms to the Poisson distribution assumption.
```

### Correlation Matrix

The correlation matrix shows relationships between different event types:

```
          | New    | Modify | Delete | Execute
----------|--------|--------|--------|--------
New       | 1.000  | 0.324  | 0.156  | 0.412
Modify    | 0.324  | 1.000  | 0.587  | 0.298
Delete    | 0.156  | 0.587  | 1.000  | 0.175
Execute   | 0.412  | 0.298  | 0.175  | 1.000
```

### EDM Prediction Results

The EDM prediction results show the model's ability to predict future event rates:

```
=== EDM Prediction Results ===
Event Type: New order
Embedding Dimension: 3
Prediction Skill: 0.42
RMSE: 2.87
Correlation (predicted vs. actual): 0.68
```

## Limitations

The program has several limitations:

1. Assumes stationarity in the underlying process
2. Limited to fixed-size time windows
3. Does not account for intraday seasonality
4. Simplified order book model
5. No support for limit order book reconstruction
6. No handling of market microstructure effects

## Future Enhancements

Potential enhancements for the program include:

1. Support for variable-sized time windows
2. Integration with real-time market data feeds
3. Advanced visualization capabilities
4. Machine learning-based prediction models
5. Support for high-frequency trading analysis
6. Incorporation of price impact models
7. Integration with order flow imbalance metrics

## Conclusion

The `orderbook_edm_analysis` program provides a comprehensive toolkit for analyzing orderbook event data using statistical and nonlinear time series analysis techniques. By combining traditional statistical tests with modern nonlinear methods like Empirical Dynamic Modeling, the program offers insights into the underlying dynamics of market microstructure that may not be apparent with conventional analysis tools. 