#!/usr/bin/env python3
"""
Orderbook Analysis Example using K-ISA Python package.

This example demonstrates how to use the K-ISA Python package to analyze
orderbook events, calculate event rates, and apply Empirical Dynamic Modeling
for prediction and analysis.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Add parent directory to path to import kisa package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kisa.market import OrderEvent, EventRates, EventType
import kisa.market as market


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Orderbook Event Analysis')
    parser.add_argument('--max-events', type=int, default=1000,
                        help='Maximum number of events to process')
    parser.add_argument('--window-size', type=int, default=60,
                        help='Size of time window in seconds')
    parser.add_argument('--edm-dimension', type=int, default=3,
                        help='Embedding dimension for EDM')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to use')
    parser.add_argument('--input', type=str, default=None,
                        help='Input file path (CSV format)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plotting data and scripts')
    
    return parser.parse_args()


def read_events_from_file(file_path: str, max_events: int) -> List[OrderEvent]:
    """Read orderbook events from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        max_events: Maximum number of events to read
        
    Returns:
        List of OrderEvent objects
    """
    events = []
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            
            # Check if first row is header
            first_row = next(reader)
            try:
                # Try to parse first row as data
                timestamp = int(first_row[0])
                event_type = int(first_row[1])
                order_id = int(first_row[2])
                price = float(first_row[3])
                quantity = int(first_row[4])
                
                # If successful, add as event
                events.append(OrderEvent(
                    timestamp=timestamp,
                    type=EventType(event_type - 1),  # Adjust for 0-based enum
                    order_id=order_id,
                    price=price,
                    quantity=quantity
                ))
            except ValueError:
                # First row is header, skip
                pass
            
            # Read remaining rows
            for row in reader:
                if len(events) >= max_events:
                    break
                
                try:
                    timestamp = int(row[0])
                    event_type = int(row[1])
                    order_id = int(row[2])
                    price = float(row[3])
                    quantity = int(row[4])
                    
                    events.append(OrderEvent(
                        timestamp=timestamp,
                        type=EventType(event_type - 1),  # Adjust for 0-based enum
                        order_id=order_id,
                        price=price,
                        quantity=quantity
                    ))
                except (ValueError, IndexError):
                    print(f"Warning: Skipping invalid row: {row}")
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    print(f"Read {len(events)} events from {file_path}")
    return events


def perform_chi_square_test(observed_counts: List[int], expected_counts: List[float]) -> Tuple[float, float]:
    """Perform chi-square goodness-of-fit test.
    
    Args:
        observed_counts: Observed frequency counts
        expected_counts: Expected frequency counts
        
    Returns:
        Tuple of (chi_square_statistic, p_value)
    """
    chi_square = 0.0
    
    for obs, exp in zip(observed_counts, expected_counts):
        if exp > 0:
            chi_square += ((obs - exp) ** 2) / exp
    
    # Degrees of freedom (number of categories - 1)
    df = len(observed_counts) - 1
    
    # Critical values for 95% confidence (simplified)
    critical_values = {
        1: 3.84,
        2: 5.99,
        3: 7.81,
        4: 9.49,
        5: 11.07,
        10: 18.31,
        15: 25.00,
        20: 31.41,
        30: 43.77
    }
    
    # Find closest critical value
    critical_value = critical_values.get(df, 0)
    if critical_value == 0:
        # Find closest key
        keys = list(critical_values.keys())
        closest_key = min(keys, key=lambda k: abs(k - df))
        critical_value = critical_values[closest_key]
    
    return chi_square, critical_value


def test_poisson_distribution(rates: List[float]) -> Dict[str, float]:
    """Test if event rates follow a Poisson distribution.
    
    Args:
        rates: List of event rates
        
    Returns:
        Dictionary with test results
    """
    # Calculate mean (lambda)
    mean = np.mean(rates)
    
    # Calculate variance
    variance = np.var(rates)
    
    # For Poisson distribution, variance should equal mean
    variance_mean_ratio = variance / mean if mean > 0 else 0
    
    # Calculate nonlinear index (simplified)
    nonlinear_index = np.mean(np.abs(np.diff(rates)))
    
    # Count frequencies of different rates
    max_rate = int(max(rates)) + 1
    observed_counts = [0] * max_rate
    for rate in rates:
        observed_counts[int(rate)] += 1
    
    # Calculate expected frequencies based on Poisson distribution
    expected_counts = []
    total_observations = len(rates)
    
    for k in range(max_rate):
        # Poisson probability mass function: P(X = k) = (lambda^k * e^-lambda) / k!
        probability = (mean ** k * np.exp(-mean)) / np.math.factorial(k)
        expected_counts.append(probability * total_observations)
    
    # Perform chi-square test
    chi_square, critical_value = perform_chi_square_test(observed_counts, expected_counts)
    
    # Determine if distribution fits Poisson
    is_poisson = chi_square <= critical_value
    
    return {
        "mean": mean,
        "variance": variance,
        "variance_mean_ratio": variance_mean_ratio,
        "nonlinear_index": nonlinear_index,
        "chi_square": chi_square,
        "critical_value": critical_value,
        "is_poisson": is_poisson
    }


def generate_plots(rates: List[EventRates], edm_results: Dict, output_dir: str):
    """Generate plots for visualization.
    
    Args:
        rates: List of event rates
        edm_results: EDM analysis results
        output_dir: Output directory for plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    timestamps = [rate.window_start for rate in rates]
    event_types = ["New order", "Modify order", "Delete order", "Execute"]
    
    # Plot event rates
    plt.figure(figsize=(12, 8))
    for i, event_type in enumerate(EventType):
        type_rates = [rate.rates[event_type.value] for rate in rates]
        plt.plot(timestamps, type_rates, label=event_types[event_type.value])
    
    plt.title("Event Rates Over Time")
    plt.xlabel("Time")
    plt.ylabel("Event Rate (events/second)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "event_rates.png"))
    
    # Plot autocorrelation
    plt.figure(figsize=(12, 8))
    for i, event_type in enumerate(EventType):
        if event_type in edm_results["autocorrelation"]:
            autocorr = edm_results["autocorrelation"][event_type]
            plt.plot(range(1, len(autocorr) + 1), autocorr, 
                    label=event_types[event_type.value])
    
    plt.title("Autocorrelation of Event Rates")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation Coefficient")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "autocorrelation.png"))
    
    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    correlation_matrix = edm_results["correlation_matrix"]
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title("Correlation Between Event Types")
    plt.xticks(range(len(event_types)), event_types, rotation=45)
    plt.yticks(range(len(event_types)), event_types)
    
    # Add correlation values
    for i in range(len(event_types)):
        for j in range(len(event_types)):
            plt.text(j, i, f"{correlation_matrix[i, j]:.2f}", 
                    ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    
    # Save data files for gnuplot
    with open(os.path.join(output_dir, "event_rates.dat"), 'w') as f:
        f.write("# Time New_Order Modify_Order Delete_Order Execute\n")
        for i, rate in enumerate(rates):
            f.write(f"{rate.window_start} {rate.rates[0]} {rate.rates[1]} {rate.rates[2]} {rate.rates[3]}\n")
    
    # Create gnuplot script for event rates
    with open(os.path.join(output_dir, "plot_rates.gp"), 'w') as f:
        f.write("""
set terminal png size 1200,800
set output 'event_rates_plot.png'
set title 'Event Rates Over Time'
set xlabel 'Time'
set ylabel 'Event Rate (events/second)'
set grid
set key outside
plot 'event_rates.dat' using 1:2 with lines title 'New Order', \\
     'event_rates.dat' using 1:3 with lines title 'Modify Order', \\
     'event_rates.dat' using 1:4 with lines title 'Delete Order', \\
     'event_rates.dat' using 1:5 with lines title 'Execute'
""")
    
    print(f"Plots and data files saved to {output_dir}")


def main():
    """Main function to demonstrate orderbook analysis."""
    args = parse_arguments()
    
    print("=== Orderbook Event Analysis ===\n")
    print(f"Configuration:")
    print(f"  Max Events: {args.max_events}")
    print(f"  Window Size: {args.window_size} seconds")
    print(f"  EDM Dimension: {args.edm_dimension}")
    print(f"  Threads: {args.threads}")
    print(f"  Input File: {args.input if args.input else 'None (using simulated data)'}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Generate Plots: {args.plot}")
    
    # Get events (from file or generate)
    if args.input:
        events = read_events_from_file(args.input, args.max_events)
    else:
        events = market.generate_simulated_events(args.max_events)
    
    if not events:
        print("No events to analyze. Exiting.")
        return
    
    # Calculate event rates
    rates = market.calculate_event_rates(events, args.window_size)
    
    if not rates:
        print("No rate data to analyze. Exiting.")
        return
    
    # Print event rates statistics
    print("\n=== Event rates statistics ===")
    print(f"{'Time window':<15} {'New order rate':<15} {'Modify order rate':<15} "
          f"{'Delete order rate':<15} {'Trade execution rate':<15} {'Total event rate':<15}")
    
    for rate in rates:
        total_rate = sum(rate.rates)
        print(f"{rate.window_start:<15} {rate.rates[0]:<15.4f} {rate.rates[1]:<15.4f} "
              f"{rate.rates[2]:<15.4f} {rate.rates[3]:<15.4f} {total_rate:<15.4f}")
    
    # Analyze autocorrelation
    autocorr_results = market.analyze_autocorrelation(rates)
    
    # Analyze correlation between event types
    correlation_matrix = market.analyze_cross_correlation(rates)
    
    # Analyze event rates using EDM
    edm_results = market.analyze_event_rates_with_edm(
        rates, args.edm_dimension, 1, 3, 1)
    
    # Test for Poisson distribution
    print("\n=== Poisson Distribution Test ===")
    poisson_results = {}
    
    for event_type in EventType:
        type_idx = event_type.value
        event_name = ["New order", "Modify order", "Delete order", "Trade execution"][type_idx]
        
        # Extract rates for this event type
        type_rates = [rate.rates[type_idx] for rate in rates]
        
        # Test for Poisson distribution
        result = test_poisson_distribution(type_rates)
        poisson_results[event_type] = result
        
        print(f"\n=== Event Type: {event_name} ===")
        print(f"Number of events: {sum(rate.counts[type_idx] for rate in rates)}")
        print(f"Mean rate (λ): {result['mean']:.2f}")
        print(f"Variance: {result['variance']:.2f}")
        print(f"Variance/Mean ratio: {result['variance_mean_ratio']:.2f}")
        print(f"Nonlinear index: {result['nonlinear_index']:.2f}")
        print(f"Chi-square test statistic: {result['chi_square']:.2f}")
        print(f"Critical value (95%): {result['critical_value']:.2f}")
        
        if result['is_poisson']:
            print("Conclusion: At 95% confidence level, the data conforms to the Poisson distribution assumption.")
        else:
            print("Conclusion: At 95% confidence level, the data does not conform to the Poisson distribution assumption.")
    
    # Combine all results
    all_results = {
        "autocorrelation": autocorr_results,
        "correlation_matrix": correlation_matrix,
        "edm_results": edm_results,
        "poisson_results": poisson_results
    }
    
    # Generate plots if requested
    if args.plot:
        generate_plots(rates, all_results, args.output_dir)
    
    # Save analysis results to file
    results_file = os.path.join(args.output_dir, "edm_analysis_results.txt")
    with open(results_file, 'w') as f:
        f.write("=== Orderbook Event Analysis Results ===\n\n")
        
        f.write("=== Configuration ===\n")
        f.write(f"Max Events: {args.max_events}\n")
        f.write(f"Window Size: {args.window_size} seconds\n")
        f.write(f"EDM Dimension: {args.edm_dimension}\n")
        f.write(f"Threads: {args.threads}\n")
        f.write(f"Input File: {args.input if args.input else 'None (using simulated data)'}\n\n")
        
        f.write("=== Poisson Distribution Test ===\n")
        for event_type in EventType:
            type_idx = event_type.value
            event_name = ["New order", "Modify order", "Delete order", "Trade execution"][type_idx]
            result = poisson_results[event_type]
            
            f.write(f"\n=== Event Type: {event_name} ===\n")
            f.write(f"Number of events: {sum(rate.counts[type_idx] for rate in rates)}\n")
            f.write(f"Mean rate (λ): {result['mean']:.2f}\n")
            f.write(f"Variance: {result['variance']:.2f}\n")
            f.write(f"Variance/Mean ratio: {result['variance_mean_ratio']:.2f}\n")
            f.write(f"Nonlinear index: {result['nonlinear_index']:.2f}\n")
            f.write(f"Chi-square test statistic: {result['chi_square']:.2f}\n")
            f.write(f"Critical value (95%): {result['critical_value']:.2f}\n")
            
            if result['is_poisson']:
                f.write("Conclusion: At 95% confidence level, the data conforms to the Poisson distribution assumption.\n")
            else:
                f.write("Conclusion: At 95% confidence level, the data does not conform to the Poisson distribution assumption.\n")
        
        f.write("\n=== EDM Prediction Results ===\n")
        for event_type, result in edm_results.items():
            type_idx = event_type.value
            event_name = ["New order", "Modify order", "Delete order", "Trade execution"][type_idx]
            
            f.write(f"\nEvent Type: {event_name}\n")
            f.write(f"Predicted Rate: {result['predicted_rate']:.4f}\n")
            f.write(f"Mean: {result['mean']:.4f}\n")
            f.write(f"Variance: {result['variance']:.4f}\n")
            f.write(f"Nonlinearity: {result['nonlinearity']:.4f}\n")
            f.write(f"Poisson Ratio: {result['poisson_ratio']:.4f}\n")
    
    print(f"\nAnalysis results saved to {results_file}")
    print("\n=== Analysis completed ===")


if __name__ == "__main__":
    main() 