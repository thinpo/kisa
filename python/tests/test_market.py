#!/usr/bin/env python3
"""
Unit tests for the market analysis module in K-ISA Python package.
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path to import kisa package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kisa.market import (
    EventType, OrderEvent, EventRates,
    generate_simulated_events, calculate_event_rates,
    analyze_autocorrelation, analyze_cross_correlation,
    analyze_event_rates_with_edm
)


class TestMarket(unittest.TestCase):
    """Test cases for market analysis functions."""

    def setUp(self):
        """Set up test data."""
        # Generate a small set of simulated events
        np.random.seed(42)  # For reproducibility
        self.events = generate_simulated_events(500)
        
        # Calculate event rates with a small window size
        self.window_size = 10  # seconds
        self.rates = calculate_event_rates(self.events, self.window_size)

    def test_event_type_enum(self):
        """Test EventType enumeration."""
        self.assertEqual(EventType.NEW.value, 0)
        self.assertEqual(EventType.MODIFY.value, 1)
        self.assertEqual(EventType.DELETE.value, 2)
        self.assertEqual(EventType.EXECUTE.value, 3)
        
        # Test total number of event types
        self.assertEqual(len(EventType), 4)

    def test_order_event_class(self):
        """Test OrderEvent data class."""
        event = OrderEvent(
            timestamp=1000,
            type=EventType.NEW,
            order_id=12345,
            price=100.50,
            quantity=10
        )
        
        self.assertEqual(event.timestamp, 1000)
        self.assertEqual(event.type, EventType.NEW)
        self.assertEqual(event.order_id, 12345)
        self.assertEqual(event.price, 100.50)
        self.assertEqual(event.quantity, 10)

    def test_event_rates_class(self):
        """Test EventRates data class."""
        rates = EventRates(
            window_start=1000,
            window_end=1010,
            counts=[5, 3, 2, 1],
            rates=[0.5, 0.3, 0.2, 0.1]
        )
        
        self.assertEqual(rates.window_start, 1000)
        self.assertEqual(rates.window_end, 1010)
        self.assertEqual(rates.counts, [5, 3, 2, 1])
        self.assertEqual(rates.rates, [0.5, 0.3, 0.2, 0.1])

    def test_generate_simulated_events(self):
        """Test generation of simulated events."""
        num_events = 100
        events = generate_simulated_events(num_events)
        
        # Check number of events
        self.assertEqual(len(events), num_events)
        
        # Check event properties
        for event in events:
            self.assertIsInstance(event, OrderEvent)
            self.assertIsInstance(event.timestamp, int)
            self.assertIsInstance(event.type, EventType)
            self.assertIsInstance(event.order_id, int)
            self.assertIsInstance(event.price, float)
            self.assertIsInstance(event.quantity, int)
            
            # Check value ranges
            self.assertGreaterEqual(event.timestamp, 0)
            self.assertGreaterEqual(event.order_id, 0)
            self.assertGreaterEqual(event.price, 0)
            self.assertGreaterEqual(event.quantity, 0)
        
        # Test with seed for reproducibility
        events1 = generate_simulated_events(10, seed=123)
        events2 = generate_simulated_events(10, seed=123)
        
        # Events should be identical with same seed
        for e1, e2 in zip(events1, events2):
            self.assertEqual(e1.timestamp, e2.timestamp)
            self.assertEqual(e1.type, e2.type)
            self.assertEqual(e1.order_id, e2.order_id)
            self.assertEqual(e1.price, e2.price)
            self.assertEqual(e1.quantity, e2.quantity)

    def test_calculate_event_rates(self):
        """Test calculation of event rates."""
        # Test with our simulated events
        rates = self.rates
        
        # Check that we have rates
        self.assertGreater(len(rates), 0)
        
        # Check rate properties
        for rate in rates:
            self.assertIsInstance(rate, EventRates)
            
            # Check window properties
            self.assertLessEqual(rate.window_start, rate.window_end)
            # Our implementation uses a fixed window size of 10 seconds (10000 ms)
            self.assertEqual(rate.window_end - rate.window_start, 10000)
            
            # Check counts and rates
            self.assertEqual(len(rate.counts), 4)  # One for each event type
            self.assertEqual(len(rate.rates), 4)  # One for each event type
            
            # Rates should be counts divided by window size
            for count, r in zip(rate.counts, rate.rates):
                self.assertAlmostEqual(r, count / self.window_size)
        
        # Test with empty events list
        empty_rates = calculate_event_rates([], self.window_size)
        self.assertEqual(len(empty_rates), 0)
        
        # Test with invalid window size
        with self.assertRaises(ValueError):
            calculate_event_rates(self.events, 0)

    def test_analyze_autocorrelation(self):
        """Test analysis of autocorrelation."""
        # Test with our calculated rates
        autocorr = analyze_autocorrelation(self.rates)
        
        # Check that we have results for each event type
        for event_type in EventType:
            self.assertIn(event_type, autocorr)
            
            # Check autocorrelation properties
            ac = autocorr[event_type]
            self.assertIsInstance(ac, np.ndarray)
            
            # Autocorrelation should be between -1 and 1
            self.assertTrue(np.all(ac >= -1.0))
            self.assertTrue(np.all(ac <= 1.0))
        
        # Test with max lag
        max_lag = 5
        autocorr_limited = analyze_autocorrelation(self.rates, max_lag)
        
        # Check that length is limited by max_lag
        for event_type in EventType:
            self.assertEqual(len(autocorr_limited[event_type]), max_lag)
        
        # Test with empty rates list
        empty_autocorr = analyze_autocorrelation([])
        self.assertEqual(len(empty_autocorr), 0)

    def test_analyze_cross_correlation(self):
        """Test analysis of cross-correlation."""
        # Test with our calculated rates
        corr_matrix = analyze_cross_correlation(self.rates)
        
        # Check matrix shape
        self.assertEqual(corr_matrix.shape, (4, 4))  # 4x4 for each event type
        
        # Check correlation properties
        self.assertTrue(np.all(corr_matrix >= -1.0))
        self.assertTrue(np.all(corr_matrix <= 1.0))
        
        # Diagonal should be 1.0 (correlation with self)
        for i in range(4):
            self.assertAlmostEqual(corr_matrix[i, i], 1.0)
        
        # Matrix should be symmetric
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(corr_matrix[i, j], corr_matrix[j, i])
        
        # Test with empty rates list
        with self.assertRaises(ValueError):
            analyze_cross_correlation([])

    def test_analyze_event_rates_with_edm(self):
        """Test EDM analysis of event rates."""
        # Test with our calculated rates
        embedding_dim = 3
        time_delay = 1
        prediction_steps = 1
        num_neighbors = 3
        
        edm_results = analyze_event_rates_with_edm(
            self.rates, embedding_dim, time_delay, prediction_steps, num_neighbors
        )
        
        # Check that we have results for each event type
        for event_type in EventType:
            self.assertIn(event_type, edm_results)
            
            # Check result properties
            result = edm_results[event_type]
            self.assertIn('predicted_rate', result)
            self.assertIn('mean', result)
            self.assertIn('variance', result)
            self.assertIn('nonlinearity', result)
            self.assertIn('poisson_ratio', result)
            
            # Check value types
            self.assertIsInstance(result['predicted_rate'], float)
            self.assertIsInstance(result['mean'], float)
            self.assertIsInstance(result['variance'], float)
            self.assertIsInstance(result['nonlinearity'], float)
            self.assertIsInstance(result['poisson_ratio'], float)
        
        # Test with invalid parameters
        with self.assertRaises(ValueError):
            analyze_event_rates_with_edm(self.rates, 0, time_delay, prediction_steps, num_neighbors)
        
        with self.assertRaises(ValueError):
            analyze_event_rates_with_edm(self.rates, embedding_dim, 0, prediction_steps, num_neighbors)
        
        # Test with empty rates list
        empty_results = analyze_event_rates_with_edm([], embedding_dim, time_delay, prediction_steps, num_neighbors)
        self.assertEqual(len(empty_results), 0)


if __name__ == "__main__":
    unittest.main() 