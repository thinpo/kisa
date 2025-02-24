#!/usr/bin/env python3
"""
Simplified LLM Calculation Example using K-ISA Python package.

This example demonstrates how to use the K-ISA Python package to implement
a simplified version of a transformer-based language model forward propagation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

# Add parent directory to path to import kisa package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import kisa


# Constants
EMBEDDING_DIM = 8
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
POSITION_SCALE = 100
EDM_EMBEDDING_DIM = 3
EDM_TIME_DELAY = 1
EDM_NUM_NEIGHBORS = 3


def print_vector(name: str, vector: np.ndarray) -> None:
    """Print a vector with its name."""
    print(f"{name}: {vector.tolist()}")


def init_vector(values: List[int]) -> np.ndarray:
    """Initialize a vector with the given values."""
    return np.array(values, dtype=np.int32)


def relu_activation(vector: np.ndarray) -> np.ndarray:
    """Apply ReLU activation function."""
    return np.maximum(vector, 0)


def layer_normalization(vector: np.ndarray, scale: int = 1000) -> np.ndarray:
    """Apply layer normalization."""
    mean = np.mean(vector)
    std = np.std(vector)
    std = max(std, 1)  # Avoid division by zero
    
    print(f"Layer Normalization - Mean: {mean:.2f}, Standard Deviation: {std:.2f}")
    
    # Normalize and scale
    return ((vector - mean) / std * scale).astype(np.int32)


def matrix_vector_mul(matrix: List[List[int]], vector: np.ndarray) -> np.ndarray:
    """Perform matrix-vector multiplication."""
    result = np.zeros_like(vector)
    for i in range(len(matrix)):
        row = np.array(matrix[i], dtype=np.int32)
        result[i] = np.sum(row * vector)
    return result


def add_positional_encoding(vector: np.ndarray, position: int) -> np.ndarray:
    """Add positional encoding to the vector."""
    pos_encoding = np.zeros_like(vector)
    for i in range(len(vector)):
        if i % 2 == 0:
            pos_encoding[i] = int(np.sin(position / (10000 ** (i / EMBEDDING_DIM))) * POSITION_SCALE)
        else:
            pos_encoding[i] = int(np.cos(position / (10000 ** ((i - 1) / EMBEDDING_DIM))) * POSITION_SCALE)
    
    result = vector + pos_encoding
    print(f"Added positional encoding {position}: Positional Encoding Result: {result.tolist()}")
    return result


def single_head_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Compute single-head attention."""
    # Calculate attention scores
    scores = np.zeros(len(key))
    for i in range(len(key)):
        scores[i] = np.sum(query * key[i])
    
    # Scale scores
    scale_factor = np.sqrt(len(query))
    scores = scores / scale_factor
    
    # Apply softmax
    exp_scores = np.exp(scores)
    attention_weights = exp_scores / np.sum(exp_scores)
    
    # Apply attention weights to values
    result = np.zeros_like(query)
    for i in range(len(value)):
        result += attention_weights[i] * value[i]
    
    return result


def multi_head_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                        num_heads: int) -> np.ndarray:
    """Compute multi-head attention."""
    head_size = len(query) // num_heads
    
    # Initialize result
    result = np.zeros_like(query)
    
    # Process each attention head
    for h in range(num_heads):
        print(f"Processing attention head {h+1}")
        
        # Extract head-specific parts of query, key, value
        q_head = query[h*head_size:(h+1)*head_size]
        k_head = key[h*head_size:(h+1)*head_size]
        v_head = value[h*head_size:(h+1)*head_size]
        
        # Compute attention for this head
        head_result = single_head_attention(q_head, np.array([k_head]), np.array([v_head]))
        
        # Add to result
        result[h*head_size:(h+1)*head_size] = head_result
    
    print("Multi-Head Attention completed")
    return result


def frequency_domain_convolution(vector: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform convolution in frequency domain."""
    # Convert to frequency domain
    vector_fft = kisa.vector_fft(vector)
    kernel_fft = kisa.vector_fft(kernel)
    
    # Multiply in frequency domain
    result_fft = kisa.vector_mul(vector_fft, kernel_fft)
    
    # Convert back to time domain
    result = kisa.vector_ifft(result_fft)
    
    return result


def extract_frequency_features(vector: np.ndarray) -> np.ndarray:
    """Extract and enhance frequency features."""
    # Convert to frequency domain
    vector_fft = kisa.vector_fft(vector)
    
    # Enhance high-frequency components
    enhanced_fft = vector_fft.copy()
    for i in range(len(vector_fft) // 2, len(vector_fft)):
        enhanced_fft[i] = int(vector_fft[i] * 1.5)
    
    # Convert back to time domain
    result = kisa.vector_ifft(enhanced_fft)
    
    return result


def advanced_fft_processing(vector: np.ndarray) -> np.ndarray:
    """Perform advanced FFT processing."""
    # Define a simple kernel for frequency domain convolution
    kernel = np.array([1, 2, 3, 4, 4, 3, 2, 1], dtype=np.int32)
    
    # Perform frequency domain convolution
    conv_result = frequency_domain_convolution(vector, kernel)
    
    # Extract frequency features
    freq_features = extract_frequency_features(vector)
    
    # Combine results
    result = (conv_result + freq_features) // 2
    
    print("Advanced FFT processing completed")
    return result


def apply_empirical_dynamic_modeling(vector: np.ndarray, 
                                   library: List[np.ndarray]) -> np.ndarray:
    """Apply Empirical Dynamic Modeling for prediction."""
    print("Applying Empirical Dynamic Modeling")
    
    # Create time-delay embedding
    embedded = np.zeros(EDM_EMBEDDING_DIM)
    for i in range(EDM_EMBEDDING_DIM):
        if i < len(vector):
            embedded[i] = vector[i]
    
    # Create library of embedded vectors
    library_vectors = []
    library_targets = []
    
    for i in range(len(library) - 1):
        lib_embedded = np.zeros(EDM_EMBEDDING_DIM)
        for j in range(EDM_EMBEDDING_DIM):
            if j < len(library[i]):
                lib_embedded[j] = library[i][j]
        
        library_vectors.append(lib_embedded)
        library_targets.append(library[i+1])
    
    if not library_vectors:
        print("Empty library, returning original vector")
        return vector
    
    # Find nearest neighbors
    distances = []
    for lib_vector in library_vectors:
        dist = np.sqrt(np.sum((embedded - lib_vector) ** 2))
        distances.append(dist)
    
    # Get indices of nearest neighbors
    indices = np.argsort(distances)[:EDM_NUM_NEIGHBORS]
    
    # Calculate weights (inverse distance)
    weights = []
    for idx in indices:
        weight = 1.0 / (distances[idx] + 1e-10)
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Make prediction
    prediction = np.zeros_like(vector)
    for i, idx in enumerate(indices):
        prediction += weights[i] * library_targets[idx]
    
    print("EDM prediction completed")
    return prediction


def transformer_layer(input_vector: np.ndarray, position: int, 
                    library: List[np.ndarray]) -> np.ndarray:
    """Apply a transformer layer to the input vector."""
    print(f"\n=== Layer {position + 1} ===")
    
    # Add positional encoding
    pos_encoded = add_positional_encoding(input_vector, position)
    
    # Define simple weight matrices for query, key, value
    query_weights = [[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]]
    
    key_weights = [[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]]
    
    value_weights = [[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]]
    
    # Calculate query, key, value vectors
    query = matrix_vector_mul(query_weights, pos_encoded)
    key = matrix_vector_mul(key_weights, pos_encoded)
    value = matrix_vector_mul(value_weights, pos_encoded)
    
    # Apply multi-head attention
    attention_output = multi_head_attention(query, key, value, NUM_HEADS)
    
    # Apply output projection (simplified as identity)
    projected_output = attention_output
    
    # Add residual connection
    residual_output = kisa.vector_add(projected_output, pos_encoded)
    
    # Apply layer normalization
    normalized_output = layer_normalization(residual_output)
    
    # Apply ReLU activation
    activated_output = relu_activation(normalized_output)
    
    # Apply advanced FFT processing
    fft_output = advanced_fft_processing(activated_output)
    
    # Apply EDM
    edm_output = apply_empirical_dynamic_modeling(fft_output, library)
    
    # Combine EDM result with current output
    final_output = (fft_output + edm_output) // 2
    
    print_vector("Layer output", final_output)
    return final_output


def main():
    """Main function to demonstrate LLM calculation."""
    print("=== Simplified LLM Calculation Demonstration ===\n")
    
    # Initialize input vector
    input_values = [10, 20, 30, 40, 50, 60, 70, 80]
    input_vector = init_vector(input_values)
    print_vector("Input Vector", input_vector)
    
    print("\nStarting LLM Calculation...")
    
    # Initialize library for EDM
    library = [input_vector]
    
    # Apply transformer layers
    output_vector = input_vector
    for i in range(NUM_LAYERS):
        output_vector = transformer_layer(output_vector, i, library)
        library.append(output_vector)
    
    print("\n=== Final Output ===")
    print_vector("Output Vector", output_vector)
    
    # Calculate statistics
    sum_val = np.sum(output_vector)
    max_val = np.max(output_vector)
    min_val = np.min(output_vector)
    
    print(f"Sum: {sum_val}")
    print(f"Max: {max_val}")
    print(f"Min: {min_val}")
    
    # Visualize input and output
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.bar(range(len(input_vector)), input_vector)
    plt.title("Input Vector")
    plt.xlabel("Index")
    plt.ylabel("Value")
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(output_vector)), output_vector)
    plt.title("Output Vector")
    plt.xlabel("Index")
    plt.ylabel("Value")
    
    plt.tight_layout()
    plt.savefig("llm_calc_result.png")
    print("Visualization saved to llm_calc_result.png")


if __name__ == "__main__":
    main() 