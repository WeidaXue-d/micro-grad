# micro-autograd 

A tiny, scalar-valued Autograd engine and Neural Network library built from scratch in pure Python. 

This project implements backpropagation (reverse-mode autodiff) over a dynamically built directed acyclic graph (DAG). It is designed to be a transparent, educational, and fully functional deep learning framework that strips away the complexity of modern libraries (like PyTorch) to reveal the pure mathematical and programmatic beauty of machine learning.

Inspired by Andrej Karpathy's micrograd

##  Features

- **Custom Autograd Engine (`engine.py`)**: Implements a `Value` object that wraps standard scalar values and tracks their gradient and mathematical lineage.
- **Dynamic Computation Graph**: Automatically builds a DAG of mathematical operations (`+`, `-`, `*`, `**`, `tanh`).
- **Topological Sorting**: Uses post-order traversal to guarantee the correct application of the Chain Rule during backpropagation.
- **Neural Network API (`nn.py`)**: An object-oriented API featuring `Neuron` (more components like `Layer` and `MLP` can be seamlessly integrated).

##  Quick Start: Training a Single Neuron

Here is a complete example of how to use `micro-autograd` to train a single neuron using Gradient Descent. We force a randomly initialized neuron to output exactly `1.0` when given the inputs `[2.0, 3.0]`.

```python
from micro_autograd.nn import Neuron

# 1. Define input data and target output
x = [2.0, 3.0]
y_true = 1.0

# 2. Initialize a Neuron with 2 inputs (random weights & bias)
n = Neuron(2)
print(f"Initial random prediction: {n(x).data:.4f}")

# 3. Training Loop (Gradient Descent)
learning_rate = 0.1

for step in range(60):
    # Forward pass
    y_pred = n(x)
    
    # Calculate Loss (Mean Squared Error)
    loss = (y_pred - y_true) ** 2
    
    # Zero gradients before backprop
    for p in n.w + [n.b]:
        p.grad = 0.0
        
    # Backward pass (calculate gradients via chain rule)
    loss.backward()
    
    # Update weights
    for p in n.w + [n.b]:
        p.data -= learning_rate * p.grad
        
    if step % 10 == 0 or step == 59:
        print(f"Step {step:2d} | Prediction: {y_pred.data:.4f} | Loss: {loss.data:.4f}")

print(f"\nFinal optimized weights: {[p.data for p in n.w]}")
