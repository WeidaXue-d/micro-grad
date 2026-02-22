# Micro-Autograd: AI From First Principles 

## Motivation
As a first-year Software Engineering student at USYD, I believe that truly mastering AI requires understanding what happens "under the hood." Instead of just importing PyTorch, I built this **Micro-Autograd Engine** from scratch to internalize the mathematical foundations of backpropagation and computational graphs.

## Features
- **Scalar-valued Autograd:** Implements backpropagation over a dynamically built computational graph.
- **Operator Overloading:** Clean Pythonic syntax for building expressions (e.g., `a + b`, `a * b`).
- **Zero Dependencies:** Built using pure Python to ensure a deep understanding of the logic.

## Core Concept: The Computational Graph
Every operation in this library creates a new `Value` object that remembers its "parents" and the operation that created it. 

### How it works:
1. **Forward Pass:** Compute the numerical result of an expression.
2. **Backward Pass:** Starting from the output, apply the **Chain Rule** recursively to compute the gradient for every internal variable.

