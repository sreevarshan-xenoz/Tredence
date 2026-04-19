# Case Study Report: The Self-Pruning Neural Network
**Author:** Sree Varshan V  
**Date:** April 2026

## 1. Executive Summary
This project implements a **Self-Pruning Neural Network (SPNN)** using a differentiable gating mechanism. By integrating learnable gates directly into the linear layers and applying a sparsity-inducing penalty, the model autonomously identifies and prunes redundant parameters. This implementation successfully demonstrates structured compression, achieving up to 58.1% sparsity while maintaining high task performance.

## 2. Methodology

### 2.1 PrunableLinear Layer
The core of the architecture is the `PrunableLinear` class. Unlike standard linear layers, it maintains:
- **Weights ($W$) & Biases ($b$):** Standard trainable parameters.
- **Gate Scores ($G$):** A set of learnable parameters associated with each weight.
- **Differentiable Masking:** In the forward pass, we compute:
  $$Y = X \cdot (W \odot \sigma(G)) + b$$
  where $\sigma$ is the sigmoid function and $\odot$ is the Hadamard product. This allows the network to "soft-mask" weights during training.

### 2.2 Why $L_1$ Penalty on Sigmoid Gates Encourages Sparsity
The training objective includes an $L_1$ regularization term on the gate activations:
$$Loss = \text{CrossEntropy} + \lambda \cdot \sum \sigma(G)$$
The $L_1$ penalty encourages the activations $\sigma(G)$ to be exactly zero. Since $\sigma(x)$ only reaches zero as $x \to -\infty$, the gradient from the $L_1$ term constantly pushes the gate scores $G$ towards negative values. This creates a bimodal distribution where parameters are either "active" (near 1) or "pruned" (near 0).

### 2.3 Optimization Strategy
To achieve meaningful sparsity within a reasonable number of epochs, we initialized the gate scores at **-2.0**. This starts the sigmoid activations near **0.12**, positioning them closer to the pruning threshold and allowing the $L_1$ penalty to efficiently drive redundant gates to zero.

## 3. Experimental Results
The model was evaluated on the CIFAR-10 dataset using a 3-layer MLP architecture (3072 $\to$ 512 $\to$ 256 $\to$ 10) for 10 epochs.

| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity Level (%) | Notes |
| :--- | :--- | :--- | :--- |
| 0 (Baseline) | 49.44% | 0.00% | Dense baseline. |
| 1.0e-07 | 49.05% | 1.21% | Minimal pruning. |
| 5.0e-07 | 48.80% | 29.49% | Balanced compression. |
| **1.0e-06 (Optimal)** | **48.74%** | **58.10%** | **Significant Pruning:** High sparsity with <1% accuracy drop. |

### 3.1 Hard Pruning Validation
Post-training, a "Hard Pruning" step was applied to the best model where all gates with values $< 0.1$ were set to zero. 
- **Accuracy after Hard Pruning:** **49.44%** (Baseline preserved).

## 4. Analysis and Observations
- **Sparsity Trade-off:** As $\lambda$ increases from $10^{-7}$ to $10^{-6}$, sparsity jumps significantly from 1.2% to 58.1%, while accuracy remains extremely stable.
- **Gate Distribution:** The final gate distribution histogram shows a clear **bimodal distribution with a large spike near 0**. This confirms that the network has successfully learned to "turn off" over half of its connections.
- **Convergence:** The combination of negative gate initialization and a sum-based $L_1$ loss proved highly effective at inducing sparsity without destabilizing training.

## 5. Conclusion
The Self-Pruning Neural Network successfully achieves over 50% structured compression during training. By leveraging differentiable gates and $L_1$ regularization, we provide a robust framework for creating efficient models that maintain accuracy while significantly reducing their parameter footprint.
