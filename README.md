# Self-Pruning Neural Network (SPNN) v2 - CNN Edition

This project implements an advanced **Self-Pruning Convolutional Neural Network** using a differentiable structured gating mechanism. The model learns to prune entire **filters** and **neurons** during training, optimizing both model size and computational efficiency.

## 🚀 Key Features (Upgraded)
- **CNN Architecture:** Transitioned from MLP to CNN for superior image feature extraction.
- **Structured Pruning:** Prunes at the filter (Conv2d) and neuron (Linear) level rather than individual weights, enabling real-world hardware acceleration.
- **Hard-Sigmoid Gating with STE:** Uses a Straight-Through Estimator (STE) to allow the model to make hard 0/1 pruning decisions during the forward pass while remaining differentiable in the backward pass.
- **Lambda Warm-up:** Gradually introduces the pruning penalty to allow the network to learn robust features before compression begins.

## 📊 Experimental Results (CIFAR-10)
Tested on the CIFAR-10 dataset for **15 epochs** with the upgraded v2 architecture.

| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity Level (%) | Notes |
| :--- | :--- | :--- | :--- |
| **0 (Baseline)** | 72.85% | 0.00% | High-performance baseline. |
| **1.0e-03** | **72.93%** | **17.96%** | **Optimal:** Higher accuracy and 18% smaller. |
| **1.0e-02** | 63.46% | **37.96%** | High compression with moderate accuracy loss. |

### Post-Pruning Validation
The best model ($\lambda = 1 \times 10^{-3}$) was subjected to **Hard Pruning**:
- **Accuracy after Hard Pruning:** **72.93%** (Zero performance loss)

## 📈 Visual Analysis

### Training Curves
![Training Curves](training_curves.png)
*The plots show accuracy stability and the emergence of sparsity as the lambda warm-up concludes (around epoch 5).*

### Gate Distribution
![Gate Distribution](gate_distribution.png)
*The bimodal distribution of gate scores demonstrates clear separation between active (positive) and pruned (negative) components.*

## 📈 Key Insights
1.  **Efficiency through Structure:** By pruning entire filters, we reduce the number of feature maps, which directly reduces the number of operations (FLOPs) required for inference.
2.  **Regularization via Pruning:** The $10^{-3}$ lambda run outperformed the baseline, suggesting that pruning acts as a powerful regularizer, forcing the model to focus on the most informative features.
3.  **Stability:** The Hard-Sigmoid STE ensures that components are either "on" or "off," leading to extremely stable performance after hard-pruning.

---
*Intern Case Study - Tredence AI Engineering*
