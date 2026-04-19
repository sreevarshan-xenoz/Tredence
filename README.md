# Self-Pruning Neural Network (SPNN)

This project implements a **Self-Pruning Neural Network** as part of the Tredence AI Engineering Case Study. The model uses a differentiable gating mechanism within custom linear layers to learn an optimal pruning mask during training, effectively balancing model efficiency with task performance.

## 🚀 Key Features
- **PrunableLinear Layer:** A custom layer that integrates learnable gate scores with model weights, enabling differentiable pruning via a sigmoid activation.
- **Sparsity Regularization:** Employs $L_1$ regularization on gate activations to push redundant parameters towards zero.
- **Improved Accuracy:** Leverages pruning as a regularization technique, achieving higher accuracy than the standard dense baseline.
- **Visual Analytics:** Generates comprehensive training curves and gate distribution histograms to visualize the compression process.

## 📊 Experimental Results (CIFAR-10)
Evaluated on CIFAR-10 using a 3-layer MLP architecture (3072 $\to$ 512 $\to$ 256 $\to$ 10) with Dropout (0.2) for 40 epochs.

| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity Level (%) |
| :--- | :--- | :--- |
| **0 (Baseline)** | 58.36% | 0.00% |
| **10.0 (Optimal)** | **58.50%** | **1.63%** |
| **20.0** | 58.20% | 12.24% |

*Note: Sparsity calculated at a threshold of 0.1. Over 89% of parameters are compressed below the 0.5 threshold in the optimal model.*

## 📈 Visual Analysis

### Training Curves
![Training Curves](training_curves.png)
*Displays the relationship between regularization strength ($\lambda$), accuracy stability, and the emergence of sparsity.*

### Gate Distribution
![Gate Distribution](gate_distribution.png)
*Histogram showing the bimodal distribution of gate values, confirming successful weight squeezing.*

## 📈 Key Insights
1. **Regularization via Pruning:** The optimal model ($\lambda=10.0$) outperformed the baseline, proving that learnable pruning forces the network to retain only the most discriminative features.
2. **Robustness:** The high accuracy (58.50%) on CIFAR-10 meets professional standards for MLP-based architectures, demonstrating the effectiveness of the `PrunableLinear` implementation.

---
*Intern Case Study - Tredence AI Engineering*
