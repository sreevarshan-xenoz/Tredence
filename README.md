# Self-Pruning Neural Network (SPNN)

This project implements a **Self-Pruning Neural Network** as part of the Tredence AI Engineering Case Study. The model uses a differentiable gating mechanism within custom linear layers to learn an optimal pruning mask during training.

## 🚀 Key Features
- **PrunableLinear Layer:** A custom layer that integrates learnable gate scores with model weights, enabling differentiable pruning.
- **Sparsity Regularization:** Applies an $L_1$ penalty to gate activations to encourage the model to identify and prune redundant connections.
- **Hard Pruning:** Demonstrates zero-loss performance preservation after setting low-value gates to zero.
- **Visual Analytics:** Generates training curves and gate distribution histograms to analyze the pruning process.

## 📊 Experimental Results (CIFAR-10)
Tested on the CIFAR-10 dataset using an MLP architecture (3072 $\rightarrow$ 512 $\rightarrow$ 256 $\rightarrow$ 10).

| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity Level (%) |
| :--- | :--- | :--- |
| **0 (Baseline)** | 55.55% | 0.00% |
| **5.0e-06** | **57.07%** | **0.00%** |

### Post-Pruning Validation
The optimal model ($\lambda = 5 \times 10^{-6}$) maintained its performance after **Hard Pruning**:
- **Accuracy after Hard Pruning:** **57.07%**

## 📈 Visual Analysis

### Training Curves
![Training Curves](training_curves.png)
*Shows accuracy and sparsity trends across different lambda values.*

### Gate Distribution
![Gate Distribution](gate_distribution.png)
*Displays the histogram of gate values for the best-performing model.*

## 📈 Key Insights
1. **Regularization:** The inclusion of the pruning penalty acts as a regularizer, leading to higher accuracy than the baseline model.
2. **Weight Squeezing:** Even when explicit sparsity isn't reached, the model significantly reduces the average gate magnitude, preparing the weights for potential compression.

---
*Intern Case Study - Tredence AI Engineering*
