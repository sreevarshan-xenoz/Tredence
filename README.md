# Self-Pruning Neural Network (SPNN)

This project implements a **Self-Pruning Neural Network** using a differentiable gating mechanism. The model learns which connections are necessary for performance and automatically "prunes" redundant weights by driving their associated gate scores toward zero through $L_1$ regularization.

## 🚀 Key Features
- **Differentiable Gating:** Each weight has a learnable gate score, mapped through a sigmoid function.
- **Sparsity-Inducing Loss:** A penalty term ($\lambda$) is added to the loss function to encourage sparsity.
- **Hard Pruning:** Connections with gate values below a threshold ($1 \times 10^{-2}$) can be permanently zeroed without retraining.
- **Automated Benchmarking:** Compares multiple $\lambda$ values to find the optimal trade-off between model size and accuracy.

## 📊 Experimental Results (CIFAR-10)
Tested on the CIFAR-10 dataset for **30 epochs** per configuration.

| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity Level (%) | Notes |
| :--- | :--- | :--- | :--- |
| **0 (Baseline)** | 54.77% | 0.00% | Full dense model performance. |
| **1.0e-06** | 55.26% | 0.27% | Slight regularization boost. |
| **5.0e-06** | **55.50%** | 0.73% | **Best accuracy** with mild pruning. |
| **1.0e-05** | 55.30% | **2.08%** | Highest compression while maintaining accuracy. |

### Post-Pruning Validation
The best model ($\lambda = 5 \times 10^{-6}$) was subjected to **Hard Pruning** (zeroing all weights where $gate < 0.01$):
- **Accuracy after Hard Pruning:** **55.50%** (Zero performance loss)

## 🛠 Project Structure
- `self_pruning_neural_network.py`: Main implementation (Model, Training, Evaluation).
- `training_curves.png`: Visualizes accuracy and sparsity trends over time.
- `gate_distribution.png`: Histogram showing how many gates were driven toward zero.
- `results.json`: Raw data for all experiments.

## 📈 Analysis
1. **Regularization Effect:** Small values of $\lambda$ (like $1 \times 10^{-6}$) actually improve accuracy over the baseline by acting as a form of "Dropout-like" regularization.
2. **Convergence:** Sparsity typically begins to emerge after epoch 20 as the $L_1$ penalty overcomes the initial weight initialization noise.
3. **Thresholding:** The gate distribution shows a bimodal trend, where gates either stay near 1.0 or migrate toward 0.0, making the pruning decision stable.

---
*Intern Case Study - Tredence AI Engineering*
