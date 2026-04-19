# self_pruning_neural_network.py
# Tredence AI Engineering Intern Case Study – The Self-Pruning Neural Network
# Author: Sree Varshan V
# April 2026

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from pathlib import Path
import json
from typing import Dict, List, Tuple

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HardSigmoidSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: gradient of hard sigmoid approximated by 
        # a windowed gradient (like in binary networks)
        return F.hardtanh(grad_output)

def hard_sigmoid_ste(x):
    return HardSigmoidSTE.apply(x)

class PrunableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        # One gate per FILTER (Structured Pruning)
        self.gate_scores = nn.Parameter(torch.empty(out_channels, 1, 1, 1))
        
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 0.5) # Start half-open

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use hard sigmoid for clear 0/1 decisions
        gates = hard_sigmoid_ste(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.conv2d(x, pruned_weights, self.bias, padding=1)

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # One gate per NEURON (Structured Pruning)
        self.gate_scores = nn.Parameter(torch.empty(out_features, 1))
        
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        bound = 1 / np.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = hard_sigmoid_ste(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN Architecture
        self.conv1 = PrunableConv2d(3, 32, 3)
        self.conv2 = PrunableConv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = PrunableConv2d(64, 128, 3)
        
        self.flatten = nn.Flatten()
        # CIFAR-10: 32x32 -> (after 3 convs with 1 pool) -> 16x16
        # Using 3 convs with 1 pool at the end for simplicity
        self.fc1 = PrunableLinear(128 * 16 * 16, 256)
        self.fc2 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_sparsity_loss(self) -> torch.Tensor:
        sparsity = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                # L1 on gate scores to push them to 0
                sparsity += torch.norm(module.gate_scores, 1)
        return sparsity

def compute_sparsity(model: nn.Module, threshold: float = 0.5) -> float:
    total = 0
    pruned = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                gates = (module.gate_scores > 0).float()
                num = gates.numel()
                total += num
                pruned += (gates == 0).sum().item()
    return (pruned / total * 100) if total > 0 else 0.0

def evaluate(model: nn.Module, testloader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def train_one_lambda(target_lambda: float, epochs: int, trainloader: DataLoader,
                     testloader: DataLoader) -> Tuple[nn.Module, float, float, List[Dict]]:
    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    warmup_epochs = 5
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Lambda Warm-up: Linear increase from 0 to target_lambda
        current_lambda = target_lambda * min(1.0, (epoch / warmup_epochs)) if epoch > 2 else 0.0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            class_loss = criterion(outputs, labels)
            sparsity_loss = model.get_sparsity_loss()
            loss = class_loss + current_lambda * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        test_acc = evaluate(model, testloader)
        sparsity_pct = compute_sparsity(model)
        avg_loss = total_loss / len(trainloader)
        
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "test_acc": test_acc,
            "sparsity": sparsity_pct
        })
        
        print(f"Epoch {epoch+1:2d}/{epochs} | λ: {current_lambda:.2e} | Loss: {avg_loss:.3f} | "
              f"Acc: {test_acc:.2f}% | Sparsity: {sparsity_pct:.2f}%")
    
    final_acc = history[-1]["test_acc"]
    final_sparsity = history[-1]["sparsity"]
    print(f"Finished λ={target_lambda} → Acc: {final_acc:.2f}% | Sparsity: {final_sparsity:.2f}%\n")
    
    return model, final_acc, final_sparsity, history

def plot_training_curves(histories: Dict[float, List[Dict]], save_path: str = "training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for lam, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        accs = [h["test_acc"] for h in hist]
        spars = [h["sparsity"] for h in hist]
        
        label = f"λ={lam}" if lam > 0 else "Baseline (λ=0)"
        ax1.plot(epochs, accs, marker='o', label=label)
        ax2.plot(epochs, spars, marker='o', label=label)
    
    ax1.set_title("Test Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Sparsity Level over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Sparsity (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved → {save_path}")

def plot_gate_distribution(model: nn.Module, save_path: str = "gate_distribution.png"):
    all_scores = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                scores = module.gate_scores.cpu().numpy().flatten()
                all_scores.append(scores)
    all_scores = np.concatenate(all_scores)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, color='#1f77b4', edgecolor='black', alpha=0.75)
    plt.title("Final Gate Score Distribution (Best Model)")
    plt.xlabel("Gate Score ( >0 means Active, <0 means Pruned)")
    plt.ylabel("Count")
    plt.axvline(0, color='red', linestyle='--', label='Pruning threshold (0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gate distribution plot saved → {save_path}")
    
    print(f"Mean score: {np.mean(all_scores):.4f} | % pruned: {(all_scores <= 0).mean()*100:.2f}%")

if __name__ == "__main__":
    set_seed(42)
    
    parser = argparse.ArgumentParser(description="Tredence Self-Pruning NN Case Study")
    parser.add_argument("--lambdas", type=str, default="0.0,1e-6,5e-6,1e-5",
                        help="Comma-separated λ values (include 0 for baseline)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs per run")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    
    lambda_list = [float(x.strip()) for x in args.lambdas.split(",")]
    
    print("Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    results = []
    histories: Dict[float, List[Dict]] = {}
    best_model = None
    best_acc = -1.0
    best_lambda = None
    
    print(f"Starting training for λ values: {lambda_list}\n")
    
    for lam in lambda_list:
        model, acc, sparsity, history = train_one_lambda(
            lam, args.epochs, trainloader, testloader
        )
        histories[lam] = history
        results.append({"lambda": lam, "test_acc": acc, "sparsity": sparsity})
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_lambda = lam
    
    plot_training_curves(histories)
    plot_gate_distribution(best_model)
    
    print("\n" + "="*70)
    print("FINAL RESULTS – COPY INTO report.md")
    print("="*70)
    print("| Lambda     | Test Accuracy (%) | Sparsity Level (%) |")
    print("|------------|-------------------|--------------------|")
    for r in results:
        lam_str = "0 (Baseline)" if r["lambda"] == 0 else f"{r['lambda']:.1e}"
        print(f"| {lam_str:10} | {r['test_acc']:.2f}             | {r['sparsity']:.2f}              |")
    
    # Hard-pruning demo
    print("\nHard-pruning the best model...")
    with torch.no_grad():
        for module in best_model.modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                mask = (module.gate_scores > 0).float()
                module.weight.data *= mask
    
    hard_acc = evaluate(best_model, testloader)
    print(f"Accuracy after hard pruning: {hard_acc:.2f}%")
    
    # Save results
    output = {
        "results": results,
        "best_lambda": float(best_lambda),
        "best_test_acc": best_acc,
        "final_sparsity": compute_sparsity(best_model)
    }
    Path("results.json").write_text(json.dumps(output, indent=2))
    print("\nAll done! Files created: training_curves.png, gate_distribution.png, results.json")