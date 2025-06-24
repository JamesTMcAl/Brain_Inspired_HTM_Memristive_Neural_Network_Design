# CSC3002 HTM-Memristor Project

This project implements a **Hierarchical Temporal Memory (HTM)** spatial pooler using a **memristor-inspired hardware model**, optimized for sparse distributed representations (SDRs). The system supports adaptive inhibition (kWTA), entropy-guided thresholding, write endurance constraints, and dynamic sparsity control using a PI controller.

It was developed as part of the final year project for CSC3002 at Queen’s University Belfast.

---

## Project Structure

- `compute_overlap.m` – Core overlap computation (CPU & GPU supported)
- `apply_kwta.m` – Adaptive k-Winner-Take-All inhibition logic
- `adjust_density.m` – Dynamic density via PI or moving-average control
- `adjust_synaptic_factors.m` – Synaptic update rules including endurance logic
- `deviceModel.m` – Models memristive endurance and write limits
- `evaluate_accuracy.m` – Measures performance (accuracy, sparsity, entropy)
- `metrics_analysis.m` – Dashboard to visualize training metrics
- `TestHTMFunctions.m` – Unit test suite for all critical functions
- `get_mnist_subset.m` – MNIST loader with subset support
- `run_all.m` – End-to-end training and evaluation script

---

## Key Features

- **Memristor-based modeling**: Includes endurance-aware learning updates and write-cycle tracking.
- **Adaptive inhibition**: Dynamic kWTA radius and activity-based boosting.
- **Entropy-aware thresholding**: Helps avoid collapsed SDRs.
- **Energy-efficient design**: Optimized for sparse and low-write configurations.
- **Hybrid classifiers**: Uses both k-NN and SVM fallback logic for robustness.
- **GPU Acceleration**: Optional GPU support for large batch training.

---

##  Usage

Ensure MATLAB is installed with Parallel Computing Toolbox (for GPU support).



