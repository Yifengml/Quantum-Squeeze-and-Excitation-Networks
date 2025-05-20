# Quantum Squeeze-and-Excitation ResNet18 Variants for CIFAR‑10

## Overview

This repository contains five self‑contained training scripts that extend **ResNet‑18** with a *Quantum Squeeze‑and‑Excitation (QSE)* module.  Each script experiments with a different qubit‑entanglement **topology** in the quantum linear layer that replaces the first fully‑connected stage of the classical SE block.

| Topology | Script                           | Backend                                            | Default log dir                              |
| -------- | -------------------------------- | -------------------------------------------------- | -------------------------------------------- |
| Tree     | `ResNet18_CIFAR10_QSE_Tree.py`   | Pennylane (`default.qubit`)                        | `Topolo/QSEResNet18_Tree` citeturn0file0  |
| Star     | `ResNet18_CIFAR10_QSE_Star.py`   | Pennylane (`default.qubit`)                        | `QSEResNet18_Star` citeturn0file1         |
| Mesh     | `ResNet18_CIFAR10_QSE_Mesh.py`   | Pennylane (`default.qubit`)                        | `QSEResNet18_Mesh` citeturn0file2         |
| Linear   | `ResNet18_CIFAR10_QSE_Linear.py` | Pennylane (`default.qubit`)                        | `QSEResNet18_Linear` citeturn0file3       |
| Circular | `IBM_circular.py`                | **IBM Quantum** (real hardware via Qiskit Runtime) | `IBM/QSEResNet18_Circular` citeturn0file4 |

All Pennylane scripts train on the full **CIFAR‑10** dataset, whereas the IBM demo downsamples to `20` training and `10` test images to keep real‑device runtime reasonable.

---

## 1  Prerequisites

### Core

```bash
python >=3.10
pytorch >=1.13     # + CUDA if available
torchvision >=0.14
pennylane >=0.33
numpy, matplotlib, scikit‑learn, tqdm
```

### Visualisation

```bash
tensorboard >=2.13
```

### IBM‑only script (`IBM_circular.py`)

```bash
qiskit ==1.2
qiskit-ibm-runtime
```

You will also need an **IBM Quantum API token**. Insert it either by

* setting the env‑var `QISKIT_IBM_TOKEN`, or
* editing the placeholder in the script (search for `save_account(...)`).

---

## 2  Installation

```bash
# clone or copy this directory
cd <project_root>

# create & activate a fresh environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate                           # Windows PowerShell

pip install --upgrade pip wheel

# core requirements
pip install torch torchvision pennylane matplotlib scikit-learn tensorboard tqdm

# for IBM Quantum demo
pip install "qiskit==1.2" qiskit-ibm-runtime
```

---

## 3  Running the Experiments

Run any script directly with Python; all hyper‑parameters are declared at the top of each file for quick tuning.

```bash
# example: Tree topology on a GPU machine
python ResNet18_CIFAR10_QSE_Tree.py    # trains for 100 epochs, BS=128, LR=5e‑3

# example: IBM Quantum circular topology (real hardware)
python IBM_circular.py                 # trains on a 30‑image toy set
```

> **Tip :** Real‑device jobs may queue for several minutes—consider testing locally with the Pennylane scripts first.

---

## 4  Monitoring with TensorBoard

Each script records training and test metrics (loss, accuracy, error‑rate, ROC‑AUC, F1, precision, recall) to the directory shown in the table above.  To visualise:

```bash
tensorboard --logdir .                # or point to an individual sub‑folder
```

---

## 5  Customising the Models

* **Epochs, batch size, learning‑rate** – change `num_epochs`, `batch_size`, `learning_rate` near the top of the script.
* **Entanglement pattern** – modify the `quantum_circuit` method inside `QuantumLinearLayer`.
* **Dataset subset** (IBM demo) – adjust `num_train`, `num_test`.
* **Logging path** – change the `SummaryWriter("…")` destination.

---

## 6  Expected Runtime

| Script                   | Approx. epochs / min (A100 GPU) | Notes                                                              |
| ------------------------ | ------------------------------- | ------------------------------------------------------------------ |
| Pennylane (any topology) | \~12 min                        | Full CIFAR‑10, simulation only                                     |
| IBM Quantum (circular)   | device‑dependent                | Queue time dominates; each batch of 1 image triggers a Sampler job |

---

## 7  Citation

If you use these scripts in academic work, please cite the corresponding paper (to appear) and acknowledge the IBM Quantum platform when running on real hardware.

---

## 8  License

MIT License — see `LICENSE` file (or adapt as needed).
