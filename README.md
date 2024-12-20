# E(3)-Equivariant Graph Neural Networks for Molecular Properties

![Diffusion](report/figures/overview_diffusion.png)

This repository contains the implementation and experiments for exploring E(3)-equivariance in Graph Neural Networks (GNNs). It demonstrates how incorporating geometric symmetry in GNNs can improve their ability to model molecular properties using the QM9 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Experiments](#experiments)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Introduction

This project illustrates findings from Hoogeboom et al. in order to show the advantages of E(3)-equivariance in molecular property prediction tasks. Specifically, we:

- Use the QM9 dataset for a regression task predicting the molecular dipole moment.
- Compare standard GNNs with E(3)-equivariant GNNs.
- Demonstrate the benefits of leveraging geometric structure in molecular data.

## Dataset

The QM9 dataset contains small molecules with up to 29 atoms and features:

- **Node features**: Atom types, atomic number, aromaticity, hybridization, and more.
- **Edge features**: Bond types (single, double, triple, aromatic).
- **Target properties**: 19 physical properties, with our focus on the dipole moment.

For simplicity:

- We filter molecules with 12 atoms or fewer, resulting in 4005 molecules.
- The dataset is split into 80% train, 10% validation, and 10% test.

## Models

We evaluate the following models:

- **LinReg**: A simple baseline that ignores graph structure.
- **MPNN**: A standard Message Passing Neural Network.
- **EGNN**: An E(3)-equivariant GNN that incorporates spatial symmetry.
- **EGNN_edge**: A variant of EGNN that uses edge features.

All models are permutation invariant, and the EGNN-based models are E(3)-equivariant.

## Experiments

Our pipeline consists of:

1. **Preprocessing**: Filtering and normalizing the QM9 dataset.
2. **Training**:
   - Models are trained for 500 epochs using the Adam optimizer and MSE loss.
   - Learning rate scheduling with ReduceLROnPlateau.
3. **Validation**: Evaluate models on unseen data to compare their generalization.

**Frameworks**:

- PyTorch Geometric for graph data.
- WandB for experiment tracking.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/EGGN_Classifier.git
   cd EGGN_Classifier

   ```

2. Set up the Environment

   ```bash
   conda create -n egnn-env python=3.8
   conda activate egnn-env
   pip install -r requirements.txt
   ```

## Usage

1. Train Models
   ```bash
   python main.py --model EGNN --epochs 500
   ```

Options for `--model`

- `LinReg`
- `MPNN`
- `EGNN`
- `EGNN_edge`

2. Evaluate Models

   ```bash
   python eval.py --model EGNN

   ```

3. Visualization

Results such as training/validation loss and learning rate are logged in WandB.

## Results

### Training Loss

- **LinReg**: Performs poorly, while GNNs effectively minimize loss.

### Validation Loss

- **EGNN-based models**: Outperform MPNNs, with EGNN_edge achieving the best generalization.

### Key Observations

- **E(3)-equivariance**: Leads to significant performance gains.
- Incorporating **edge features** improves model expressivity.

Figures and detailed discussions are in the report.

## Contributing

Contributions are welcome! Especially implementing an SE(3) equivariant model as it should produce better results. Please:

1. Fork the repository.
2. Create a new branch (`feature/new-feature`).
3. Submit a pull request.

## Acknowledgements

- Hoogeboom et al. for their foundational work on E(3)-equivariant GNNs.
- PyTorch Geometric for their framework for graph data.
- The QM9 dataset creators.
