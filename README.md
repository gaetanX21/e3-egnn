# E(3)-Equivariant Graph Neural Networks for Molecular Properties

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
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project replicates and extends findings from Hoogeboom et al. to illustrate the advantages of E(3)-equivariance in molecular property prediction tasks. Specifically, we:
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
   git clone https://github.com/your_username/E3-equivariant-GNN.git
   cd E3-equivariant-GNN
