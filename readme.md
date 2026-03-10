Hidden Symmetry Discovery using Deep Learning
GSoC 2026 – ML4SCI Evaluation Tasks

This repository contains my implementation of the evaluation tasks for the ML4SCI GSoC 2026 project: "Discovery of Hidden Symmetries and Conservation Laws".

The goal of these tasks is to explore how deep learning models can discover symmetry transformations directly from data. Inspired by recent work in symmetry-aware machine learning and latent symmetry discovery, the tasks investigate how neural networks can learn transformations that preserve certain invariants.

The experiments are conducted on rotated MNIST (digits 1 and 2) to study rotational symmetry in a controlled setting.

Project Overview

Many physical systems contain symmetries and conservation laws that govern their behavior. However, when data is represented in abstract feature spaces or neural network latent spaces, these symmetries may become difficult to express analytically.

This project explores methods for:

Learning latent representations of data

Discovering symmetry transformations in latent space

Identifying transformations that preserve an invariant quantity

Constructing symmetry-invariant neural networks

The repository implements four tasks that progressively build toward discovering and utilizing hidden symmetries.

Tasks Implemented
Task 1 — Latent Representation Learning

The first step is to learn a latent representation of the dataset using a Variational Autoencoder (VAE).

Dataset preparation:

MNIST digits 1 and 2

Each image is rotated by 30° increments

Total rotations per image: 12

A VAE is trained to encode images into a latent space:
z=f(x)
The objective is to obtain a structured latent space where rotations correspond to smooth transformations.
Outputs:

Reconstruction quality analysis

Latent space visualization using PCA

Latent trajectories under rotation

This step verifies whether the latent space captures the rotational structure of the dataset.

Task 2 — Supervised Symmetry Discovery

In this task, the goal is to learn a model that maps one latent vector to its rotated latent counterpart.

Using the latent vectors from Task 1:

z(t+1)=T(z(t))

where 
𝑇
T is a neural network representing a rotation transformation.

The transformation model is trained to predict the next rotation in the sequence.

Key components:

Latent transformation network

Cycle consistency constraint

Iterative transformation analysis

Outputs:

Latent rotation trajectories

Reconstruction of rotated images

Vector field visualization of transformations

This demonstrates how symmetry transformations can be learned directly in latent space.

Task 3 — Unsupervised Symmetry Discovery

This task implements a method inspired by Oracle-Preserving Latent Flows.

Steps:

Train a classifier on latent representations (oracle model)
y=C(z)

Train a generator network that produces transformation directions:

z′=z+ϵG(z)

Enforce oracle invariance
C(z)≈C(z′)

Loss function:

L=Linv+λLnorm
The generator learns directions in latent space that preserve the classifier output.

Outputs:

Symmetry generator vector fields

Latent symmetry flows

Cycle consistency verification

Multi-symmetry generator discovery

This allows discovery of continuous symmetry transformations without explicit supervision.

Task 4 — Rotation Invariant Network (Bonus Task)

Using the discovered symmetry generators, a rotation-invariant classifier is constructed.

For a latent vector $z$, multiple symmetry transformations are generated:

$z_t = z + t\epsilon G(z)$

Predictions are aggregated across these transformations:

$$
y = \frac{1}{k}\sum_{t=1}^{k} C(z_t)
$$

This ensures that predictions remain stable under symmetry operations.

Evaluation compares:

Baseline classifier

Symmetry-invariant classifier

The invariant model demonstrates improved stability under rotations.

Visualizations Included

The repository contains multiple visual diagnostics:

Latent space PCA projections

Symmetry vector fields

Latent trajectories

Decoded symmetry flows

Rotation-invariant prediction comparison

These help verify that the learned transformations correspond to actual dataset symmetries.

Technologies Used

Python

PyTorch

NumPy

Scikit-learn

Matplotlib

Torchvision