# dpd-baselines
PyTorch-based Digital Predistortion (DPD) baselines for RF power amplifiers.

## Motivation
This project aims to provide:
- modular DPD building blocks,
- composable model architectures,
- reproducible baselines from classical and modern DPD literature.

## Repository structure
- src/dpd_baselines/ # core DPD library
- scripts/ # training and evaluation entrypoints
- papers/ # reproductions of published DPD models
- configs/ # experiment configurations
- tests/ # unit and smoke tests


## Core concepts
- **Blocks**: atomic DPD components (Delay, FIR, Polynomial)
- **Models**: compositions of blocks (MP, GMP, Neural DPD)
- **Search**: greedy and optimization-based structure selection

