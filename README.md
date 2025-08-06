# UHSNet: Deep Learning-Based Smart Proxy Modeling for Underground Hydrogen Storage
## Overview

UHSNet is a deep learning-based smart proxy model designed for efficient spatio-temporal hydrogen saturation estimation in underground hydrogen storage (UHS) systems within saline aquifers. This repository contains the implementation of the UHSNet model, which leverages advanced deep learning techniques to provide a fast and accurate alternative to traditional computational fluid dynamics (CFD) simulations for optimizing UHS processes.
For more details, refer to the published paper: UHSNet: Deep learning-based smart proxy modeling for underground hydrogen storage ([DOI](https://doi.org/10.1016/j.energy.2025.136763)).

## Features
Deep Learning Architecture: Integrates dilated convolutional layers, residual connections, and a hybrid Huber-MAE loss function for enhanced predictive performance and generalization.
High-Fidelity Dataset: Built using CFD simulations enriched with laboratory data and thermodynamic principles, sampled via Latin Hypercube for comprehensive coverage.
Performance: Achieves a 15% reduction in Mean Absolute Percentage Error (MAPE) compared to baseline CNNs and U-Net variants, with prediction times under 1 second—up to 10⁴× faster than CFD.

Applications: Supports real-time decision-making and large-scale scenario evaluations for UHS in saline aquifers, aiding renewable energy system planning.

## Repository Contents
Code: Implementation of the UHSNet model, including training scripts and model architecture.

## Citation
If you use this code or model in your research, please cite:

Asghari, M., Emami Niri, M., & Sedaee, B. (2025). UHSNet: Deep learning-based smart proxy modeling for underground hydrogen storage. Energy, 329, 136763. https://doi.org/10.1016/j.energy.2025.136763

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions, bug reports, or feature requests.

## Contact
For questions or inquiries, please contact Milad Asghari.
