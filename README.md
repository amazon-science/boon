# BOON: Boundary correction for neural operators

![Image](resources/operator_bdy.png)

Nadim Saad*, Gaurav Gupta*, Shima Alizadeh, Danielle C. Maddix\
**Guiding continuous operator learning through Physics-based boundary constraints**\
[arXiv:2212.07477](https://arxiv.org/abs/2212.07477)\
(*equal contribution authors)


## Setup

### Requirements
The code package is developed using Python 3.8 and Pytorch 1.11 with cuda 11.6. The code could be executed on CPU/GPU but GPU is preferred. All experiments were conducted on Tesla V100 16GB.

## Experiments
### Data
Generate the data using the scripts provided in the 'Data' directory. The scripts use Matlab 2018+. A sample generated dataset for all the experiments is available below.

[BOON PDE datasets](https://drive.google.com/drive/folders/1tj3dBlM6NQk6qo9cwyLaJmvLnXTho0yD?usp=sharing)

### Scripts
Detailed notebooks for reproducing all the experiments in the paper are provided. The cases of 1D, 1D time-varying, 2D time-varying are shown in the respective notebooks for all the three boundary conditions of Dirichlet, Neumann, and Periodic.

### 1D Stokes' second problem
As an example, a complete pipeline is shown for the 1D time-varying PDE with Dirichlet boundary condition in the attached `examples_1d_multi_step.ipynb` notebook.

### lid-Cavity (Navier Stokes)
A complete pipeline is shown for the 2D time-varying PDE with Dirichlet boundary condition in the attached `examples_3d_multi_step.ipynb` notebook.

## Citation
If you use this code, or our work, please cite:
```
@misc{saad2022BOON,
  author = {Saad, Nadim and Gupta, Gaurav and Alizadeh, Shima and Maddix, Danielle C.},
  title = {Guiding continuous operator learning through Physics-based boundary constraints},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2212.07477},
}
```
