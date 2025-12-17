# SketchDNN: Joint Continuous-Discrete Diffusion for CAD Sketch Generation
This repository contains code for the paper [Chereddy & Femiani SketchDNN: Joint Continuous-Discrete Diffusion for CAD Sketch Generation. In Internation Conference on Machine Learning (ICML) 2025](https://arxiv.org/abs/2507.11579)

SketchDNN is a generative diffusion model for Computer Aided Design (CAD) sketches/diagrams that employs a joint coninuous-discrete diffusion paradigm based on a novel Gaussian-Softmax diffusion process.
***
## Environment Setup
To setup the Conda environment for SketchDNN, run the following commands:

    git clone https://github.com/Sathware/SketchDNN.git
    conda env create -f environment.yaml
***
## Model Training
To train the model, run the following command:

    python3 train.py

The dataset will automatically be downloaded, processed, and configured if not already present. Furthermore, training details can be visualized via Tensorboard with the following command:

    tensorboard --logdir runs
***
## Model Evaluation
All model evaluation code is available in the notebook `evaluation.ipynb`

