# aind-disrnn-wrapper

The wrapper capsule in the AIND-disRNN MLOps stack:

<img width="1380" height="875" alt="image" src="https://github.com/user-attachments/assets/e029c0e3-ce47-4f65-b61f-42c8bd5b053a" />

## Installation in HPC
To install the capsule in an HPC environment, follow these steps:
1. Create a new conda environment:
   ```bash
   conda create -n disrnn python=3.12 -y
   conda activate disrnn
   ```
2. Install the required packages in editable mode:
   ```bash
   pip install -e .
   ```
    if want to use GPU version, use
    ```bash
    pip install -e ".[gpu]"
    ```
   with dev tools
    ```bash
    pip install -e ".[dev]"
    ```