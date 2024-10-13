# DNN Interpretability for Knowledge Discovery: COVID-19 Morbidity Classification

This project aims to apply interpretability methods to DNN models trained for the classification of COVID-19 morbidity on an [iCTCF dataset](https://ngdc.cncb.ac.cn/ictcf/) (clinical features and computed tomography scans). It provides tools for data preprocessing, model training, hyperparameter tuning, and interpretability analysis, including prototype generation, feature importance, and saliency maps.

## Data and Results

A subset of the data used in this project, including clinical features (CF) and computed tomography (CT) scans for a few patients, is available for download from our Google Drive folder. Additionally, the folder contains:
- Trained models
- Results files.
- The final report for the project (master's thesis).
- A PDF with supplementary material.

You can access the Google Drive folder here:  
[Google Drive - iCTCF dataset, results, and supplementary material](https://drive.google.com/drive/folders/1Jd2gXZMPqqW83ZChOMmxD5o_FD8DOpGM?usp=share_link)

## Additional packages
Ensure you are using **Python version 3.10**. You can install the additional dependencies by running the following command with the `requirements.txt` file provided in the project folder:

```bash
pip install -r requirements.txt
```

## Project Structure

The `src` folder contains the following subdirectories and Python scripts:

- **data/**: 
  - Handles downloading data, creating the `cf` and `ct` datasets, and performing exploratory data analysis.
  
- **preprocessing/**:
  - Specific preprocessing methods for the `cf` and `ct` datasets.
  
- **model/**:
  - Contains the model classes, model trainer class, and preparation of the optimizer.
  
- **train.py**:
  - Script for training the model on the datasets.
  
- **sweep.py**:
  - Script for running hyperparameter tuning using [Weights and Biases Sweeps](https://docs.wandb.ai/guides/sweeps).
  
- **evaluate.py**:
  - Script to evaluate a pre-trained model on the test dataset (as well as train and validation).

- **interpretability/**:
  - **Prototype generation**
  - **Global feature importance**
  - **Feature Attribution - saliency maps**
  - **Latent space analysis**

- **VAE/**:
  - Contains all scripts related to Variational Autoencoder (VAE) training and analysis for VAE-based prototype generation.

- **utils/**:
  - Additional utility functions to assist with tasks across the project.
