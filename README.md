# Punjabi-Poet-Attribution-Using-Deep-Learning

## Overview

This repository contains a deep learning-based Poet Attribution model for Punjabi poetry, which aims to address issues related to plagiarism and misattribution in the digital age. The model is designed to identify the poet of a given poem written in three different scripts: **Shahmukhi**, **Gurmukhi**, and **Roman**. The project uses multilingual deep learning models such as **DistilBERT**, **Bi-LSTM**, and **Bi-GRU** for classification as well as Machine Learning Models, achieving high accuracy across the scripts.

## Paper
The details of the research, methodology, and results are outlined in the paper titled "Deep Learning-based Poet Attribution Model for Punjabi Poetry". Due to access restrictions, the full paper is not available publicly, but you can contact the authors for more information or access to the paper. https://ieeexplore.ieee.org/document/10737982

## Dataset

The dataset consists of 830 poems from **11 distinct poets**, including renowned poets like **Bulleh Shah**, **Waris Shah**, and **Shiv Kumar Batalvi**, among others. These poems are available in three different scripts:

- **Roman**
- **Gurmukhi**
- **Shahmukhi**

The dataset was curated from GitHub and Folk Punjab, ensuring diversity and cultural relevance in the chosen poets.

## Scripts

- **Roman**: Used predominantly in online platforms.
- **Gurmukhi**: Commonly used in India.
- **Shahmukhi**: Used in Pakistan.

## Methodology

### Data Preparation
- The dataset is split into **training**, **validation**, and **test** sets.
- Tokenization is performed to convert the poems into sub-words using appropriate tokenizers.
- **DistilBERT** is used to generate embeddings for the poems in a 768-dimensional vector space.

### Models Used

- **Bi-GRU**: Bidirectional Gated Recurrent Unit for sequential data processing.
- **Bi-LSTM**: Bidirectional Long Short-Term Memory for capturing context in both forward and backward directions.
- **DistilBERT**: A compact version of BERT optimized for efficiency and multilingual capabilities.

### Machine Learning Models

- **Random Forest**
- **Softmax Regression**
- **SVM (Support Vector Machine)**

### Model Evaluation

The models are evaluated using accuracy as the primary evaluation criterion, with **DistilBERT** showing the best performance across different scripts.

## Installation

To run this project locally, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Punjabi-Poet-Attribution.git
cd Punjabi-Poet-Attribution
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model, run the provided Jupyter notebook (`PPC.ipynb`):

```bash
jupyter notebook PPC.ipynb
```

The notebook contains all the necessary code to preprocess the data, train the models, and evaluate their performance.

### Model Inference
Once the model is trained, you can use it to predict the poet of a given Punjabi poem by passing the poem as input to the trained model.

## Results

The results obtained from testing the models on the dataset are as follows:

| Model       | Gurmukhi (%) | Roman (%) | Shahmukhi (%) |
|-------------|--------------|-----------|---------------|
| **Bi-LSTM** | 87.95        | 91.57     | 81.93         |
| **Bi-GRU**  | 89.16        | 87.95     | 80.72         |
| **DistilBERT** | 90.36     | 87.95     | 87.95         |
| **SVM**     | 85.54        | 89.16     | 80.72         |
| **Softmax** | 86.75        | 84.34     | 79.53         |
| **Random Forest** | 63.86  | 71.08     | 68.67         |

## Limitations

- The dataset contains limited representations of some poets, which could affect the model’s generalization ability.
- A more diverse dataset with more poets would likely improve model performance.
- The model performs better on the Roman script compared to Shahmukhi and Gurmukhi.

