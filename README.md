[![DOI](https://zenodo.org/badge/417444396.svg)](https://zenodo.org/badge/latestdoi/417444396) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b0DSqMmnEgoXoWW53VxKpT-N8moPU2DA?usp=sharing)

# PhageRBPdetection

## News

PhageRBPdetect v3 is here! We have both improved and simplified the entire PhageRBPdetection pipeline using a finetuned ESM-2 model. Read below to get started right away.

## General information

This is the repository related to our published manuscript in the special issue 'Virus Bioinformatics 2022' in *Viruses*:
"Identification of phage receptor-binding protein sequences with hidden Markov models and an extreme gradient boosting classifier". 
Read and/or download the manuscript here: https://www.mdpi.com/1999-4915/14/6/1329.

The repository contains the following code and data:
1. <ins>PhageRBPdetect_v3</ins>: the most recent version of our PhageRBPdetect tool, including two _benchmark.py files for comparing performance on the original datasets, a _training.py file to finetune an ESM-2 model with our dataset or sequences of your own choosing and a _inference.py file to make predictions for new sequences using our finetuned model.
2. <ins>PhageRBPdetect_v2</ins>: all of the code related to our originally published work, including a notebook containing all the analyses (_notebook.ipynb), a notebook to compute ProtTrans embeddings (_protein_embeddings.ipynb), a notebook to make predictions for new sequences (_make_predictions.ipynb), and a standalone version that can be run from the command line (_standalone.py).
3. <ins>data</ins>: folder containing the two originally trained XGBoost models, the collection of RBP-related HMMs and a examples FASTA file of three sequences that are RBPs (should be predicted as 1).

## Benchmarking results

PhageRBPdetect v3 reaches an improved or stable performance across every metric we measured on the original data. These updated benchmark results are found in the table below.

| Method                     | F1 score | MCC score | Sensitivity | Specificity |
| -------------------------- | -------- | --------- | ----------- | ----------- |
| RBP domains (HMMs)         | 72.0%    | 70.2%     | 66.4%       | 98.5%       |
| PhANNs                     | 69.8%    | 67.9%     | 81.6%       | 95.8%       |
| ProtTransBert+XGBoost      | 84.0%    | 82.3%     | 91.6%       | 97.9%       |
| ProtTransBert+HMMs+XGBoost | 84.8%    | 83.8%     | 92.2%       | 98.0%       |
| ESM-2 + XGBoost            | 85.0%    | 83.9%     | 90.9%       | 98.1%       |
| ESM-2 Finetuned (T33)      | 86.4%    | 85.4%     | 91.6%       | 98.4%       |

## Make predictions yourself

#### 1. Predictions with PhageRBPdetect v3

The easiest way to start making predictions with the streamlined and best-performing model is to make predictions with our [Google Collab notebook](https://colab.research.google.com/drive/1b0DSqMmnEgoXoWW53VxKpT-N8moPU2DA?usp=sharing) and follow the three steps there! You can also run it locally with the `PhageRBPdetect_v3_inference.py` file.

#### 2. Predictions with the older PhageRBPdetect v2 (original work in published manuscript)

Using the RBPdetect_make_predictions notebook: Simply open the *PhageRBPdetect_v2_make_predictions.ipynb* notebook and follow the instructions. If you are new to Jupyter notebooks, one of the easiest ways to get started is by installing [Anaconda](https://www.anaconda.com/products/individual). Currently, this notebook allows you to make predictions for either both parallel approaches separately, as well as the combined approach using both language embeddings and HMM scores in an XGBoost model.

Using the RBPdetect_standalone script: This is the commandline version of the tool, which can also be used to make predictions. Currently, this tool allows to make predictions for both parallel approaches separately. Instructions are provided in the file itself. Warning: as computing embeddings is computationally demanding, we do not recommend running the pipeline on a personal computer without sufficient GPU capabilities.

#### 3. Finetuning the model

You can also finetune the ESM-2 model yourself on either our data or your own data. See the `PhageRBPdetect_v3_training.py` file in the `PhageRBPdetect_v3` folder.

## Original datasets

To reproduce our analyses from the manuscript with the originally collected datasets, you can download these datasets from Zenodo at https://doi.org/10.5281/zenodo.6607535.
