[![DOI](https://zenodo.org/badge/417444396.svg)](https://zenodo.org/badge/latestdoi/417444396)

# PhageRBPdetection

**Important notice December 2023:** We have discovered problems with installing the bio-embeddings package on both Kaggle and Colab platforms, resulting in the inability to proceed and make predictions. We aim to resolve this by moving the entire pipeline to ESM-2 in the next weeks. Thank you for understanding.

## General information

This is the repository related to our published manuscript in the special issue 'Virus Bioinformatics 2022' in *Viruses*:
"Identification of phage receptor-binding protein sequences with hidden Markov models and an extreme gradient boosting classifier". 
Read and/or download the manuscript here: https://www.mdpi.com/1999-4915/14/6/1329.

The repository contains the following code and data:
1. <ins>RBPdetect_notebook</ins>: the main notebook containing all the analyses that are discussed in the manuscript.
2. <ins>RBPdetect_protein_embeddings</ins>: notebook for computing protein language embeddings that serve as the feature representation for our machine learning models.
3. <ins>RBPdetect_make_predictions</ins>: notebook containing the code and explanation to make predictions for sequences of your choice.
4. <ins>RBPdetect_standalone.py</ins>: a standalone full version of the prediction pipeline (including both our approaches) that can be run from the command line.
5. <ins>RBPdetect_utils</ins>: Python script containing all the needed functions for the analyses.
6. <ins>RBPdetect_domains</ins>: deprecated code of initial analyses, available for reference purposes only.
7. <ins>data</ins>: folder containing the two trained XGBoost models, the collection of RBP-related HMMs and a examples FASTA file of three sequences that are RBPs (should be predicted as 1).

## Make predictions yourself

To get started making predictions for your own sequences of choice, start by cloning/copying/downloading this repository. From there, you have two options for making predictions.

### 1. Using the RBPdetect_make_predictions notebook

Simply open the *RBPdetect_make_predictions.ipynb* notebook and follow the instructions. If you are new to Jupyter notebooks, one of the easiest ways to get started is by installing [Anaconda](https://www.anaconda.com/products/individual). Currently, this notebook allows you to make predictions for either both parallel approaches separately, as well as the combined approach using both language embeddings and HMM scores in an XGBoost model.

### 2. Using the RBPdetect_standalone script

This is the commandline version of the tool, which can also be used to make predictions. Currently, this tool allows to make predictions for both parallel approaches separately. Instructions are provided in the file itself. Warning: as computing embeddings is computationally demanding, we do not recommend running the pipeline on a personal computer without sufficient GPU capabilities.

## Original datasets

To reproduce our analyses from the manuscript with the originally collected datasets, you can download these datasets from Zenodo at https://doi.org/10.5281/zenodo.6607535.
