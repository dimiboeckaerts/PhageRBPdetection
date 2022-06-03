[![DOI](https://zenodo.org/badge/417444396.svg)](https://zenodo.org/badge/latestdoi/417444396)

# PhageRBPdetection

This is the repository related to our manuscript in submission for the special issue 'Virus Bioinformatics 2022' in *Viruses*:
"Identification of phage receptor-binding protein sequences with hidden Markov models and an extreme gradient boosting classifier".

The repository contains the following code and data:
1. <ins>RBPdetect_notebook</ins>: the main notebook containing all the analyses that are discussed in the manuscript.
2. <ins>RBPdetect_protein_embeddings</ins>: notebook for computing protein language embeddings that serve as the feature representation for our machine learning model.
3. <ins>RBPdetect_make_predictions</ins>: notebook containing the code and explanation to make predictions for sequences of your choice.
4. <ins>RBPdetect_standalone.py</ins>: a standalone full version of the prediction pipeline (including both our approaches) that can be run from the command line.
5. <ins>RBPdetect_utils</ins>: Python script containing all the needed functions for the analyses.
6. <ins>RBPdetect_domains</ins>: deprecated code of initial analyses, available for reference purposes only.
7. <ins>data</ins>: folder containing the trained XGBoost model, the collection of RBP-related HMMs and an examples FASTA file.

To get started making predictions for your own sequences of choice, simply clone/copy/download this repository and open the *RBPdetect_make_predictions* notebook and follow the instructions. If you are new to Jupyter notebooks, one of the easiest ways to get started is by installing [Anaconda](https://www.anaconda.com/products/individual). Alternatively, you can also make predictions with the *RBPdetect_standalone.py* script from the command line (see the file itself for instructions and needed libraries). However, as computing embeddings is computationally demanding, we do not recommend running the pipeline on a personal computer without sufficient GPU capabilities.

To reproduce our analyses from the manuscript with the originally collected datasets, you can download these datasets from Zenodo at https://doi.org/10.5281/zenodo.6607535.
