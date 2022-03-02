# PhageRBPdetection

This is the repository related to our manuscript in submission:
"Dual identification of novel phage receptor-binding proteins based on protein domains and machine learning".

The repository contains the following code and data:
1. <ins>RBPdetect_notebook</ins>: the main notebook containing all the analyses that are discussed in the manuscript.
2. <ins>RBPdetect_protein_embeddings</ins>: notebook for computing protein language embeddings that serve as the feature representation for our machine learning model.
3. <ins>RBPdetect_make_predictions</ins>: notebook containing the code and explanation to make predictions for sequences of your choice.
4. <ins>RBPdetect_make_utils</ins>: Python script containing all the needed functions for the analyses.
5. <ins>RBPdetect_domains</ins>: deprecated code of initial analyses, available for reference purposes only.
6. <ins>data</ins>: folder containing the trained XGBoost model, the collection of RBP-related HMMs and an examples FASTA file


To get started making predictions for your own sequences of choice, simply clone/copy/download this repository and open the *RBPdetect_make_predictions* notebook and follow the instructions. If you are new to jupyter notebooks, one of the easiest ways to get started is by installing [Anaconda](https://www.anaconda.com/products/individual).
