"""
PhageRBPdetect (ESM2-fine) - inference
@author: dimiboeckaerts
@date: 2023-12-21

FIRST, DOWNLOAD THE MODEL FILE FROM ZENODO: https://zenodo.org/records/10515367 
THEN, SET THE PATHS & FILES BELOW & RUN THE SCRIPT

INPUTS: a FASTA file with proteins you want to make predictions for, and a fine-tuned ESM-2 model
OUTPUTS: predictions and scores for each protein in the FASTA file

Notes: 
You will probably want to run this script on a GPU-enabled machine (e.g. Google Colab or Kaggle).
The ESM-2 T12 model can run on a single GPU with 16GB of memory.
You can also directly access our Google Colab notebook here: https://colab.research.google.com/drive/1b0DSqMmnEgoXoWW53VxKpT-N8moPU2DA?usp=sharing.
The results will be saved in a `predictions.csv` file in the path. The file will contain 3 columns: the protein names, 
the binary prediction (0: predicted not an RBP, 1: predicted an RBP) and the associated score 
(between 0 and 1, the higher, the more confident the model is in it being an RBP).

Any feedback or questions? Feel free to send me an email: dimi.boeckaerts@gmail.com.
"""

# 0 - SET THE PATHS
# ------------------------------------------
path = '../data'
fasta_name = 'sequences.fasta'
model_name = 'RBPdetect_v4_ESMfine' # should be a folder in the path!

# 1 - TRAINING THE MODEL
# ------------------------------------------
# load libraries
import os
import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

GPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
device = torch.device("cuda")

# initiation the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(path+'/'+model_name)
model = AutoModelForSequenceClassification.from_pretrained(path+'/'+model_name)
if torch.cuda.is_available():
    model.eval().cuda()
    print("using cuda")
else:
    model.eval()
    print("Using CPU")

# make predictions
sequences = [str(record.seq) for record in SeqIO.parse(path+'/'+fasta_name, 'fasta')]
names = [record.id for record in SeqIO.parse(path+'/'+fasta_name, 'fasta')]

predictions = []
scores = []
for sequence in tqdm(sequences):
    encoding = tokenizer(sequence, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output = model(**encoding)
        predictions.append(int(output.logits.argmax(-1)))
        scores.append(float(output.logits.softmax(-1)[:, 1]))

# save the results
results = pd.concat([pd.DataFrame(names, columns=['protein_name']),
                         pd.DataFrame(predictions, columns=['preds']),
                        pd.DataFrame(scores, columns=['score'])], axis=1)
results.to_csv(path+'/predictions.csv', index=False)

"""### Step 3: save predictions!

The results will be saved in a `predictions.csv` file in the content folder on the left, where you uploaded the FASTA file as well. From there, you can right click the .csv file and download the predictions!

The file will contain 3 columns: the protein names, the binary prediction (0: predicted not an RBP, 1: predicted an RBP) and the associated score (between 0 and 1, the higher, the more confident the model is in it being an RBP).

Any feedback or questions? Feel free to send me an email: dimi.boeckaerts@gmail.com.
"""