"""
PhageRBPdetect (ESM2-fine) - benchmarking
@author: dimiboeckaerts
@date: 2023-12-19

Notes: 
You will probably want to run this script on a GPU-enabled machine (e.g. Google Colab or Kaggle).
The ESM-2 T12 model can run on a single GPU with 16GB of memory.
Taken from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb#scrollTo=3d2edc14.
If you want to train the ESM-2 T33 model, you will need a machine with 32GB or more memory.
"""
#!pip install evaluate datasets

# 0 - SET THE PATHS
# ------------------------------------------
path = '/Users/dimi/GoogleDrive/PhD/3_PHAGEBASE/32_DATA/RBP_detection'

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from evaluate import load
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split


# 1 - TRAIN & TUNING THE MODEL
# ------------------------------------------
# define path & model checkpoint
model_checkpoint = "facebook/esm2_t12_35M_UR50D" # esm2_t12_35M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ["WANDB_DISABLED"] = "true"

# get data until SEPT 2021
RBPs = pd.read_csv(path+'/annotated_RBPs_2022-01.csv')
nonRBPs = pd.read_csv(path+'/annotated_nonRBPs_2022-01.csv')
nonRBPs_sub = nonRBPs.sample(n=10*RBPs.shape[0], random_state=42)
nonRBPs_sub = nonRBPs_sub.reset_index(drop=True)
months = ['OCT-2021', 'NOV-2021', 'DEC-2021']
to_delete_rbps = [i for i, date in enumerate(RBPs['RecordDate']) if any(x in date for x in months)]
rbps_upto2021 = RBPs.drop(to_delete_rbps)
rbps_upto2021 = rbps_upto2021.reset_index(drop=True)
to_delete_nonrbps = [i for i, date in enumerate(nonRBPs_sub['RecordDate']) if any(x in date for x in months)]
nonrbps_upto2021 = nonRBPs_sub.drop(to_delete_nonrbps)
nonrbps_upto2021 = nonrbps_upto2021.reset_index(drop=True)
RBPseqs = rbps_upto2021['ProteinSeq'].tolist()
nonRBPseqs = [seq[:2000] for seq in list(nonrbps_upto2021['ProteinSeq'])]
sequences = RBPseqs + nonRBPseqs
labels = [1]*rbps_upto2021.shape[0] + [0]*nonrbps_upto2021.shape[0]

# process the data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.1, stratify=labels, random_state=42)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)
train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)
train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

# define function for metric
metric = load("f1")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# finetune the model (takes around 1h per epoch on NVIDIA P100 GPU)
nlabels = len(set(labels))
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=nlabels)
batch_size = 2
args = TrainingArguments(
    'RBPdetect_ESM2finetune',
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# save the model & tokenizer
model_path = path+'/RBPdetect_v3_ESMfine'
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# 2 - BENCHMARKING
# ------------------------------------------
model_path = path+'/RBPdetect_v3_ESMfineT33'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval().cuda() # or without cuda if not available

RBPs = pd.read_csv(path+'/annotated_RBPs_2022-01.csv')
nonRBPs = pd.read_csv(path+'/annotated_nonRBPs_2022-01.csv')
nonRBPs_sub = nonRBPs.sample(n=10*RBPs.shape[0], random_state=42)
nonRBPs_sub = nonRBPs_sub.reset_index(drop=True)
months = ['OCT-2021', 'NOV-2021', 'DEC-2021']
to_delete_rbps = [i for i, date in enumerate(RBPs['RecordDate']) if all(x not in date for x in months)]
rbps_2021 = RBPs.drop(to_delete_rbps)
rbps_2021 = rbps_2021.reset_index(drop=True)
to_delete_nonrbps = [i for i, date in enumerate(nonRBPs_sub['RecordDate']) if all(x not in date for x in months)]
nonrbps_2021 = nonRBPs_sub.drop(to_delete_nonrbps)
nonrbps_2021 = nonrbps_2021.reset_index(drop=True)
testdata = list(rbps_2021['ProteinSeq']) + list(nonrbps_2021['ProteinSeq'])
testlabels = [1]*rbps_2021.shape[0] + [0]*nonrbps_2021.shape[0]

predictions = []
scores = []
for sequence in tqdm(testdata):
    encoding = tokenizer(sequence, return_tensors="pt", truncation=True).to('cuda:0')#.to('mps:0') # or without cuda if not available
    with torch.no_grad():
        output = model(**encoding)
        predictions.append(int(output.logits.argmax(-1)))
        scores.append(float(output.logits.softmax(-1)[:, 1]))

esm_results = pd.concat([pd.DataFrame(predictions, columns=['preds']), 
                        pd.DataFrame(scores, columns=['score'])], axis=1)
results_path = '/Users/dimi/GoogleDrive/PhD/3_PHAGEBASE/33_RESULTS/RBP_detection'
esm_results.to_csv(results_path+'/esm_finetuneT33_test_predictions.csv', index=False)