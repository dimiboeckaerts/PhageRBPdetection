"""
PhageRBPdetect (ESM2-fine) - training
@author: dimiboeckaerts
@date: 2023-12-21

SET THE PATHS BELOW, THEN RUN THE SCRIPT

INPUTS: a .csv file with annotated RBPs and a .csv file with annotated non-RBPs
OUTPUTS: a fine-tuned ESM-2 model for RBP detection

Notes: 
You will probably want to run this script on a GPU-enabled machine (e.g. Google Colab or Kaggle).
The ESM-2 T12 model can run on a single GPU with 16GB of memory.
Taken from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb#scrollTo=3d2edc14.
"""

# 0 - SET THE PATHS
# ------------------------------------------
path = './data'
RBPfile = 'annotated_RBPs_2022-01.csv'
nonRBPfile = 'annotated_nonRBPs_2022-01.csv'

# 1 - TRAINING THE MODEL
# ------------------------------------------
# libraries
import torch
import pandas as pd
import numpy as np
from evaluate import load
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# define path & model checkpoint
model_checkpoint = "facebook/esm2_t12_35M_UR50D" # esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t33_650M_UR50D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ["WANDB_DISABLED"] = "true"

# load data
RBPs = pd.read_csv(path+'/'+RBPfile)
nonRBPs = pd.read_csv(path+'/'+nonRBPfile)
nonRBPs_sub = nonRBPs.sample(n=10*RBPs.shape[0], random_state=42)
nonRBPs_sub = nonRBPs_sub.reset_index(drop=True)
RBPseqs = RBPs['ProteinSeq'].tolist()
nonRBPseqs = [seq[:2000] for seq in list(nonRBPs_sub['ProteinSeq'])] # cutoff after 2000AAs, to avoid memory issues
sequences = RBPseqs + nonRBPseqs
labels = [1]*RBPs.shape[0] + [0]*nonRBPs_sub.shape[0]

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