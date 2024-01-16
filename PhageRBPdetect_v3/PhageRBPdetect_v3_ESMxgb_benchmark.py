"""
PhageRBPdetect (ESM2-XGB) - benchmarking
@author: dimiboeckaerts
@date: 2023-12-17
"""

# 0 - SET THE PATHS
# ------------------------------------------
path = '/Users/dimi/GoogleDrive/PhD/3_PHAGEBASE/32_DATA/RBP_detection'

import esm
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef 
import pandas as pd


# 1 - COMPUTE EMBEDDINGS
# ------------------------------------------
# load the ESM2 model
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D() # esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t33_650M_UR50D
last_layer = model.num_layers
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
#tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D") # "esm2_t33_650M_UR50D"
#model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
RBPs = pd.read_csv(path+'/annotated_RBPs_2022-01.csv')
nonRBPs = pd.read_csv(path+'/annotated_nonRBPs_2022-01.csv')
nonRBPs_sub = nonRBPs.sample(n=10*RBPs.shape[0], random_state=42)
nonRBPs_sub = nonRBPs_sub.reset_index(drop=True)

print('Computing embeddings...')
bar = tqdm(total=len(RBPs['ProteinSeq']), position=0, leave=True)
sequence_representations = []
for i, sequence in enumerate(RBPs['ProteinSeq']):
    data = [(RBPs['protein_id'][i], sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[last_layer], return_contacts=True)
    token_representations = results["representations"][last_layer]
    for j, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[j, 1 : len(seq) + 1].mean(0))
    bar.update(1)
bar.close()
embeddings_df = pd.concat([RBPs['RecordDate'], pd.DataFrame(sequence_representations).astype('float')], axis=1)
embeddings_df.to_csv(path+'/RBP_embeddings_esm.csv', index=False)

bar = tqdm(total=len(nonRBPs_sub['ProteinSeq']), position=0, leave=True)
sequence_representations = []
for i, sequence in enumerate(nonRBPs_sub['ProteinSeq']):
    data = [(nonRBPs_sub['protein_id'][i], sequence[:2000])]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[last_layer], return_contacts=True)
    token_representations = results["representations"][last_layer]
    for j, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[j, 1 : len(seq) + 1].mean(0))
    bar.update(1)
bar.close()
embeddings_df = pd.concat([nonRBPs_sub['RecordDate'], pd.DataFrame(sequence_representations).astype('float')], axis=1)
embeddings_df.to_csv(path+'/nonRBP_embeddings_esm.csv', index=False)

# bar = tqdm(total=RBPs.shape[0])
# RBP_embeddings = np.zeros((RBPs.shape[0], 320))
# for i, sequence in enumerate(RBPs['ProteinSeq']):
#     inputs = tokenizer(sequence[:1024], return_tensors="pt")
#     outputs = model(**inputs)
#     token_embeddings = outputs.last_hidden_state
#     sequence_embeddings = torch.mean(token_embeddings[:, 1:-1], dim=1)
#     RBP_embeddings[i, :] = sequence_embeddings.detach().numpy()
#     bar.update(1)
# bar.close()

# 2 - TRAIN & TUNING THE MODEL
# ------------------------------------------
# get data up untill SEPT 2021
rbps_em = pd.read_csv(path+'/RBP_embeddings_esm.csv')
nonrbps_em = pd.read_csv(path+'/nonRBP_embeddings_esm.csv')
months = ['OCT-2021', 'NOV-2021', 'DEC-2021']
to_delete_rbps = [i for i, date in enumerate(rbps_em['RecordDate']) if any(x in date for x in months)]
rbps_upto2021 = rbps_em.drop(to_delete_rbps)
rbps_upto2021 = rbps_upto2021.reset_index(drop=True)
to_delete_nonrbps = [i for i, date in enumerate(nonrbps_em['RecordDate']) if any(x in date for x in months)]
nonrbps_upto2021 = nonrbps_em.drop(to_delete_nonrbps)
nonrbps_upto2021 = nonrbps_upto2021.reset_index(drop=True)

rbp_embed = np.asarray(rbps_upto2021.iloc[:, 1:])
nonrbp_embed = np.asarray(nonrbps_upto2021.iloc[:, 1:])

features = np.concatenate((rbp_embed, nonrbp_embed))
labels = np.asarray([1]*rbp_embed.shape[0] + [0]*nonrbp_embed.shape[0])
print('Check?', features.shape[0]==labels.shape[0])

# do grid search to tune hyperparams
print('Tuning hyperparameters...')
cv = StratifiedKFold(n_splits=5, shuffle=True)
imbalance = rbp_embed.shape[0]/nonrbp_embed.shape[0]
cpus = 6
xgbmodel = xgb.XGBClassifier(scale_pos_weight=1/imbalance, n_jobs=cpus, use_label_encoder=False)
params_xgb = {'max_depth': [3, 5], 'n_estimators': [250, 500], 'learning_rate': [0.05, 0.1, 0.2]}
tuned_xgb = GridSearchCV(xgbmodel, cv=cv, param_grid=params_xgb, scoring='f1', verbose=2)
tuned_xgb.fit(features, labels, eval_metric='logloss')
print('Params and results: ', tuned_xgb.best_params_, tuned_xgb.best_score_) #.cv_results_
# {'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 500}

# train the best model on all data up untill 2021
xgb_all = xgb.XGBClassifier(**tuned_xgb.best_params_, scale_pos_weight=1/imbalance, 
                            n_jobs=cpus, use_label_encoder=False)
xgb_all.fit(features, labels, eval_metric='logloss')

# save model for later benchmark with phANNs
xgb_all.save_model(path+'/RBPdetect_esm_xgb.json')


# 3 - BENCHMARKING
# ------------------------------------------
# get old labels for performance comparison
rbps_emo = pd.read_csv(path+'/annotated_RBPs_2022-01_embeddings.csv')
nonrbps_emo = pd.read_csv(path+'/annotated_nonRBPs_2022-01_embeddings.csv')
months = ['OCT-2021', 'NOV-2021', 'DEC-2021']
to_delete_rbpso = [i for i, date in enumerate(rbps_emo['RecordDate']) if all(x not in date for x in months)]
rbps_2021o = rbps_emo.drop(to_delete_rbpso)
rbps_2021o = rbps_2021o.reset_index(drop=True)
to_delete_nonrbps = [i for i, date in enumerate(nonrbps_emo['RecordDate']) if all(x not in date for x in months)]
nonrbps_2021o = nonrbps_emo.drop(to_delete_nonrbps)
nonrbps_2021o = nonrbps_2021o.reset_index(drop=True)
testlabelso = np.asarray([1]*rbps_2021o.shape[0] + [0]*nonrbps_2021o.shape[0])
print(len(testlabelso))

# load data and select 2021 (oct-nov-dec) data as test set
rbps_em = pd.read_csv(path+'/RBP_embeddings_esm.csv')
nonrbps_em = pd.read_csv(path+'/nonRBP_embeddings_esm.csv')
months = ['OCT-2021', 'NOV-2021', 'DEC-2021']
to_delete_rbps = [i for i, date in enumerate(rbps_em['RecordDate']) if all(x not in date for x in months)]
rbps_2021 = rbps_em.drop(to_delete_rbps)
rbps_2021 = rbps_2021.reset_index(drop=True)
to_delete_nonrbps = [i for i, date in enumerate(nonrbps_em['RecordDate']) if all(x not in date for x in months)]
nonrbps_2021 = nonrbps_em.drop(to_delete_nonrbps)
nonrbps_2021 = nonrbps_2021.reset_index(drop=True)
testdata = pd.concat([rbps_2021, nonrbps_2021], axis=0)
print('Check?', testdata.shape[0]==(rbps_2021.shape[0]+nonrbps_2021.shape[0]))

# features for our XGBoost model
testfeatures = np.asarray(testdata.iloc[:, 1:])
testlabels = np.asarray([1]*rbps_2021.shape[0] + [0]*nonrbps_2021.shape[0])

# load model and make preds
xgb_saved = xgb.XGBClassifier()
xgb_saved.load_model(path+'/RBPdetect_esm_xgb.json')
score_xgb = xgb_saved.predict_proba(testfeatures)[:,1]
preds_xgb = (score_xgb > 0.5)*1

# save predictions and scores
xgbesm_results = pd.concat([pd.DataFrame(preds_xgb, columns=['preds']), 
                        pd.DataFrame(score_xgb, columns=['score'])], axis=1)
results_path = '/Users/dimi/GoogleDrive/PhD/3_PHAGEBASE/33_RESULTS/RBP_detection'
xgbesm_results.to_csv(results_path+'/esm_xgb_test_predictions.csv', index=False)

# load results
results_path = '/Users/dimi/GoogleDrive/PhD/3_PHAGEBASE/33_RESULTS/RBP_detection'
phanns_results = pd.DataFrame({}, columns=['preds', 'score'])
for i in range(1,7):
    part = pd.read_csv(path+'/phanns_predictions_part'+str(i)+'.csv')
    phanns_results = pd.concat([phanns_results, part], axis=0)
domain_results = pd.read_csv(results_path+'/domains_test_predictions.csv')
xgb_results = pd.read_csv(results_path+'/xgboost_test_predictions.csv')
xgbhmm_results = pd.read_csv(results_path+'/xgboost_HMMscores_test_predictions.csv')
xgbesm_results= pd.read_csv(results_path+'/esm_xgb_test_predictions.csv')
esm_results = pd.read_csv(results_path+'/esm_finetune_test_predictions.csv')
esm33_results = pd.read_csv(results_path+'/esmT33_finetune_test_predictions.csv')


# compute performances
domain_f1 = round(f1_score(testlabelso, list(domain_results['preds'])), 4)
domain_mcc = round(matthews_corrcoef(testlabelso, list(domain_results['preds'])), 4)
domain_tn, domain_fp, domain_fn, domain_tp = confusion_matrix(testlabelso, list(domain_results['preds'])).ravel()
domain_sensitivity = round(domain_tp / (domain_tp + domain_fn), 4)
domain_specificity = round(domain_tn / (domain_tn + domain_fp), 4)
phanns_f1 = round(f1_score(testlabelso, list(phanns_results['preds'])), 4)
phanns_mcc = round(matthews_corrcoef(testlabelso, list(phanns_results['preds'])), 4)
phanns_tn, phanns_fp, phanns_fn, phanns_tp = confusion_matrix(testlabelso, list(phanns_results['preds'])).ravel()
phanns_sensitivity = round(phanns_tp / (phanns_tp + phanns_fn), 4)
phanns_specificity = round(phanns_tn / (phanns_tn + phanns_fp), 4)
xgb_f1 = round(f1_score(testlabelso, list(xgb_results['preds'])), 4)
xgb_mcc = round(matthews_corrcoef(testlabelso, list(xgb_results['preds'])), 4)
xgb_tn, xgb_fp, xgb_fn, xgb_tp = confusion_matrix(testlabelso, list(xgb_results['preds'])).ravel()
xgb_sensitivity = round(xgb_tp / (xgb_tp + xgb_fn), 4)
xgb_specificity = round(xgb_tn / (xgb_tn + xgb_fp), 4)
xgbhmm_f1 = round(f1_score(testlabelso, list(xgbhmm_results['preds'])), 4)
xgbhmm_mcc = round(matthews_corrcoef(testlabelso, list(xgbhmm_results['preds'])), 4)
xgb_tn, xgb_fp, xgb_fn, xgb_tp = confusion_matrix(testlabelso, list(xgbhmm_results['preds'])).ravel()
xgbhmm_sensitivity = round(xgb_tp / (xgb_tp + xgb_fn), 4)
xgbhmm_specificity = round(xgb_tn / (xgb_tn + xgb_fp), 4)
xgbesm_f1 = round(f1_score(testlabels, list(xgbesm_results['preds'])), 4)
xgbesm_mcc = round(matthews_corrcoef(testlabels, list(xgbesm_results['preds'])), 4)
xgb_tn, xgb_fp, xgb_fn, xgb_tp = confusion_matrix(testlabels, list(xgbesm_results['preds'])).ravel()
xgbesm_sensitivity = round(xgb_tp / (xgb_tp + xgb_fn), 4)
xgbesm_specificity = round(xgb_tn / (xgb_tn + xgb_fp), 4)
esm_f1 = round(f1_score(testlabels, list(esm_results['preds'])), 4)
esm_mcc = round(matthews_corrcoef(testlabels, list(esm_results['preds'])), 4)
esm_tn, esm_fp, esm_fn, esm_tp = confusion_matrix(testlabels, list(esm_results['preds'])).ravel()
esm_sensitivity = round(esm_tp / (esm_tp + esm_fn), 4)
esm_specificity = round(esm_tn / (esm_tn + esm_fp), 4)
esm33_f1 = round(f1_score(testlabels, list(esm33_results['preds'])), 4)
esm33_mcc = round(matthews_corrcoef(testlabels, list(esm33_results['preds'])), 4)
esm33_tn, esm33_fp, esm33_fn, esm33_tp = confusion_matrix(testlabels, list(esm33_results['preds'])).ravel()
esm33_sensitivity = round(esm33_tp / (esm33_tp + esm33_fn), 4)
esm33_specificity = round(esm33_tn / (esm33_tn + esm33_fp), 4)


scores_df = pd.DataFrame({
    'Method': ['Domain-based', 'PhANNs', 'PTB-XGBoost', 'HMM-PTB-XGBoost', 'ESM2-XGBoost', 'ESM-FineTuned', 'ESM-T33'],
    'F1 Score': [domain_f1, phanns_f1, xgb_f1, xgbhmm_f1, xgbesm_f1, esm_f1, esm33_f1],
    'MCC Score': [domain_mcc, phanns_mcc, xgb_mcc, xgbhmm_mcc, xgbesm_mcc, esm_mcc, esm33_mcc],
    'Sensitivity': [domain_sensitivity, phanns_sensitivity, xgb_sensitivity, xgbhmm_sensitivity, xgbesm_sensitivity, esm_sensitivity, esm33_sensitivity],
    'Specificity': [domain_specificity, phanns_specificity, xgb_specificity, xgbhmm_specificity, xgbesm_specificity, esm_specificity, esm33_specificity],
})
print(scores_df)