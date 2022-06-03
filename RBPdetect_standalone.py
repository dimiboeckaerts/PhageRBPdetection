"""
This script provides the functionalities to run our methods from the command line, starting from a
FASTA file of protein sequences you want to make predictions for.

WARNING: 
Computing embeddings is computationally intensive and is not recommended on a personal laptop
without sufficient GPU capabilities. Computing the embedding vector for even a single sequence 
can take 10+ minutes on a regular CPU. You can use the provided notebook in a Google Colab or 
Kaggle environment instead and work with free GPUs in the cloud.

INPUTS:
--dir: the directory which contains the FASTA file with name 'sequences.fasta', and the HMM file and trained XGBoost model (.json file)
--hmmer_path: path to HMMER (e.g. /Users/Sally/hmmer-3.3.1)

USAGE FROM CMD LINE:
python RBPdetect_standalone.py --dir /directory/with/files --hmmer_path /path/to/hmmer

@author: dimiboeckaerts
"""

# LIBRARIES
# ------------------------------------------
import os
import math
import subprocess
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from Bio.SearchIO import HmmerIO
from bio_embeddings.embed import ProtTransBertBFDEmbedder
from xgboost import XGBClassifier


# PARSE ARGUMENTS
# ------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dir',dest='dir', required=True)
parser.add_argument('--hmmer_path',dest='hmmer_path', required=True)
args = parser.parse_args()


# FUNCTIONS
# ------------------------------------------
def hmmpress_python(hmm_path, pfam_file):
    """
    Presses a profiles database, necessary to do scanning.
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    press_str = 'hmmpress ' + pfam_file
    command = cd_str+'; '+press_str
    press_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    press_out, press_err = press_process.communicate()

    return press_out, press_err

def single_hmmscan_python(hmm_path, pfam_file, fasta_file):
    """
    Does a hmmscan for a given FASTA file of one (or multiple) sequences,
    against a given profile database. Assuming an already pressed profiles
    database (see function above).
    
    INPUT: all paths to the hmm, profiles_db, fasta and results file given as strings.
            results_file should be a .txt file
    OUPUT: ...
    """

    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_str = cd_process.communicate()

    # scan the sequences
    scan_str = 'hmmscan ' + pfam_file + ' ' + fasta_file + ' > hmmscan_out.txt'
    scan_process = subprocess.Popen(scan_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    scan_out, scan_err = scan_process.communicate()
    
    # process output
    results_handle = open('hmmscan_out.txt')
    scan_res = HmmerIO.Hmmer3TextParser(results_handle)
    os.remove('hmmscan_out.txt')
    
    return scan_out, scan_err, scan_res

def RBPdetect_domains_protein(path, pfam_file, fasta_file, N_blocks=[], C_blocks=[], detect_others=True):
    """
    Same function as above but for a protein FASTA file as input!

    Inputs:
    - path: path to HMM software for detection of the domains
    - pfam_file: path to local Pfam database file (string)
    - fasta_file: path to sequences FASTA file (string)
    - N_blocks: list of structural (N-terminal) domains as strings (corresponding to names in Pfam database)
    - C_blocks: list of binding (C-terminal) domains as strings (corresponding to names in Pfam database)

    Output:
    - a dataframe of RBPs
    """
    N_list = []; C_list = []
    rangeN_list = []; rangeC_list = []
    sequences_list = []; identifiers_list = []
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, 'fasta')]
    names = [record.id for record in SeqIO.parse(fasta_file, 'fasta')]
    bar = tqdm(total=len(sequences), leave=True, desc='Scanning the proteins')
    
    for i, sequence in enumerate(sequences):
        N_sequence = []; C_sequence = []
        rangeN_sequence = []; rangeC_sequence = []
        
        # make single-sequence FASTA file
        temp_fasta = open('single_sequence.fasta', 'w')
        temp_fasta.write('>'+names[i]+'\n'+sequence+'\n')
        temp_fasta.close()
        
        # scan HMM
        _, _, scan_res = single_hmmscan_python(path, pfam_file, 'single_sequence.fasta')

        # fetch domains in the results
        domains = []; scores = []; biases = []; ranges = []
        for line in scan_res:   
            try:   
                for hit in line.hits:
                    hsp = hit._items[0] # highest scoring domain
                    aln_start = hsp.query_range[0]
                    aln_stop = hsp.query_range[1]

                    if (hit.bitscore >= 18) & (hit.id not in domains):
                        domains.append(hit.id)
                        scores.append(hit.bitscore)
                        biases.append(hit.bias)
                        ranges.append((aln_start,aln_stop))
            except IndexError: # some hits don't have an individual domain hit
                pass
        
        # remove temp fasta file
        os.remove('single_sequence.fasta')
        
        # loop over the domains, if any
        if len(domains) > 0:
            for j, dom in enumerate(domains):
                OM_score = math.floor(math.log(scores[j], 10)) # order of magnitude
                OM_bias = math.floor(math.log(biases[j]+0.00001, 10))
                
                # N-terminal block
                if (OM_score > OM_bias) and (dom in N_blocks):
                    N_sequence.append(dom)
                    rangeN_sequence.append(ranges[j])
                
                # C-terminal block
                elif (OM_score > OM_bias) and (dom in C_blocks) and (scores[j] >= 25):
                    C_sequence.append(dom)
                    rangeC_sequence.append(ranges[j])
                
                # other block
                elif (detect_others == True) and (OM_score > OM_bias) and (dom not in N_blocks) and (dom not in C_blocks):
                    if ranges[j][1] <= 200:
                        N_sequence.append('other')
                        rangeN_sequence.append(ranges[j])
                    elif (ranges[j][1] > 200) and (scores[j] >= 25):
                        C_sequence.append('other')
                        rangeC_sequence.append(ranges[j])
                 
            # add to the global list
            if (len(N_sequence) > 0) or (len(C_sequence) > 0):
                N_list.append(N_sequence)
                C_list.append(C_sequence)
                rangeN_list.append(rangeN_sequence)
                rangeC_list.append(rangeC_sequence)
                sequences_list.append(sequence)
                identifiers_list.append(names[i])

        # update bar
        bar.update(1)
    bar.close()

    # make dataframe
    detected_RBPs = pd.DataFrame({'identifier':identifiers_list, 'DNASeq':sequences_list, 'N_blocks':N_list, 'C_blocks':C_list, 
                                'N_ranges':rangeN_list, 'C_ranges':rangeC_list})
    return detected_RBPs

def compute_protein_embeddings(fasta_file):
    embedder = ProtTransBertBFDEmbedder()
    names = [record.id for record in SeqIO.parse(fasta_file, 'fasta')]
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, 'fasta')]
    embeddings = [embedder.reduce_per_protein(embedder.embed(sequence)) for sequence in tqdm(sequences)]
    embeddings_df = pd.concat([pd.DataFrame({'ID':names}), pd.DataFrame(embeddings)], axis=1)
    return embeddings_df


# RUN HMM PREDICTIONS
# ------------------------------------------
print('Running HMM predictions...')

# file names
pfam_file = args.dir+'/RBPdetect_phageRBPs.hmm'
xgb_file = args.dir+'/RBPdetect_xgb_model.json'
fasta_file = args.dir+'/sequences.fasta'

# press the .hmm file for further use
output, err = hmmpress_python(args.hmmer_path, pfam_file)

# define HMMs to be detected as RBP-related
N_blocks = ['Phage_T7_tail', 'Tail_spike_N', 'Prophage_tail', 'BppU_N', 'Mtd_N', 
           'Head_binding', 'DUF3751', 'End_N_terminal', 'phage_tail_N', 'Prophage_tailD1', 
           'DUF2163', 'Phage_fiber_2', 'unknown_N0', 'unknown_N1', 'unknown_N2', 'unknown_N3', 'unknown_N4', 
            'unknown_N6', 'unknown_N10', 'unknown_N11', 'unknown_N12', 'unknown_N13', 'unknown_N17', 'unknown_N19', 
            'unknown_N23', 'unknown_N24', 'unknown_N26','unknown_N29', 'unknown_N36', 'unknown_N45', 'unknown_N48', 
            'unknown_N49', 'unknown_N53', 'unknown_N57', 'unknown_N60', 'unknown_N61', 'unknown_N65', 'unknown_N73', 
            'unknown_N82', 'unknown_N83', 'unknown_N101', 'unknown_N114', 'unknown_N119', 'unknown_N122', 
            'unknown_N163', 'unknown_N174', 'unknown_N192', 'unknown_N200', 'unknown_N206', 'unknown_N208']
C_blocks = ['Lipase_GDSL_2', 'Pectate_lyase_3', 'gp37_C', 'Beta_helix', 'Gp58', 'End_beta_propel', 
            'End_tail_spike', 'End_beta_barrel', 'PhageP22-tail', 'Phage_spike_2', 
            'gp12-short_mid', 'Collar', 
            'unknown_C2', 'unknown_C3', 'unknown_C8', 'unknown_C15', 'unknown_C35', 'unknown_C54', 'unknown_C76', 
            'unknown_C100', 'unknown_C105', 'unknown_C112', 'unknown_C123', 'unknown_C179', 'unknown_C201', 
            'unknown_C203', 'unknown_C228', 'unknown_C234', 'unknown_C242', 'unknown_C258', 'unknown_C262', 
            'unknown_C267', 'unknown_C268', 'unknown_C274', 'unknown_C286', 'unknown_C292', 'unknown_C294', 
            'Peptidase_S74', 'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']

# do domain-based detections
domain_based_detections = RBPdetect_domains_protein(args.hmmer_path, pfam_file, fasta_file, N_blocks=N_blocks, 
                                                         C_blocks=C_blocks, detect_others=False)

names = [record.id for record in SeqIO.parse(fasta_file, 'fasta')]
domain_preds = []
for pid in names:
    if pid in list(domain_based_detections['identifier']):
        domain_preds.append(1)
    else:
        domain_preds.append(0)

# save the domain-based results separately (optionally)
domain_results = pd.DataFrame(domain_preds, columns=['preds'])
domain_results.to_csv(args.dir+'/domains_test_predictions.csv', index=False)


# RUN XGBOOST PREDICTIONS
# ------------------------------------------
print('Computing embeddings... (this can take a while on a CPU)')
embeddings = compute_protein_embeddings(fasta_file)
embeddings_array = np.asarray(embeddings.iloc[:, 2:])

# load trained model
print('Running XGBoost predictions...')
xgb_saved = XGBClassifier()
xgb_saved.load_model(xgb_file)

# make predictions with the XGBoost model
score_xgb = xgb_saved.predict_proba(embeddings)[:,1]
preds_xgb = (score_xgb > 0.5)*1

# save predictions and scores (optionally)
xgb_results = pd.concat([pd.DataFrame(preds_xgb, columns=['preds']), 
                        pd.DataFrame(score_xgb, columns=['score'])], axis=1)
xgb_results.to_csv(args.dir+'/xgboost_test_predictions.csv', index=False)
