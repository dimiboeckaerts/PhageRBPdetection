"""
RBP DETECTION

Created on Tue Sep  1 18:47:06 2020

@author: dimiboeckaerts
"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import re
import json
import math
import pandas as pd
import seaborn as sns
import networkx as nx
from tqdm import tqdm
from Bio import SeqIO
from Bio import Entrez
from Bio.Seq import Seq
import matplotlib.pyplot as plt
from Bio.Blast import NCBIWWW, NCBIXML
import time
import urllib
import subprocess
import numpy as np
from Bio.SearchIO import HmmerIO

# 1 - FUNCTIONS
# --------------------------------------------------
def hmmpress_python(hmm_path, pfam_file):
    """
    Presses a profiles database, necessary to do scanning.
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_err = cd_process.communicate()
    
    # compress the profiles db
    press_str = 'hmmpress ' + pfam_file
    press_process = subprocess.Popen(press_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    press_out, press_err = press_process.communicate()

    return press_out, press_err

def hmmbuild_python(hmm_path, output_file, msa_file):
    """
    Build a profile HMM from an input multiple sequence alignment (Stockholm format).
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_err = cd_process.communicate()
    
    # compress the profiles db
    press_str = 'hmmbuild ' + output_file + ' ' + msa_file
    press_process = subprocess.Popen(press_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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


def hmmscan_python(hmm_path, pfam_file, sequences_file, counts=False):
    """
    Expanded version of the function above for a file with multiple sequences,
    where the results are parsed one by one to fetch the names of domains that
    we're interested in. Assumes an already pressed profiles database (see 
    funtion above).
    
    HMMSCAN = scan a (or multiple) sequence(s) for domains.
    """
    
    domains = []
    scores = []
    biases = []
    ranges = []
    count_dict = {}
    for sequence in SeqIO.parse(sequences_file, 'fasta'):
        # make single-sequence FASTA file
        temp_fasta = open('single_sequence.fasta', 'w')
        temp_fasta.write('>'+sequence.id+'\n'+str(sequence.seq)+'\n')
        temp_fasta.close()
        
        # scan HMM
        _, _, scan_res = single_hmmscan_python(hmm_path, pfam_file, 'single_sequence.fasta')
        
        # fetch domains in the results
        for line in scan_res:   
            try:   
                for hit in line.hits:
                    hsp = hit._items[0] # highest scoring domain
                    aln_start = hsp.query_range[0]
                    aln_stop = hsp.query_range[1]
        
                    if (hit.bitscore >= 25) & (hit.id not in domains):
                        domains.append(hit.id)
                        scores.append(hit.bitscore)
                        biases.append(hit.bias)
                        ranges.append((aln_start,aln_stop))
                        count_dict[hit.id] = 1
                    elif (hit.bitscore >= 25) & (hit.id in domains):
                        count_dict[hit.id] += 1
            except IndexError: # some hits don't have an individual domain hit
                pass
    
    # remove last temp fasta file
    os.remove('single_sequence.fasta')
    
    if counts == True:
        return domains, count_dict
    else:
        return domains, scores, biases, ranges
    

def hmmfetch_python(hmm_path, pfam_file, domains, output_file):
    """
    Fetches the HMM profiles for given domains. Necessary to do hmmsearch with
    afterwards.
    
    INPUT: paths to files and domains as a list of strings.
    """
    
    # save domains as txt file
    domain_file = open('selected_domains.txt', 'w')
    for domain in domains:
        domain_file.write(domain+'\n')
    domain_file.close()
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_str = cd_process.communicate()
    
    # fetch selected domains in new hmm
    fetch_str = 'hmmfetch -f ' + pfam_file + ' selected_domains.txt'
    fetch_process = subprocess.Popen(fetch_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fetch_out, fetch_err = fetch_process.communicate()
    
    # write to specified output file
    hmm_out = open(output_file, 'wb')
    hmm_out.write(fetch_out)
    hmm_out.close()
    
    return fetch_out, fetch_err


def hmmsearch_python(hmm_path, selected_profiles_file, sequences_db):
    """
    HMMSEARCH = search (selected) domain(s) in a sequence database.
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_str = cd_process.communicate()
    
    # search domains in sequence database
    search_str = 'hmmsearch ' + selected_profiles_file + ' ' + sequences_db + ' > hmmsearch_out.txt'
    search_process = subprocess.Popen(search_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    search_out, search_err = search_process.communicate()
    
    # process output
    results_handle = open('hmmsearch_out.txt')
    search_res = HmmerIO.Hmmer3TextParser(results_handle)
    os.remove('hmmsearch_out.txt')
    
    sequence_hits = []
    sequence_scores = []
    sequence_bias = []
    sequence_range = []
    for line in search_res:
        for hit in line:
            try:
                hsp = hit._items[0]
                aln_start = hsp.query_range[0]
                aln_stop = hsp.query_range[1]
                if (hit.bitscore >= 25):
                    sequence_hits.append(hit.id)
                    sequence_scores.append(hit.bitscore)
                    sequence_bias.append(hit.bias)
                    sequence_range.append((aln_start,aln_stop))
            except IndexError:
                pass
                
    return sequence_hits, sequence_scores, sequence_bias, sequence_range


def hmmsearch_thres_python(hmm_path, selected_profiles_file, sequences_db, threshold):
    """
    HMMSEARCH = search (selected) domain(s) in a sequence database.
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_str = cd_process.communicate()
    
    # search domains in sequence database
    search_str = 'hmmsearch ' + selected_profiles_file + ' ' + sequences_db + ' > hmmsearch_out.txt'
    search_process = subprocess.Popen(search_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    search_out, search_err = search_process.communicate()
    
    # process output
    results_handle = open('hmmsearch_out.txt')
    search_res = HmmerIO.Hmmer3TextParser(results_handle)
    os.remove('hmmsearch_out.txt')
    
    sequence_hits = []
    sequence_scores = []
    sequence_bias = []
    sequence_range = []
    for line in search_res:
        for hit in line:
            try:
                hsp = hit._items[0]
                aln_start = hsp.query_range[0]
                aln_stop = hsp.query_range[1]
                if (hit.bitscore >= threshold):
                    sequence_hits.append(hit.id)
                    sequence_scores.append(hit.bitscore)
                    sequence_bias.append(hit.bias)
                    sequence_range.append((aln_start,aln_stop))
            except IndexError:
                pass
                
    return sequence_hits, sequence_scores, sequence_bias, sequence_range


def gene_domain_search(hmmpath, selected_profiles_file, gene_sequence):
    """
    This function translates a given gene sequence to its protein sequence, 
    saves it as a FASTA file and searches the given domain(s) in the protein
    sequence.
    """
    # translate
    protein_sequence = str(Seq(gene_sequence).translate())[:-1] # don't need stop codon

    # write fasta
    temp_fasta = open('protein_sequence.fasta', 'w')
    temp_fasta.write('>protein_sequence'+'\n'+protein_sequence+'\n')
    temp_fasta.close()
    
    # search domain
    hits, scores, biases, ranges = hmmsearch_python(hmmpath, selected_profiles_file, 'protein_sequence.fasta')

    # delete fasta
    os.remove('protein_sequence.fasta')
    
    return hits, scores, biases, ranges


def gene_domain_scan(hmmpath, pfam_file, gene_hits):
    """
    This function does a hmmscan on the gene_hit(s) after translating them to
    protein sequences and saving it in one FASTA file.
    """

    # write the protein sequences to file
    hits_fasta = open('protein_hits.fasta', 'w')
    for i, gene_hit in enumerate(gene_hits):
        protein_sequence = str(Seq(gene_hit).translate())[:-1]
        hits_fasta.write('>'+str(i)+'_proteindomain_hit'+'\n'+protein_sequence+'\n')
    hits_fasta.close()
        
    # fetch domains with hmmscan
    domains, scores, biases, ranges = hmmscan_python(hmmpath, pfam_file, 'protein_hits.fasta')
    
    return domains, scores, biases, ranges


def all_domains_scan(path, pfam_file, gene_sequences):
    """
    scan all sequences and make dictionary of results
    """
    domain_dict = {'gene_sequence': [], 'domain_name': [], 'position': [], 'score': [], 'bias': [], 'aln_range': []}
    bar = tqdm(total=len(gene_sequences), desc='Scanning the genes', position=0, leave=True)
    for gene in gene_sequences:
        hits, scores, biases, ranges = gene_domain_scan(path, pfam_file, [gene])
        for i, dom in enumerate(hits):
            domain_dict['gene_sequence'].append(gene)
            domain_dict['domain_name'].append(dom)
            domain_dict['score'].append(scores[i])
            domain_dict['bias'].append(biases[i])
            domain_dict['aln_range'].append(ranges[i])
            if ranges[i][1] > 200:
                domain_dict['position'].append('C')
            else:
                domain_dict['position'].append('N')
            
        bar.update(1)
    bar.close()
    
    return domain_dict


def domain_RBP_predictor(path, pfam_file, sequences, structural=[], binding=[], chaperone=[], archive=[]):
    """
    This function predicts whether or not a sequence is a phage RBP based on known
    related phage RBP protein domains in Pfam. Predictions are made based on the
    knowledge that RBPs are modular, consisting of a structural (N-terminal) domain,
    a C-terminal binding domain and optionally a chaperone domain at the C-end.
    
    Inputs: 
        - structural, binding, chaperone: curated lists of phage-related domains
            in Pfam.
        - path: path to HMM software for detection of the domains
        - pfam_file: link to local Pfam database file (string)
        - sequences: 
            * link to FASTA file of gene sequences to predict (string)
            OR 
            * list of sequences as string
            OR 
            * dictionary of domains (cfr. function 'all_domains_scan')
    Output:
        - a pandas dataframe of sequences in which at least one domain has been
            detected.
    """
    
    # initializations
    predictions_dict = {'gene_sequence': [], 'structural_domains': [], 
                        'binding_domains': [], 'chaperone_domains': [],
                        'archived_domains': []}
    if type(sequences) == 'string':
        # make list of sequences
        sequence_list = []
        for record in SeqIO.parse(sequences, 'fasta'):
            sequence_list.append(str(record.seq))
        # make domain dictionary
        domains_dictionary = all_domains_scan(path, pfam_file, sequence_list)
        
    elif type(sequences) == 'list':
        # make domain dictionary
        domains_dictionary = all_domains_scan(path, pfam_file, sequences)
    else: # dict
        domains_dictionary = sequences
    
    # make predictions based on the domains_dictionary
    domain_sequences = list(set(domains_dictionary['gene_sequence']))
    for sequence in domain_sequences:
        # detect all domains at correct indices: every line in domains_dictionary
        # corresponds to a gene sequence and a domain in it (one sequence can
        # have multiple domains, thus multiple lines in the dict).
        domains = []
        indices = [i for i, gene in enumerate(domains_dictionary['gene_sequence']) 
                    if sequence == gene]
        for index in indices:
            OM_score = math.floor(math.log(domains_dictionary['score'][index], 10)) # order of magnitude
            OM_bias = math.floor(math.log(domains_dictionary['bias'][index]+0.00001, 10))
            if (OM_score > OM_bias):
                domains.append(domains_dictionary['domain_name'][index])
               
        # detect the domains of interest
        struct_detected = [dom for dom in domains if dom in structural]
        binding_detected = [dom for dom in domains if dom in binding]
        chaperone_detected = [dom for dom in domains if dom in chaperone]
        archive_detected = [dom for dom in domains if dom in archive]
        
        # append results dictionary
        if (len(struct_detected) > 0) or (len(binding_detected) > 0) or (len(chaperone_detected) > 0):
            predictions_dict['gene_sequence'].append(sequence)
            predictions_dict['structural_domains'].append(struct_detected)
            predictions_dict['binding_domains'].append(binding_detected)
            predictions_dict['chaperone_domains'].append(chaperone_detected)
            predictions_dict['archived_domains'].append(archive_detected)
    
    return predictions_dict


# 2 - EXAMPLE
# --------------------------------------------------
# define paths and press new database
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
new_pfam = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A_extended.hmm'
output, err = hmmpress_python(path, new_pfam)

genes_df = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase_df.csv', index_col=0)
unique_genes = genes_df.columns
phage_genomes = genes_df.index

# redo the all_domains scan to update the json files (takes a long time!)
prodomains_dict = all_domains_scan(path, new_pfam, list(unique_genes))
domaindump = json.dumps(prodomains_dict)
domfile = open(dom_path+'/prodomains_dict.json', 'w')
domfile.write(domaindump)
domfile.close()

# define updated domain lists (structs, bindings, chaps) & make predictions
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
structs = ['Phage_T7_tail', 'Tail_spike_N', 'Prophage_tail', 'BppU_N', 'Mtd_N', 
           'Head_binding', 'DUF3751', 'End_N_terminal', 'phage_tail_N', 'Prophage_tailD1', 
           'DUF2163', 'Phage_fiber_2', 'phage_RBP_N1', 'phage_RBP_N4', 'phage_RBP_N26', 
           'phage_RBP_N28', 'phage_RBP_N34', 'phage_RBP_N45']
bindings = ['Lipase_GDSL_2', 'Pectate_lyase_3', 'gp37_C', 'Beta_helix', 'Gp58', 'End_beta_propel', 
            'End_tail_spike', 'End_beta_barrel', 'PhageP22-tail', 'Phage_spike_2', 
            'gp12-short_mid', 'Collar', 'phage_RBP_C2', 'phage_RBP_C10', 'phage_RBP_C24',
            'phage_RBP_C43', 'phage_RBP_C59', 'phage_RBP_C60', 'phage_RBP_C62', 'phage_RBP_C67',
            'phage_RBP_C79', 'phage_RBP_C97', 'phage_RBP_C111', 'phage_RBP_C115', 'phage_RBP_C120'
            'phage_RBP_C126', 'phage_RBP_C138', 'phage_RBP_C43', 'phage_RBP_C157', 'phage_RBP_C164', 
            'phage_RBP_C175', 'phage_RBP_C180', 'phage_RBP_C205', 'phage_RBP_C217', 'phage_RBP_C220', 
            'phage_RBP_C221', 'phage_RBP_C223', 'phage_RBP_C234', 'phage_RBP_C235', 'phage_RBP_C237',
            'phage_RBP_C249', 'phage_RBP_C259', 'phage_RBP_C267', 'phage_RBP_C271', 'phage_RBP_C277',
            'phage_RBP_C281', 'phage_RBP_C292', 'phage_RBP_C293', 'phage_RBP_C296', 'phage_RBP_C300', 
            'phage_RBP_C301', 'phage_RBP_C319', 'phage_RBP_C320', 'phage_RBP_C321', 'phage_RBP_C326', 
            'phage_RBP_C331', 'phage_RBP_C337', 'phage_RBP_C338', 'phage_RBP_C340']
chaps = ['Peptidase_S74', 'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']
archive = ['Exo_endo_phos', 'NosD', 'SASA', 'Peptidase_M23', 'Phage_fiber']

domfile_pro = open(dom_path+'/prodomains_dict.json')
prodomains_dict = json.load(domfile_pro)
preds = domain_RBP_predictor(path, new_pfam, prodomains_dict, structs, bindings, chaps, archive)
preds_df = pd.DataFrame.from_dict(preds)
preds_df.iloc[:50, 1:]