"""
RBP DETECTION utility functions

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
from tqdm.notebook import tqdm
from Bio import SeqIO
from Bio import Entrez
from Bio.Seq import Seq
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
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
    press_str = 'hmmpress ' + pfam_file
    command = cd_str+'; '+press_str
    press_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    press_out, press_err = press_process.communicate()

    return press_out, press_err


def hmmbuild_python(hmm_path, output_file, msa_file):
    """
    Build a profile HMM from an input multiple sequence alignment (Stockholm format).
    """
    
    cd_str = 'cd ' + hmm_path # change dir
    press_str = 'hmmbuild ' + output_file + ' ' + msa_file
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


def hmmscan_python(hmm_path, pfam_file, sequences_file, threshold=18):
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
        
                    if (hit.bitscore >= threshold) & (hit.id not in domains):
                        domains.append(hit.id)
                        scores.append(hit.bitscore)
                        biases.append(hit.bias)
                        ranges.append((aln_start,aln_stop))
            except IndexError: # some hits don't have an individual domain hit
                pass
    
    # remove last temp fasta file
    os.remove('single_sequence.fasta')

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


def gene_domain_scan(hmmpath, pfam_file, gene_hits, threshold=18):
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
    domains, scores, biases, ranges = hmmscan_python(hmmpath, pfam_file, 'protein_hits.fasta', threshold)
    
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


def RBP_domain_predict(path, pfam_file, sequences, structural=[], binding=[], chaperone=[], archive=[]):
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


def RBPdetect_domains(path, pfam_file, sequences, identifiers, N_blocks=[], C_blocks=[], detect_others=True):
    """
    This function serves as the main function to do domain-based RBP detection based on
    either a manually curated list of RBP-related Pfam domains or Pfam domains appended with 
    custom HMMs. If custom HMMs are added, these HMMs must correspondingly be added in the Pfam
    database that is scanned!

    Inputs:
    - path: path to HMM software for detection of the domains
    - pfam_file: link to local Pfam database file (string)
    - sequences: list of strings, DNA sequences
    - identifiers: corresponding list of identifiers for the sequences (string)
    - N_blocks: list of structural (N-terminal) domains as strings (corresponding to names in Pfam database)
    - C_blocks: list of binding (C-terminal) domains as strings (corresponding to names in Pfam database)

    Output:
    - a dataframe of RBPs
    """
    bar = tqdm(total=len(sequences), leave=True, desc='Scanning the genes')
    N_list = []; C_list = []
    rangeN_list = []; rangeC_list = []
    sequences_list = []; identifiers_list = []
    for i, sequence in enumerate(sequences):
        N_sequence = []
        C_sequence = []
        rangeN_sequence = []
        rangeC_sequence = []
        domains, scores, biases, ranges = gene_domain_scan(path, pfam_file, [sequence], threshold=18)
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
                identifiers_list.append(identifiers[i])

        # update bar
        bar.update(1)
    bar.close()

    # make dataframe
    detected_RBPs = pd.DataFrame({'identifier':identifiers_list, 'DNASeq':sequences_list, 'N_blocks':N_list, 'C_blocks':C_list, 
                                'N_ranges':rangeN_list, 'C_ranges':rangeC_list})
    return detected_RBPs


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


def cdhit_python(cdhit_path, input_file, output_file, c=0.50, n=3):
    """
    This function executes CD-HIT clustering commands from within Python. To install
    CD-HIT, do so via conda: conda install -c bioconda cd-hit. By default, CD-HIT
    works via a global alignment approach, which is good for our application as
    we cut the sequences to 'one unknown domain' beforehand.
    
    Input:
        - cdhit_path: path to CD-HIT software
        - input_file: FASTA file with protein sequences
        - output file: path to output (will be one FASTA file and one .clstr file)
        - c: threshold on identity for clustering
        - n: word length (3 for thresholds between 0.5 and 0.6)
    """
    
    cd_str = 'cd ' + cdhit_path # change directory
    raw_str = './cd-hit -i ' + input_file + ' -o ' + output_file + ' -c ' + str(c) + ' -n ' + str(n) + ' -d 0'
    command = cd_str+'; '+ raw_str
    #cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #cd_out, cd_err = cd_process.communicate()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    
    return stdout, stderr


def clustalo_python(clustalo_path, input_file, output_file, out_format='fa'):
    """
    This function executes the basic command to run a local Clustal Omega MSA.
    You need to install Clustal Omega locally first, see http://www.clustal.org/omega/.
    The basic command is: clustalo -i my-in-seqs.fa -o my-out-seqs.fa -v
    
    Dependencies: subprocess
    
    Input:
        - clustalo_path: path to clustalo software
        - input_file: FASTA file with (protein) sequences
        - output_file: path to output file for MSA
        - out_format: format of the output (fa[sta],clu[stal],msf,phy[lip],selex,
                        st[ockholm],vie[nna]; default: fasta) as string
        
    Output: stdout and stderr are the output from the terminal. Results are saved 
            as given output_file.
    """
    
    cd_str = 'cd ' + clustalo_path # change dir
    raw_str = './clustalo -i ' + input_file + ' -o ' + output_file + ' -v --outfmt ' + out_format
    command = cd_str+'; '+ raw_str
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    
    return stdout, stderr


def build_custom_HMMs(detected_RBPs, data_dir, cd_path, hmm_path):
    """
    This function builds new HMMs starting from a dataframe of detected RBPs.

    Input:
    - detected_RBPs: dataframe with sequences and domains, output of RBPdetect_domains
    - data_dir: directory to store new HMMs (string)
    - cdpath: path to CD-HIT

    Output: newly constructed HMMs
    """
    # check unknown N-termini and C-termini and write fasta files
    print('Checking unknown sequences...')
    unknown_N = open(data_dir+'/unknown_N_sequences.fasta', 'w')
    unknown_C = open(data_dir+'/unknown_C_sequences.fasta', 'w')
    for i, sequence in enumerate(detected_RBPs['DNASeq']):
        protein = str(Seq(sequence).translate())[:-1]
        # check empty N_blocks (first 200 AAs)
        if (len(detected_RBPs['N_blocks'][i]) == 0) and (len(protein) > 200):
            identifier_N = detected_RBPs['identifier'][i]
            unknown_N.write('>'+identifier_N+'\n'+protein[:200]+'\n')
        # check empty C_blocks (starting from 200 AAs, longer then 50 AAs)
        elif (len(detected_RBPs['C_blocks'][i]) == 0) and (len(protein) > 250):
            ranges_N = detected_RBPs['N_ranges'][i]
            last_N_domain = max([ranges[1] for ranges in ranges_N])
            identifier_C = detected_RBPs['identifier'][i]
            unknown_C.write('>'+identifier_C+'\n'+protein[last_N_domain:]+'\n')
    unknown_N.close()
    unknown_C.close()

    # cluster fasta files with CD-HIT
    print('Clustering with CD-HIT...')
    # os.mkdir(data_dir+'/clusters')
    input_N = data_dir+'/unknown_N_sequences.fasta'
    output_N = data_dir+'/clusters/unknown_N_sequences'
    outN, errN = cdhit_python(cd_path, input_N, output_N)
    input_C = data_dir+'/unknown_C_sequences.fasta'
    output_C = data_dir+'/clusters/unknown_C_sequences'
    outC, errC = cdhit_python(cd_path, input_C, output_C)
    
    # make separate fastas for each cluster
    clusters_unknown_N = open(data_dir+'/clusters/unknown_N_sequences.clstr')
    sequences_N = []; ids_N = []
    for record in SeqIO.parse(input_N, 'fasta'): # get sequences in a list
        sequences_N.append(str(record.seq))
        ids_N.append(record.description) # sequences and ids match indices
    cluster_iter = -1; cluster_sequences = []; cluster_indices = []; clusters_N = []
    for line in clusters_unknown_N.readlines():
        if line[0] == '>': # new cluster
            if (cluster_iter >= 0) & (len(cluster_sequences) >= 5): # finish old if not the first
                fasta = open(data_dir+'/clusters/unknown_N_sequences_cluster_'+str(cluster_iter)+'.fasta', 'w')
                for i, seq in enumerate(cluster_sequences):
                    this_id = cluster_indices[i]
                    fasta.write('>sequence_'+str(this_id)+'\n'+seq+'\n')
                fasta.close()
                clusters_N.append(str(cluster_iter))
            # initiate new cluster
            cluster_sequences = []; cluster_indices = []
            cluster_iter += 1
        else: # in a cluster
            current_id = line.split('>')[1].split('...')[0]
            current_index = ids_N.index(current_id) #current_index = [i for i, description in enumerate(ids_N) if current_id == description]
            cluster_indices.append(current_index)
            cluster_sequences.append(sequences_N[current_index])
    
    clusters_unknown_C = open(data_dir+'/clusters/unknown_C_sequences.clstr')
    sequences_C = []; ids_C = []
    for record in SeqIO.parse(input_C, 'fasta'): # get sequences in a list
        sequences_C.append(str(record.seq))
        ids_C.append(record.description) # sequences and ids match indices
    cluster_iter = -1; cluster_sequences = []; cluster_indices = []; clusters_C = []
    for line in clusters_unknown_C.readlines():
        if line[0] == '>': # new cluster
            if (cluster_iter >= 0) & (len(cluster_sequences) >= 5): # finish old if not the first
                fasta = open(data_dir+'/clusters/unknown_C_sequences_cluster_'+str(cluster_iter)+'.fasta', 'w')
                for i, seq in enumerate(cluster_sequences):
                    this_id = cluster_indices[i]
                    fasta.write('>sequence_'+str(this_id)+'\n'+seq+'\n')
                fasta.close()
                clusters_C.append(str(cluster_iter))
            # initiate new cluster
            cluster_sequences = []; cluster_indices = []
            cluster_iter += 1
        else: # in a cluster
            current_id = line.split('>')[1].split('...')[0]
            current_index = ids_C.index(current_id) #current_index = [i for i, description in enumerate(ids_C) if current_id == description]
            cluster_indices.append(current_index)
            cluster_sequences.append(sequences_C[current_index])
            
    # construct MSA for each cluster
    print('Building MSAs...')
    # os.mkdir(data_dir+'/clustalo')
    for cluster in clusters_N:
        infile = data_dir+'/clusters/unknown_N_sequences_cluster_'+cluster+'.fasta'
        outfile = data_dir+'/clustalo/unknown_N_sequences_MSA_clst_'+cluster+'.sto'
        Nout, Nerr = clustalo_python(data_dir+'/clustalo', infile, outfile, out_format='st')
    for cluster in clusters_C:
        infile = data_dir+'/clusters/unknown_C_sequences_cluster_'+cluster+'.fasta'
        outfile = data_dir+'/clustalo/unknown_C_sequences_MSA_clst_'+cluster+'.sto'
        Cout, Cerr = clustalo_python(data_dir+'/clustalo', infile, outfile, out_format='st')
        
    # construct new HMM for each cluster
    print('Building HMMs...')
    # os.mkdir(data_dir+'/profiles')
    for cluster in clusters_N:
        outfile = data_dir+'/profiles/unknown_N_sequences_'+cluster+'.hmm'
        msafile = data_dir+'/clustalo/unknown_N_sequences_MSA_clst_'+cluster+'.sto'
        bout, berr = hmmbuild_python(hmm_path, outfile, msafile)
    for cluster in clusters_C:
        outfile = data_dir+'/profiles/unknown_C_sequences_'+cluster+'.hmm'
        msafile = data_dir+'/clustalo/unknown_C_sequences_MSA_clst_'+cluster+'.sto'
        bout, berr = hmmbuild_python(hmm_path, outfile, msafile)

    print('Done.')
    return


def RBP_collection(MillardLab_tsv, MillardLab_genbank, directory):
    """
    This function loops over the filtered and unfiltered phage genome data of MillardLab to construct a collection of 
    (unfiltered) RBP and nonRBP sequence records for further analysis.
    
    Input:
    - MillardLab_tsv: handle (string) to the filtered tsv file of MillardLab phage genomes
    - MillardLab_genbank: handle (string) to the unfiltered phage genomes downloaded from NCBI by MillardLab
    - directory: string of location to store the sequence file at.

    Output: Dataframes of RBPs and nonRBPs
    """

    # load input data & make dict
    tsv_data = pd.read_csv(MillardLab_tsv, sep='\t')
    records = SeqIO.parse(MillardLab_genbank, 'gb')
    rbp_dict = {'phage_id':[], 'protein_id': [], 'Organism': [], 'Host':[], 'ProteinName': [], 'ProteinSeq': [], 
                'DNASeq': [], 'RecordDate': []}
    nonrbp_dict = {'phage_id':[], 'protein_id': [], 'Organism': [], 'Host':[], 'ProteinName': [], 'ProteinSeq': [], 
                'DNASeq': [], 'RecordDate': []}
    rbp_re = r'tail.?(?:spike|fib(?:er|re))|^recept(?:o|e)r.?(?:binding|recognizing).*(?:protein)?|^RBP'

    # loop over all records
    stop_iter = 0
    while stop_iter >= 0:
        try:
            record = next(records)
            rindex = list(tsv_data['Accession']).index(record.name)
            record_realm = tsv_data['Realm'][rindex]
            # if in filtered tsv data and is phage, check the CDSs
            if (record.name in list(tsv_data['Accession'])) and (record_realm in ['Duplodnaviria', 'Unclassified']):
                host = '-'
                if 'host' in record.features[0].qualifiers:
                    host = record.features[0].qualifiers['host'][0]
                elif 'lab_host' in record.features[0].qualifiers:
                    host = record.features[0].qualifiers['lab_host'][0]
                elif 'strain' in record.features[0].qualifiers:
                    host = record.features[0].qualifiers['strain'][0]
                org = record.annotations['organism']
                date = record.annotations['date']
                
                # look for the CDSs and get their infos
                for feature in record.features:
                    if feature.type == 'CDS':
                        try:
                            pname = feature.qualifiers['product'][0]
                            pseq = feature.qualifiers['translation'][0]
                            dnaseq = str(feature.location.extract(record).seq)
                            pid = feature.qualifiers['protein_id'][0]
                            
                            # collect in RBPs or nonRBPs
                            if re.search(rbp_re, pname, re.IGNORECASE) is not None:
                                rbp_dict['phage_id'].append(record.name)
                                rbp_dict['protein_id'].append(pid)
                                rbp_dict['Organism'].append(org)
                                rbp_dict['Host'].append(host)
                                rbp_dict['ProteinName'].append(pname)
                                rbp_dict['ProteinSeq'].append(pseq)
                                rbp_dict['DNASeq'].append(dnaseq)
                                rbp_dict['RecordDate'].append(date)
                            else:
                                nonrbp_dict['phage_id'].append(record.name)
                                nonrbp_dict['protein_id'].append(pid)
                                nonrbp_dict['Organism'].append(org)
                                nonrbp_dict['Host'].append(host)
                                nonrbp_dict['ProteinName'].append(pname)
                                nonrbp_dict['ProteinSeq'].append(pseq)
                                nonrbp_dict['DNASeq'].append(dnaseq)
                                nonrbp_dict['RecordDate'].append(date)    
                        except KeyError:
                            pass
                        
            stop_iter += 1         
            if stop_iter%1000 == 0:
                print('iteration:', stop_iter)
        except StopIteration:
            stop_iter = -1
        except:
            pass
    
    # make dataframe and save
    annotated_rbps = pd.DataFrame(data=rbp_dict)
    annotated_nonrbps = pd.DataFrame(data=nonrbp_dict)
    annotated_rbps.to_csv(directory+'/annotated_RBPs_unfiltered.csv', index=False)
    annotated_nonrbps.to_csv(directory+'/annotated_nonRBPs_unfiltered.csv', index=False)
    print('Wrote RBP and nonRBP databases to directory.')

    return


def RBP_filters(RBPs_unfiltered, nonRBPs_unfiltered, directory, timestamp):
    """
    This function applies several data processing filters to both the annotated RBP and annotated nonRBP set.
    
    Inputs:
    - RBPs_unfiltered: unfiltered annotated RBPs, DataFrame
    - nonRBPs_unfiltered: unfiltered annotated nonRBPs, DataFrame
    - directory: to store output files in
    - timestamp: current month/year for saving (e.g. '2020-01')
    
    Output: filtered RBP and nonRBP databases
    """
    to_delete_rbps = []
    to_delete_nonrbps = []
    keywords = ['adaptor','wedge','baseplate','hinge','connector','structural','component',
                'assembly','chaperone','attachment','capsid','proximal','measure']
    hypotheticals = ['probable','probably','uncharacterized','uncharacterised','putative',
                     'hypothetical','unknown','predicted']
    
    # loop over RBPs
    for i, rbpseq in enumerate(RBPs_unfiltered['ProteinSeq']):
        rbpname = RBPs_unfiltered['ProteinName'][i]
        
        # filter unknown AAs
        if re.search(r'[^ACDEFGHIKLMNPQRSTVWY]', rbpseq) is not None:
            to_delete_rbps.append(i)
        # filter keywords
        if any(key in rbpname.lower() for key in keywords):
            to_delete_rbps.append(i)
        # filter length
        if (len(rbpseq) < 200) or (len(rbpseq) > 2000):
            to_delete_rbps.append(i)
            
    # loop over nonRBPs
    for i, nonrbpseq in enumerate(nonRBPs_unfiltered['ProteinSeq']):
        nonrbpname = nonRBPs_unfiltered['ProteinName'][i]
        
        # filter unknown AAs
        if re.search(r'[^ACDEFGHIKLMNPQRSTVWY]', nonrbpseq) is not None:
            to_delete_nonrbps.append(i)
        # filter hypotheticals
        if any(hyp in nonrbpname.lower() for hyp in hypotheticals):
            to_delete_nonrbps.append(i)
        # filter length
        if len(nonrbpseq) < 30:
            to_delete_nonrbps.append(i)
            
    # delete
    to_delete_rbps = list(set(to_delete_rbps))
    to_delete_nonrbps = list(set(to_delete_nonrbps))
    RBPs = RBPs_unfiltered.drop(to_delete_rbps)
    RBPs = RBPs.reset_index(drop=True)
    nonRBPs = nonRBPs_unfiltered.drop(to_delete_nonrbps)
    nonRBPs = nonRBPs.reset_index(drop=True)
    
    # filter identicals
    RBPs.drop_duplicates(subset = ['ProteinSeq'], inplace = True)
    nonRBPs.drop_duplicates(subset = ['ProteinSeq'], inplace = True)
    
    # filter dubious ones (RBP-nonRBPs identicals)
    to_delete_dubiousRBPs = [i for i, sequence in enumerate(RBPs['ProteinSeq']) if sequence in nonRBPs['ProteinSeq']]
    to_delete_dubiousnonRBPs = [i for i, sequence in enumerate(nonRBPs['ProteinSeq']) if sequence in RBPs['ProteinSeq']]
    RBPs = RBPs.drop(to_delete_dubiousRBPs)
    RBPs = RBPs.reset_index(drop=True)
    nonRBPs = nonRBPs.drop(to_delete_dubiousnonRBPs)
    nonRBPs = nonRBPs.reset_index(drop=True)
    print(RBPs.shape, nonRBPs.shape)
    
    # save new databases
    RBPs.to_csv(directory+'/annotated_RBPs_'+timestamp+'.csv', index=False)
    nonRBPs.to_csv(directory+'/annotated_nonRBPs_'+timestamp+'.csv', index=False)
    print('Wrote filtered databases to directory.')
    
    return


def phanns_predict(sequences_df, phanns_dir, results_dir, suffix=''):
    """
    This function predicts the class of a bunch of sequences using PhANNs models (cfr. Cantu et al., 2020)
    
    Input:
    - sequences_df: a dataframe with protein sequences (ProteinSeq) and corresponding ids (protein_id)
        to make predictions for.
    - phanns_dir: the directory of the web_server module of the PhANNs repository
    """
    
    # save all the sequences as separate fastas in the /uploads directory of PhANNs
    for i, sequence in enumerate(sequences_df['ProteinSeq']):
        # write FASTA file in directory
        this_id = list(sequences_df['protein_id'])[i]
        fasta = open(phanns_dir+'/uploads/'+this_id+'.fasta', 'w')
        fasta.write('>'+this_id+'\n'+sequence+'\n')
        fasta.close()
    
    # run PhANNs predictions
    cd_str = 'cd ' + phanns_dir # change directory
    raw_str = 'python run_server_once.py'
    command = cd_str+'; '+ raw_str
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    
    # save results to a dataframe for later comparisons
    print('Getting results...')
    phanns_preds = []; phanns_score = []
    for this_id in sequences_df['protein_id']:
        this_result = pd.read_csv(phanns_dir+'/csv_saves/'+this_id+'.csv', index_col=0)
        this_class = this_result.idxmax(axis=1)[0]
        if this_class == 'Tail fiber':
            phanns_preds.append(1)
        else:
            phanns_preds.append(0)
        phanns_score.append(this_result['Confidence'][0])
        
    phanns_df = pd.DataFrame({'preds': phanns_preds, 'score': phanns_score})
    phanns_df.to_csv(results_dir+'/phanns_predictions'+suffix+'.csv', index=False)
    print('Done.')
    
    return


def protein_architecture_plot(sequences, domains, locations, label_dict=[], count_threshold=0, save_fig=False):
    """
    Plots the different architectures (combinations of modules) for a given
    set of proteins, domains and their locations.

    Input:
    - sequences: list of protein sequences to plot
    - domains: list of lists with the domain names for each protein
    - locations: list of lists of tuples with the location of each corresponding domain
    - label_dict: optional dict with categories for labels {labelx: [domain1, ...], labely: [...], ...}
    - count_threshold: threshold under which not to plot the domains, based on the number of occurrences
    - save_fig: option to save the figure
    """
    # initiations
    y_place = 0
    protein_lengths = [round(len(x)) for x in sequences]
    unique_combos = [list(x) for x in set(tuple(x) for x in domains)] # get unique combos
    domain_counts = [domains.count(x) for x in unique_combos] # count unique combos
    sorted_unique_combos = [(x,y) for y, x in sorted(zip(domain_counts, unique_combos))] # sort
    sorted_unique_combos = [combo for combo in sorted_unique_combos if combo[1] > count_threshold] # delete under thres

    # give all unique domains or labels a separate color
    merged_domains = [dom for current_domains in sorted_unique_combos for dom in current_domains[0]]
    unique_domains = list(set(merged_domains))
    if len(label_dict) > 0:
        for key in label_dict.keys():
            new_domains = [value for value in label_dict[key] if value in unique_domains]
            label_dict[key] = new_domains
        cmap_names = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']
        colors_per_label = [plt.get_cmap(cmap_names[i])(np.linspace(0.8,0.4,len(label_dict[label]))) 
                                                  for i, label in enumerate(list(label_dict.keys()))]
        colors_dict = {}
        for i, unique_label in enumerate(list(label_dict.keys())):
            for j, domain in enumerate(label_dict[unique_label]):
                colors_dict[domain] = colors_per_label[i][j]
    else:
        cmap = plt.cm.turbo(np.linspace(0.0, 1.0, len(unique_domains)))
        colors_dict = dict([(dom, cmap[i]) for i, dom in enumerate(unique_domains)])

    # set up plot and params
    y_count = max(5, int(len(sorted_unique_combos)/4))
    y_box = int(y_count*0.6)
    x_count = min(200, max(protein_lengths))
    x_legend = min(800, max(protein_lengths))
    fig, ax = plt.subplots(figsize=(8,y_count))

    # loop over unique combos and plot
    protein_lengths = []
    for i, current in enumerate(sorted_unique_combos):
        current_domains = current[0]
        current_count = current[1]
        y_place += y_count
        index = domains.index(current_domains)
        current_protein = sequences[index]
        current_locations = locations[index]
        backbone_length = round(len(current_protein))
        protein_lengths.append(backbone_length)

        # plot backbone
        backbone = plt.Rectangle((x_count, y_place), backbone_length, y_count*0.1, fc='grey')
        ax.add_patch(backbone)
        ax.annotate(str(current_count), xy=(1, y_place-(y_box/2)))

        # loop over domains
        for j, dom in enumerate(current_domains):
            # plot each domain at correct location
            loc = current_locations[j]

            if len(label_dict) > 0:
                current_label = [key for key, value in label_dict.items() if (dom in value)][0]
                current_color = colors_dict[dom]
            else:
                current_label = dom
                current_color = colors_dict[dom]
            patch = mpatch.FancyBboxPatch((x_count+loc[0], y_place-(y_box/2)), loc[1]-loc[0], y_box, 
                boxstyle='Round, pad=0.2, rounding_size=0.8', fc=current_color, label=current_label)
            ax.add_patch(patch)
    ax.set_xlim(0, x_count+max(protein_lengths) +x_legend/4)

    ax.set_ylim(0, y_place+max(10,y_count))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys())
    ax.axis('off')
    #ax.set_title('Protein domain architectures', size=14)
    for i, label in enumerate(list(label_dict.keys())):
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(),cmap=plt.get_cmap(cmap_names[i])), 
                     ax=ax, fraction=0.03+0.0015*i, pad=0.02)
        cbar.ax.set_ylabel(label, rotation=270, labelpad=-2.5)
        cbar.ax.get_yaxis().set_ticks([])
    #fig.tight_layout()

    if save_fig:
        fig.savefig('protein_architecture_plot.png', dpi=400)

    fig.show()

    return sorted_unique_combos


# 2 - EXAMPLE
# --------------------------------------------------
# define paths and press new database
#path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
#dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
#new_pfam = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A_extended.hmm'
#output, err = hmmpress_python(path, new_pfam)

#genes_df = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase_df.csv', index_col=0)
#unique_genes = genes_df.columns
#phage_genomes = genes_df.index

# redo the all_domains scan to update the json files (takes a long time!)
#prodomains_dict = all_domains_scan(path, new_pfam, list(unique_genes))
#domaindump = json.dumps(prodomains_dict)
#domfile = open(dom_path+'/prodomains_dict.json', 'w')
#domfile.write(domaindump)
#domfile.close()

# define updated domain lists (structs, bindings, chaps) & make predictions
#dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
#N_blocks = ['Phage_T7_tail', 'Tail_spike_N', 'Prophage_tail', 'BppU_N', 'Mtd_N', 
#           'Head_binding', 'DUF3751', 'End_N_terminal', 'phage_tail_N', 'Prophage_tailD1', 
#           'DUF2163', 'Phage_fiber_2', 'phage_RBP_N1', 'phage_RBP_N4', 'phage_RBP_N26', 
#           'phage_RBP_N28', 'phage_RBP_N34', 'phage_RBP_N45']
#C_blocks = ['Lipase_GDSL_2', 'Pectate_lyase_3', 'gp37_C', 'Beta_helix', 'Gp58', 'End_beta_propel', 
#            'End_tail_spike', 'End_beta_barrel', 'PhageP22-tail', 'Phage_spike_2', 
#            'gp12-short_mid', 'Collar', 'phage_RBP_C2', 'phage_RBP_C10', 'phage_RBP_C24',
#            'phage_RBP_C43', 'phage_RBP_C59', 'phage_RBP_C60', 'phage_RBP_C62', 'phage_RBP_C67',
#            'phage_RBP_C79', 'phage_RBP_C97', 'phage_RBP_C111', 'phage_RBP_C115', 'phage_RBP_C120'
#            'phage_RBP_C126', 'phage_RBP_C138', 'phage_RBP_C43', 'phage_RBP_C157', 'phage_RBP_C164', 
#            'phage_RBP_C175', 'phage_RBP_C180', 'phage_RBP_C205', 'phage_RBP_C217', 'phage_RBP_C220', 
#            'phage_RBP_C221', 'phage_RBP_C223', 'phage_RBP_C234', 'phage_RBP_C235', 'phage_RBP_C237',
#            'phage_RBP_C249', 'phage_RBP_C259', 'phage_RBP_C267', 'phage_RBP_C271', 'phage_RBP_C277',
#            'phage_RBP_C281', 'phage_RBP_C292', 'phage_RBP_C293', 'phage_RBP_C296', 'phage_RBP_C300', 
#            'phage_RBP_C301', 'phage_RBP_C319', 'phage_RBP_C320', 'phage_RBP_C321', 'phage_RBP_C326', 
#            'phage_RBP_C331', 'phage_RBP_C337', 'phage_RBP_C338', 'phage_RBP_C340', 'Peptidase_S74', 
#            'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']

#domfile_pro = open(dom_path+'/prodomains_dict.json')
#prodomains_dict = json.load(domfile_pro)
#preds = RBP_domain_predict(path, new_pfam, prodomains_dict, structs, bindings, chaps, archive)
#preds_df = pd.DataFrame.from_dict(preds)
#preds_df.iloc[:50, 1:]