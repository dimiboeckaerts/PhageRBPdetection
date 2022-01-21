"""
DOMAIN-BASED RBP DETECTION

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

import phagebase_utils as pbu

    
# 1 - BASIC TESTS
# --------------------------------------------------
"""
STATUS: DONE

In this section, the basic functionalities are tested.

Qs & Remarks:
    - file names CANNOT contain spaces (e.g. Google Drive), this will mess up
        the cmd line execution.
    - cannot loop over result anymore after hmmsearch... strange.
    - if we don't need to use the GeneBase_df, then why do we need it? GeneBase_df
        only contains the unique genes, while GeneBase contains all the genes.
        GeneBase_df is a df with zeros or one corresponding to which gene (column)
        is present in what genome (row).
"""

# file names
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'
sequence_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/KP32B_s.fasta'
sequences_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/KP32B.fasta'
selected_hmm = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7tail.hmm'
Cterm_domains_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/c_term_domains.hmm'


# test1: press a db
output, err = pbu.hmmpress_python(path, pfam_db)

# test2: scan a single sequence
_, _, scan_df = pbu.single_hmmscan_python(path, pfam_db, sequence_file)

# test3: get domains    
doms, scores, biases, ranges = pbu.hmmscan_python(path, pfam_db, sequences_file)

# test4: selected profile in sequence database
hits, scores, bias, aln_range = pbu.hmmsearch_python(path, selected_hmm, sequences_file)

# test5: selected profile in single sequence file
hits, scores, bias, aln_range = pbu.hmmsearch_python(path, selected_hmm, sequence_file)

# test5b: multiple profiles, single sequence
hits, scores, bias, aln_range = pbu.hmmsearch_python(path, Cterm_domains_file, sequence_file)
hits, desc, scores, bias, aln_range = pbu.hmmsearch_python(path, Cterm_domains_file, sequences_file)


# test6: selected profile without a hit in single sequence file
rand_protein = pbu.random_protein(length=456)
random_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/randomsequence.fasta'

f = open(random_file, 'w')
f.write('>randomized_protein'+'\n'+rand_protein+'\n')
f.close()

hits, scores, bias, aln_range = pbu.hmmsearch_python(path, selected_hmm, random_file) # hits = empty

# test x: graphs
B = nx.Graph()
B.add_nodes_from([1,2,3,4], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(['a','b','c'], bipartite=1)
B.add_edges_from([(1,'a'), (1,'b'), (2,'b'), (2,'c'), (3,'c'), (4,'a')])
color_map = []
for node in B:
    if (node == 'a') or (node == 'b'):
        color_map.append('green')
    else:
        color_map.append('blue')
nx.draw(B, node_color=color_map)
plt.show()

top = nx.bipartite.sets(B)[0]
pos = nx.bipartite_layout(B, top)
nx.draw(B, pos=pos)

X, Y = nx.bipartite.sets(B)
#pos = dict()
#pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
#pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
plt.figure(figsize=(7, 8))
plt.title('This is a graph')
nx.draw(B, pos=pos, with_labels=False)
for p in pos:  # raise text positions
    pos[p][1] += 0.05
nx.draw_networkx_labels(B, pos)
plt.show()

# test x: graohs with dict
test_dict = {1: ['a', 'b'], 2: ['b', 'c'], 3: ['c']}
dict_values = list(set([item for sub in list(test_dict.values()) for item in sub]))
B = nx.Graph()
B.add_nodes_from(list(test_dict.keys()), bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(dict_values, bipartite=1)
edge_list = []
for key in test_dict.keys():
    for value in list(test_dict[key]):
        edge_list.append((key, value))
B.add_edges_from(edge_list)
top = nx.bipartite.sets(B)[0]

pbu.graph_dictionary(test_dict)


# 2 - DETECTING PHAGE_T7_TAIL
# --------------------------------------------------
"""
STATUS: DONE

Idea is to loop over every genome of PhageBase (as genes), and see which of the
genes have a hit for the T7_tail domain. These genes should be saved:
- we need to know what genome they were detected in (to also know in which
    genomes we didn't detect anything)
- we need to know the gene sequence itself, to then, in a second step,
    detect what is at the C-terminal end of those sequences.
        
Qs & remarks
- the entire thing should take about 55 minutes to run.
- How much hits do we expect to have? At least < 3532, the number of genomes
- Maybe the e-value threshold is too low?

Results:
- E-value < 0.001: 67 gene hits in 148 genomes
- bitscore >= 50: 2 gene hits in 8 genomes
- bitscore >= 25: 30 gene hits in 58 genomes
"""
# load the dataframe & set paths
genes_df = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase_df.csv', index_col=0)
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
T7_tail_hmm = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7tail.hmm'

# get the relevant data
unique_genes = genes_df.columns
phage_genomes = genes_df.index

# STEP 1: loop over the unique genes and search for hits to T7_tail
T7_genome_hits = []
T7_gene_hits = []
bar = tqdm(total=len(unique_genes), position=0, leave=True)

for i, gene in enumerate(unique_genes):
    sequence_hits = pbu.gene_domain_search(path, T7_tail_hmm, gene)
    
    # check the corresponding genome
    if len(sequence_hits) != 0:
        genome_index = genes_df.iloc[:,i] != 0
        genome_hits = list(phage_genomes[genome_index])
        
        # save the genome hit
        for name in genome_hits:
            T7_genome_hits.append(name)
        
        # save the gene hit
        T7_gene_hits.append(gene)
    
    # update bar
    bar.update(1)
bar.close()
none_detected = [genome for genome in phage_genomes if genome not in T7_genome_hits]

# check how much we detected
print('number of genes detected: ', len(T7_gene_hits))
print('number of genomes detected in: ', len(T7_genome_hits))

# save hits to lists
gene_hits_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7_gene_hits.txt'
file_hits = open(gene_hits_file, 'w')
for hit in T7_gene_hits:
    file_hits.write(hit+'\n')
file_hits.close()
genome_hits_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7_genome_hits.txt'
file_hits = open(genome_hits_file, 'w')
for hit in T7_genome_hits:
    file_hits.write(hit+'\n')
file_hits.close()


# 3 - LOOKING AT THE C-TERM END OF HITS
# --------------------------------------------------
"""
STATUS: DONE

STEP 2: 
With the hits we found in the previous section, we now want to look at 
what other domains are present in these sequences. As we already know what is 
at the N-terminus of these hits, we're now actually looking at the C-terminus 
of the sequences. This is a HMMscan.
- E-value < 0.001: 'Tail_spike_N', 'Lipase_GDSL_2', 'Pectate_lyase_3', 
    'Beta_helix', 'Peptidase_S74', 'YadA_stalk', 'YadA_head', 'ESPR', 
    'YadA_anchor', 'Apolipoprotein', 'DUF1664', 'Filament', 'UPF0242'
- bitscore >= 25: 'Lipase_GDSL_2', 'Tail_spike_N', 'Beta_helix', 'YadA_stalk', 
    'YadA_head', 'ESPR', 'YadA_anchor', 'Apolipoprotein', 'DUF1664'.


STEP 3: 
Now the newly found (C-terminal) domains can be searched (HMMsearch) for
again in the entire database, and the question is how many new hits we find AND
in how many genomes these hits are present. Also, we expect the C-terminus to be
species specific, so we should find more sequences within the same species. On
the other hand, cross-species C-terminal domain hits may indicate an HGT.
- E-value < 0.001: 1264 gene hits in 3419 genomes
- bitscore >= 25: 294 gene hits in 628 genomes
This ofcourse includes the genes and genomes we had in step 2 (because we look
over all of the genes again).


STEP 4: 
what is N-terminally present in these genes?
- bitscore >= 25: 'Prophage_tail', 'BppU_N', 'Lipase_GDSL_2', 'Tail_spike_N',
     'Phage_T7_tail', 'Beta_helix', 'DUF3751', 'DnaB', 'Lipase_GDSL'

Qs & remarks:
- do we want to check one sequence at a time, or just do all hits together?
- MAKE OUTPUT of each step in the iterative process more consistent and traceable.
- Are all of these hits RBPs?? Were are they located on each of the genomes? (PLOT
or location?). 
"""
# paths
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'

# files step 1
gene_hits_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7_gene_hits.txt'
genome_hits_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7_genome_hits.txt'
f_genes = open(gene_hits_file)
f_genomes = open(genome_hits_file)
T7_gene_hits = [genehit[:-1] for genehit in f_genes]
T7_genome_hits = [genomehit[:-1] for genomehit in f_genomes]

# test7: test with unique genes subset
T7_genes = unique_genes[:200]
T7_detections = {}
bar = tqdm(total=len(T7_genes), position=0, leave=True)
for i, gene_hit in enumerate(T7_genes):
    # get the other domains in each sequence
    domains = pbu.gene_domain_scan(path, pfam_db, [gene_hit], 'Phage_T7_tail')
    
    # make dictionary for processing
    for dom in domains:
        if dom in T7_detections.keys():
            T7_detections[dom].append(gene_hit)
        else:
            T7_detections[dom] = [gene_hit]
    
    # update bar
    bar.update(1)
bar.close()

# STEP 2: go for all T7 hits -> detect all domains but the T7 domain itself 
# (because we know that one is N-terminally located).
T7_detections = {}
bar = tqdm(total=len(T7_gene_hits), position=0, leave=True)
for i, gene_hit in enumerate(T7_gene_hits):
    # get the other domains in each sequence
    domains = pbu.gene_domain_scan(path, pfam_db, [gene_hit], 'Phage_T7_tail')
    
    # make dictionary for processing
    for dom in domains:
        if dom in T7_detections.keys():
            T7_detections[dom].append(gene_hit)
        else:
            T7_detections[dom] = [gene_hit]
    
    # update bar
    bar.update(1)
bar.close()

# STEP 3: fetch those domains and search them again in the entire database
Cterm_domains_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/c_term_domains.hmm'
Cterm_selection = ['Lipase_GDSL_2', 'Tail_spike_N', 'Beta_helix']
#output, error = hmmfetch_python(path, pfam_db, list(T7_detections.keys()), Cterm_domains_file)
output, error = pbu.hmmfetch_python(path, pfam_db, Cterm_selection, Cterm_domains_file)
bar = tqdm(total=len(unique_genes), position=0, leave=True)

Cterm_scan = {}
Cterm_genome_hits = []
for i, gene in enumerate(unique_genes):
    sequence_hits = pbu.gene_domain_search(path, Cterm_domains_file, gene)
    
    # check the corresponding genome
    if len(sequence_hits) != 0:
        # build the dictionary of gene-domain hits
        Cterm_scan[gene] = sequence_hits
        
        # get the genomes to look for cross-species
        genome_index = genes_df.iloc[:,i] != 0
        genome_hits = list(phage_genomes[genome_index])
        
        # save the genome hit
        for name in genome_hits:
            Cterm_genome_hits.append(name)
    
    # update bar
    bar.update(1)
bar.close()

# STEP 4: finally, check the N-terminus of new hits!
Nterm_scan = {}
bar = tqdm(total=len(Cterm_scan.keys()), position=0, leave=True)
for i, gene_hit in enumerate(Cterm_scan.keys()):
    # get the other domains in each sequence
    domains = pbu.gene_domain_scan(path, pfam_db, [gene_hit])
    
    # make dictionary for processing
    for dom in domains:
        if dom in Nterm_scan.keys():
            Nterm_scan[dom].append(gene_hit)
        else:
            Nterm_scan[dom] = [gene_hit]
    
    # update bar
    bar.update(1)
bar.close()

counts = []
for domain in Nterm_scan.keys():
    counts.append(len(Nterm_scan[domain]))
    if len(Nterm_scan[domain]) > 10:
        print(len(Nterm_scan[domain]), domain)
plt.hist(counts, bins=20)


# 4 - ITERATIVE DISCOVERY PROPHAGE RBPs
# --------------------------------------------------
"""
STATUS: DONE

Look N-terminal, check the hits for C-terminal domains and then search those 
domains and go back to N-term. This means that in the N-terminal iteration,
we want to filter for domains in the first e.g. 250 AAs of the sequence.

Qs & remarks:
- do we want to check one sequence at a time, or just do all hits together?
- However, when giving up the T7tail.hmm manually (not fetching it first), the
search is slower again (35s versus 54s per iteration). Now it does find 5 hits!
Which is weird, because should be exactly the same HMM. But still, previously
we found 30 hits, so the fact that N-term is trimmed does seem to matter. Maybe
adjust the bitscore threshold for the length of the cut. After iteration 0, it
stopped, didn't find anything N-terminally anymore, but this is due to the domain hmm
we manually replaced! It scans for T7_tail again, but ofcourse there are none of
those left. Should repeat without it.
- 21/10/20: HMMfetch does seem the be the problem (see test9, repeat with and 
    without fetching first). FIXED: HMMfetch needs to get a list of domains, not
    just the string!
- 21/10/20: the scores with N-trimmed are actually a little higher! the shorter
    the sequence, the less noise if it's actually that domain that's there, so
    higher score?!
- 22/10/20: enige bezorgdheid -> geeft 1 enkel domein al dan niet uitsluitsel 
    over of een eiwit al dan niet een RBP is? Of moeten we nog steeds gaan kijken
    naar voorkomen van meerdere domeinen in hetzelfde eiwit (of bijv. bepaalde
    welgekende domeinen als belangrijker beschouwen, anderen discarden?).
- 04/12/12: after redoing the iterative discovery and keeping scores, bias and 
    aln_range in a dictionary, now we can start looking at those in more detail
    and decide on an appropriate cutoff. Previously, a cutoff for score/bias 
    greater than 10 was chosen together with aln_range cutoff of 125. But this was
    way to strict (discovered only 148 genes in 2 iterations). After looking at the
    biases and ranges, a new proposal is to set the score/bias cutoff at 4 or 5
    and the aln_range cutoff at 50 (you still need both, sometimes good range but
    also an enormous bias). The high biases and short ranges seem to appear both
    at the N and C terminus. A ratio > 5 had 1748 hits (out of 54731).
    
    RATIO >= 5 (range>=50): 720 genes - 3 iterations (1794 Nterm_d, 10544 Cterm_d)
    RATIO >= 4 (range>=50): 722 genes - 3 iterations (1806 Nterm_d, 10642 Cterm_d)
    RATIO >= 3 (range>=50): 722 genes - 3 iterations (1823 Nterm_d, 10729 Cterm_d)
    It seems as if ratio doesn't have a very disruptive effect on the detected
    genes, but aln_range probably has. Let's check it.
    
    ratio >= 4 / range >= 40: 1945 genes - 10 iterations (1906 Nterm_d, 11232 Cterm_d)
    ratio >= 4 / range >= 35: 2370 genes - 10 iterations (1942 Nterm_d, 11400 Cterm_d)
    ratio >= 3 / range >= 35: 2509 genes - 10 iterations (1962 Nterm_d, 11489 Cterm_d)

    Can we make a data-driven solution on a cutoff for aln_range? Plot histo?
    A histogram with 150 bins shows a maximum of 4179 occurences in the bin with
    a mean aln_range of 45.48. As the maximum resides there, we lose a lot of 
    sequences when setting the threshold above that.
    We'll take the slightly lower cutoffs to work with for two reasons: (1) We'll
    have a validation afterwards anyway and (2) we know phage RBP are among the 
    most frequently mutated genes in the genome so it makes sense to allow for 
    somewhat more bias and/or shorter alignment lengths.
    
- 07/12/20:
    Let's implement the block list and see how many hits we get now.
    ratio >= 3 / range >= 35: 627 genes - 3 iterations
    Tail_spike_N still appears at the C-terminus. In total, 11 C-terminal hits
    of TSPN appear in the data. All of them have a low bias and an alignment range
    of 57 or 61 AAs.
    It's also weird that we don't find any of the 'more typical' RBP domains that
    we found before. This is a second reason to perhaps further decrease the alignment
    length.
    
    REPEAT WITHOUT THRESHOLD FOR ALN LENGTH?
- 08/12/20:
    A second smaller peak is seen at around 750 for a cutoff of 17.68 on a histogram
    with 200 bins. To also include this smaller peak, a threshold is set at 15.
    ratio >= 3 / range >= 15: 1017 genes - 4 iterations
    
    If we decrease the range threshold al the way down to zero:
    ratio >= 3 / range >= 0: 1023 genes - 4 iterations
    Which is not a lot more, likely because all of those low-scoring domains are
    domains we don't expect in RBPs anyway and are in the blocking list. Thus,
    with a validation afterwards in mind, and to simplify things, we will simply
    skip the aln_range threshold and continue working with these 1023 RBP hits.

- 09/12/20:
    One other option would be to, instead of looking at aln_range (you can actually
    also look at enveloppe length) to look at the score of the single best scoring
    domain. If that domain is also significant, you keep the domain.
    The bias is a correction term for biased sequence composition that was applied 
    to the sequence bit score. Sometimes the bias can be too aggressive, correcting
    for too much and causing you to miss remote homologs.
    
    Barplot: In general, combinations of N/C terminals occur only sporadically 
    (174 times out of the 1022 detections). 918 sequences have only a single domain
    (N terminal or C terminal). 918+174 = 1092 which means some sequences do have
    more than 2 domains in their sequences (and are counted double in the 174 hits)
    This means there's still lot of room for improvement to build new HMMs. 
"""

## TESTS
## ----------
# test8: RBP iterative detect on 5000 genes subset
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
genes_sub = unique_genes[:1000]
detections, dom_dict, Nterm_doms = pbu.RBP_NCiterations(path, dom_path, pfam_db, genes_sub, 'Phage_T7_tail', max_iter=100)

# test9: KP32 with and without N-trim
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'
sequence_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/KP32B_s.fasta'
sequences_file = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/KP32B.fasta'
selected_hmm = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7tail.hmm'
fetch_res, err = pbu.hmmfetch_python(path, pfam_db, ['Phage_T7_tail'], dom_path+'/Ntermdomain.hmm')
fasta200 = open(dom_path+'KP32B_200.fasta', 'w')
for record in SeqIO.parse(sequences_file, 'fasta'):
    fasta200.write('>'+record.id+'\n'+str(record.seq)[:200])
fasta200.close()
result, hits_list, scores = pbu.hmmsearch_python(path, T7_tail_hmm, sequences_file)
result_200, hits_list_200, scores_200 = pbu.hmmsearch_python(path, T7_tail_hmm, dom_path+'KP32B_200.fasta')

## DISCOVERY
## ----------
# load data
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'
T7_tail_hmm = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7tail.hmm'

# get the relevant data
genes_df = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase_df.csv', index_col=0)
unique_genes = genes_df.columns
phage_genomes = genes_df.index

# run on full dataset: make domain dictionaries (don't run twice)
domains_dictionary = pbu.all_domains_scan(path, pfam_db, unique_genes)
domaindump = json.dumps(domains_dictionary)
domfile = open(dom_path+'/alldomains_dict.json', 'w')
domfile.write(domaindump)
domfile.close()

# Load dictionaries & fill up
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
domfile_test = open(dom_path+'/alldomains_dict.json')
domains_dict = json.load(domfile_test)
Nterm_d = {}; Cterm_d = {}
bias_count = 0
for i,pos in enumerate(domains_dict['position']):
    OM_score = math.floor(math.log(domains_dict['score'][i], 10)) # order of magnitude
    OM_bias = math.floor(math.log(domains_dict['bias'][i]+0.00001, 10))
    aln_range = domains_dict['aln_range'][i]

    # fill up N- and C-dictionaries
    if (pos == 'N') & (OM_score > OM_bias):
        domain = domains_dict['domain_name'][i]
        if domain in Nterm_d.keys(): # if domain present, append gene
            Nterm_d[domain].append(domains_dict['gene_sequence'][i])
        else: # if not present, make new list
            Nterm_d[domain] = [domains_dict['gene_sequence'][i]]
    elif (pos == 'C') & (OM_score > OM_bias):
        gene = domains_dict['gene_sequence'][i]
        if gene in Cterm_d.keys(): # if gene present, append domain
            Cterm_d[gene].append(domains_dict['domain_name'][i])
        else: # if not, make new list
            Cterm_d[gene] = [domains_dict['domain_name'][i]]
            
    # assess domain quality
    if OM_score <= OM_bias:
        bias_count += 1
        print('domain: ', domains_dict['domain_name'][i])
        print('position: ', domains_dict['position'][i], 'score: ', domains_dict['score'][i],
              'bias: ', domains_dict['bias'][i], 'range: ', domains_dict['aln_range'][i])
        print()
print('total: ', bias_count)

# plot histogram of alignment lengths for cutoff (mode = 45.48, sign. drop at 34.36)
fig, ax = plt.subplots(figsize=(12,10))
ax.hist(domains_dict['aln_range'], bins=250)
ax.set_xlabel('domain alignment length')
ax.set_ylabel('number of occurences')
fig.savefig('domain_alignment_range_histogram200bins.png', dpi=400)

# run discovery
blocked = ['ESPR', 'YadA_head', 'Phage_lysozyme2', 'CHAP', 'Amidase_5', 'Metallophos', 'PTR', 
           'YadA_stalk', 'YadA_anchor', 'Apolipoprotein', 'DUF1664', 'Glucosamidase', 'SH3_5', 
           'LysM', 'Glyco_hydro_25', 'Amidase_2']
unique_detected, total_detected, detections, d_dict, Nterm_domain_list = pbu.RBP_NCiterations(path, dom_path, pfam_db, 
                Nterm_d, Cterm_d, 'Phage_T7_tail', block_list=blocked, max_iter=15)

## RESULTS
## ----------
# Plot detected gene numbers
fig, ax = plt.subplots(figsize=(12,10))
plt.plot(list(range(len(unique_detected))), unique_detected)
plt.xlabel('iteration')
plt.ylabel('number of RBPs detected')

# check Tail_spike_N at the C-terminus
count = 0
for sequence in Cterm_d.keys():
    if '' in Cterm_d[sequence]:
        print(sequence)
        count += 1
print(count)
for i, position in enumerate(domains_dict['position']):
    domain = domains_dict['domain_name'][i]
    if (position == 'C') & (domain == 'Tail_spike_N'):
        print(domains_dict['domain_name'][i])
        print(domains_dict['score'][i])
        print(domains_dict['bias'][i])
        print(domains_dict['aln_range'][i])
        
# How prevalent are the domains?
domain_counts = {}
Cterm_list = list(set([Cdom for sub in list(d_dict.values()) for Cdom in sub 
                       if Cdom not in blocked]))
domain_list = list(set(Cterm_list + list(d_dict.keys())))    
for domain in domain_list:
    domain_counts[domain] = 0
combo_list = []
single_count = 0
for gene in detections:
    # N-terminus
    Nterm_iter = []
    for Ndomain in domain_list:
        if Ndomain in Nterm_d.keys():
            if gene in Nterm_d[Ndomain]:
                domain_counts[Ndomain] += 1
                Nterm_iter.append(Ndomain)
    # C-terminus
    Cterm_iter = []
    if gene in Cterm_d.keys():
        for Cdomain in Cterm_d[gene]:
            if Cdomain in domain_list:
                domain_counts[Cdomain] += 1
                Cterm_iter.append(Cdomain)
    # Combos
    CL_iter = [(x,y) for x in Nterm_iter for y in Cterm_iter]
    if len(CL_iter) == 0:
        single_count += 1
    for combo in CL_iter:
        combo_list.append(combo)
              
# Bar plot individual domains
domcount_df = pd.DataFrame.from_dict(domain_counts, orient='index', columns=['count'])
domcount_df.index.name = 'domain'
domcount_df.reset_index(inplace=True)
domcount_df.sort_values(by=['count'], inplace=True, ascending=False)
fig, ax = plt.subplots(figsize=(12,10))
barpl = sns.barplot(x='count', y='domain', data=domcount_df)
fig.tight_layout()
fig.savefig(dom_path+'/barplot_detected_individual_domains.png', dpi=400)
        
# Bar plot combinations
combo_counts = {}
for item in combo_list:
    if item in combo_counts.keys():
        combo_counts[item] += 1
    else:
        combo_counts[item] = 1
combo_df = pd.DataFrame.from_dict(combo_counts, orient='index', columns=['count'])
combo_df.index.name = 'N/C combination'
combo_df.reset_index(inplace=True)
combo_df.sort_values(by=['count'], inplace=True, ascending=False)
fig, ax = plt.subplots(figsize=(12,10))
barpl = sns.barplot(x='count', y='N/C combination', data=combo_df)
fig.tight_layout()
fig.savefig(dom_path+'/barplot_domain_combos.png', dpi=400)

# How much in total? Without only considering unique genes!
phagegenebase = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase.csv')
total_RBP_detections = 0
for detected in detections:
    total_RBP_detections += list(phagegenebase['sequence']).count(detected)
    
# or (not noticeably faster...)
sum([list(phagegenebase['sequence']).count(detected) for detected in detections])


# 5 - ITERATIVE DISCOVERY PROPHAGE & MILL RBPs
# --------------------------------------------------
"""
STATUS: DONE

One of the reasons we find 'only' 1022 unique hits is because we're confined to
proghage data. However, we actually want to make an RBP-detection tool that is more
generally applicable than that (keeping in mind a PHALP-like database we want to
set up in the future). Thus, it makes sense to also include domains of lytic phages, 
which is why we turn to the MillardLab phage genome data.

Two options:
    - either run discovery on ALL genes, which has an immense size
    - or run discovery only on the 5000+ annotated RBPs that Alexander collected
We will start with the annotated RBPs and see where we end up.

4324 domains were detected in the 4966 annotated RBP genes.

11/01/21: Start implementing the new strategy: merge both dicts and make iterative
discovery based on the entire dict w/o necessatily taking into account the N- and
C-termini. After merging, we have a total of 59523 domains in all sequences
(these are unfiltered, filtering happens at the iterations step; cfr. RBP_iterations).

12/01: with all of these new domains, we might need a second cutoff on top of
the order of magnitude, in a first iteration, 100+ domains are found, which might
be overly optimistic... -> use block list as restriction

13/01: with the old discovery method applied to the full data, we now detect 1094
unique prophage RBPs, which is only a little more than if we wouldn't have had the data... (1022)
This indicates that adding the Millard RBPs to find new domains will not per se
aid in detecting more RBPs in the prophage data... However it does aid in developing
a more complete list of known building blocks which is important anyway.

Results are not consistent between both new and old discovery. You would assume that
by not restricting yourself to either looking only at N or C at one time, you would
find more domains, but we also miss some domains that were detected before but not
with the new method... in particular: Gp58, CE2_N and Phage_fiber_C after iteration 1.

15/01: found that the inconsistency is not one at all. Iterations are different in the
first version and updated version, so you can't compare iteration 1_old with 1_new. In the
iteration afterwards (2_new), the domains in question are found. 
The only thing we are left with at this point is the question of whether or not to also
include a restriction on aln_range.
Better to first update the block list and then see if we still find a lot of rubbish. And
anyway it won't really matter once we have our curated list. Better to find a little more
false positives now and weed them out manually.
Potentially, a cutoff of 15-20 is appropriate to automatically filter some false positives out.

19/01: after improving detecting (not limited to iteratively jumping from N to C)
and manually curating domains and expanding the blocked list, we now have 1706 unique
detections after 4 iterations (without cutoff for aln range), and a total of 4267
detections. This again proves that manually curating the domains does a good job 
(looking at the biology instead of making an arbitrary or even data-driven cutoff for aln range).

22/01: with the final blocked domains added to the list, we find a total of 1544 genes
in 4 iterations. No more new domains have been discovered throughout the iterations,
so this concludes the iterative discovery.
"""

## Data collection and processing
## -------------------------------
milrbps = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/MillardLab_RBPs.csv', sep='\t')
to_delete = [index for index, name in enumerate(milrbps['ProteinName']) if 'attachment' in name]
milrbps.drop(index=to_delete, inplace=True)
milrbps.reset_index(inplace=True)

## HMM scan
## -------------------------------
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'
mildomains_dict = pbu.all_domains_scan(path, pfam_db, list(milrbps['DNASeq']))

dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
domaindump = json.dumps(mildomains_dict)
domfile = open(dom_path+'/milldomains_dict.json', 'w')
domfile.write(domaindump)
domfile.close()

## Merge dicts
## -------------------------------
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
domfile_mil = open(dom_path+'/milldomains_dict.json')
mildomains_dict = json.load(domfile_mil)
domfile_pro = open(dom_path+'/alldomains_dict.json')
prodomains_dict = json.load(domfile_pro)

mildomains_dict['source'] = ['millard']*len(mildomains_dict['score'])
prodomains_dict['source'] = ['prophage']*len(prodomains_dict['score'])

alldomains_dict = prodomains_dict.copy()
for i, sequence in enumerate(mildomains_dict['gene_sequence']):
    alldomains_dict['gene_sequence'].append(sequence)
    alldomains_dict['domain_name'].append(mildomains_dict['domain_name'][i])
    alldomains_dict['position'].append(mildomains_dict['position'][i])
    alldomains_dict['score'].append(mildomains_dict['score'][i])
    alldomains_dict['bias'].append(mildomains_dict['bias'][i])
    alldomains_dict['aln_range'].append(mildomains_dict['aln_range'][i])
    alldomains_dict['source'].append(mildomains_dict['source'][i])

## Explore relevant possibe cutoffs
## -------------------------------
# plot histogram of alignment lengths for cutoff (mode = 46.87, sign. drop at 34.36)
fig, ax = plt.subplots(figsize=(12,10))
ax.hist(alldomains_dict['aln_range'], bins=250)
ax.set_xlabel('domain alignment length')
ax.set_ylabel('number of occurences')
fig.savefig('domain_alignment_range_histogram200bins.png', dpi=400)

## Run (updated) discovery
## -------------------------------
blocked = ['ESPR', 'YadA_head', 'Phage_lysozyme2', 'CHAP', 'Amidase_5', 'Metallophos',
'PTR', 'YadA_stalk', 'YadA_anchor', 'Apolipoprotein', 'DUF1664', 'Glucosamidase', 'SH3_5',
'LysM', 'Glyco_hydro_25', 'Amidase_2', 'BLOC1_2', 'Tox-WTIP', 'CarboxypepD_reg',
'Reprolysin_3', 'Reprolysin_5', 'Reprolysin_4', 'Lipase_GDSL', 'NAGPA', 'Laminin_G_3',
'C1q', 'BNR', 'Sortilin-Vps10', 'PSII_BNR', 'Tachylectin', 'BppL_N', 'Shufflon_N', 'TMP_2',
'Tape_meas_lam_C', 'Lebercilin', 'Glucosaminidase', 'SWM_repeat', 'SlyX', 'CC2-LZ', 'F5_F8_type_C',
'Tropomyosin', 'ILEI', 'fn3', 'Y_Y_Y', 'DUF1640', 'SLT', 'PhageMin_Tail', 'HisKA',
'NMT1', 'Lig_chan-Glu_bd', 'Muramidase', 'HATPase_c', 'Hpt', 'PAS_9', 'NMT1_3',
'EAL', 'DUF1906', 'PG_binding_3', 'PAS', 'NMT1_2', 'PAS_8', 'PAS_3', 'Response_reg',
'GGDEF', 'PAS_4', 'VPS13_C', 'SBP_bac_3', 'PG_binding_1', 'YqaJ', 'SLT_2', 'Phage_HK97_TLTM',
'DUF4183', 'DUF4369', 'Phage_T4_gp36', 'CE2_N', 'Phage-tail_3', 'DUF4815', 
'SF-assemblin', 'Phage_BR0599']
unique_detected, total_detected, genes_detected, domains_detected = pbu.RBP_iterations(alldomains_dict, 
        'Phage_T7_tail', block_list=blocked, max_iter=10)


# 7 - DIRECT DOMAIN-BASED PREDICTION - ROUND 1
# --------------------------------------------------
"""
STATUS: DONE

We've added domains to the struct, binding and chap list if we are convinced that
those domains, in phage genomes, only occur for RBPs. We also include an archive
list with domains that might occur in phage RBPs but not solely. Therefore, we
insist these domains to also occur with another domain of the struct, binding or
chap list before we consider it to be an RBP.

28/01: we find 1292 unique gene hits and 3200 in total that are predicted as
an RBP.

29/01: actually, we can also build new HMMs based on the MillardLab data...
Strangely, a domain is not found when it is searched for again in the prophage data.
Somehow, the domains_dict mentions a domain for a sequence while it is not found when
repeating the scan... double checked on pfam website and it shouldn't be there. So
rerunning the all_domains_scan for the prophage data and mildomains (which is also
good because we get to recompute aln_range properly). In the meantime, we can also
continue with clustering (as all hits are double checked).

From the barplot and the counts, we now see that 1871 sequences of the 3367 (including
the annotated RBPs of MillardLab), only have a (or multiple) known structural domains
without any binding/chaperone domain or the other way around.

Looking at the clusters manually:
    - we see that most clusters are formed within groups of sequences that also
        share the known domain at the other end (structural or binding). Very few
        sequences are clustered together based on their N-terminal or C-terminal
        end that share two different domains at the other end.
    - we also see some peculiarities; in particular with the binding domain 'Collar'.
        Sometimes, this domain is detected so far in the sequence that we don't
        only cut the structural (N-terminal part) but actually just most of the entire
        sequence (also the binding part), which is not wanted. E.g. some sequence cuts
        are still 1683 or 778 AAs long.
Maybe good idea to limit to clusters residing between 100 and 500 AAs? (both for
struct and binding cutsequences)
-> for the Cshared_CDHIT, we're looking at the N-terminus, so we don't expect it to
    be larger than 200 AAs (see RBP paper + paper Aga). But on the other hand,
    clusters of only 15AAs long won't be interesting to take forth (don't represent
    a domain...)
-> for the Nshared_CDHIT, we're looking at everything after the N-terminal struct
    domain, so this can vary quite a bit. Here I'd say we rather want a minimum 
    than a maximum cutoff.
We will put it at >50 for C-term and 50 < N-term < 250. 

After recomputing the clusters, we now see:
    - 6 unknown_N clusters with more than 5 members, so 6 new HMMs that represent 
        unknown structural (N-terminal) domains
    - 47 unknown_C clusters with more than 5 members! (out of 347, which is only
        13.5%...)
"""

## Make predictions for prophage+MIL
## -------------------------------
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'
structs = ['Phage_T7_tail', 'Tail_spike_N', 'Prophage_tail', 'BppU_N', 'Mtd_N', 
           'Head_binding', 'DUF3751', 'End_N_terminal', 'phage_tail_N', 'Prophage_tailD1', 
           'DUF2163', 'Phage_fiber_2']
bindings = ['Lipase_GDSL_2', 'Pectate_lyase_3', 'gp37_C', 'Beta_helix', 'Gp58', 'End_beta_propel', 
            'End_tail_spike', 'End_beta_barrel', 'PhageP22-tail', 'Phage_spike_2', 
            'gp12-short_mid', 'Collar']
chaps = ['Peptidase_S74', 'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']
archive = ['Exo_endo_phos', 'NosD', 'SASA', 'Peptidase_M23']

domfile_mil = open(dom_path+'/milldomains_dict.json')
mildomains_dict = json.load(domfile_mil)
domfile_pro = open(dom_path+'/alldomains_dict.json')
prodomains_dict = json.load(domfile_pro)
mildomains_dict['source'] = ['millard']*len(mildomains_dict['score'])
prodomains_dict['source'] = ['prophage']*len(prodomains_dict['score'])
alldomains_dict = prodomains_dict.copy()
for i, sequence in enumerate(mildomains_dict['gene_sequence']):
    alldomains_dict['gene_sequence'].append(sequence)
    alldomains_dict['domain_name'].append(mildomains_dict['domain_name'][i])
    alldomains_dict['position'].append(mildomains_dict['position'][i])
    alldomains_dict['score'].append(mildomains_dict['score'][i])
    alldomains_dict['bias'].append(mildomains_dict['bias'][i])
    alldomains_dict['aln_range'].append(mildomains_dict['aln_range'][i])
    alldomains_dict['source'].append(mildomains_dict['source'][i])

preds = pbu.domain_RBP_predictor(path, pfam_db, alldomains_dict, structs, bindings, chaps, archive)
preds_df = pd.DataFrame.from_dict(preds)
preds_df.iloc[:50, 1:]

# count total detections
phagegenebase = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase.csv')
total_RBP_det = 0
for detected in preds_df['gene_sequence']:
    total_RBP_det += list(phagegenebase['sequence']).count(detected)

## Group sequences sharing domains
## -------------------------------
shared_Nterm = {}
shared_Cterm = {}
for i, struct in enumerate(preds_df['structural_domains']):
    binding = preds_df['binding_domains'][i]
    chap = preds_df['chaperone_domains'][i]
    sequence = preds_df['gene_sequence'][i]
    archive = preds_df['archived_domains'][i]
    
    # one or multiple structural domains
    if (len(struct) >= 1) and (len(binding) == 0) and (len(chap) == 0) and (len(archive) == 0):
        Nterm = ''.join(struct)
        if Nterm in shared_Nterm.keys():
            shared_Nterm[Nterm].append(sequence)
        else:
            shared_Nterm[Nterm] = [sequence]
    
    # one or multiple binding domains
    if (len(binding) >= 1) and (len(struct) == 0) and (len(chap) == 0) and (len(archive) == 0):
        Cterm = ''.join(binding)
        if Cterm in shared_Cterm.keys():
            shared_Cterm[Cterm].append(sequence)
        else:
            shared_Cterm[Cterm] = [sequence]
            
# make barplot
count_Nterm = [len(shared_Nterm[key]) for key in shared_Nterm.keys()]
count_Cterm = [len(shared_Cterm[key]) for key in shared_Cterm.keys()]
counts = count_Nterm + count_Cterm
keys = list(shared_Nterm.keys()) + list(shared_Cterm.keys())
domains_df = pd.DataFrame(data={'domain': keys, 'counts': counts}).sort_values(by='counts', ascending=False)
fig, ax = plt.subplots(figsize=(12, 8))
ax = sns.barplot(x='counts', y='domain', data=domains_df)
ax.set_title('Number of sequences that share given domain(s) \n w/o other known domains in the sequence')
fig.tight_layout()
#fig.savefig('shared_domains_barplot.png', dpi=400)

# export unknown pieces of protein sequences to fasta for clustering (CD-HIT)
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/clusters'
fasta_N = open(dom_path+'/'+ 'unknown_C_cutsequences.fasta', 'w')
bar = tqdm(total=sum([len(shared_Nterm[key]) for key in shared_Nterm.keys()]), position=0, leave=True)
for key in shared_Nterm.keys():
    for i, sequence in enumerate(shared_Nterm[key]):
        doms, _, _, ranges = pbu.gene_domain_scan(path, pfam_db, [sequence])
        index = [j for j, dom_match in enumerate(doms) if dom_match == key]
        if len(index) > 0:
            sequence_cut = max([ranges[j][1] for j in index])
            protein = str(Seq(sequence).translate())[sequence_cut:-1] # select C-terminus
            if len(protein) >= 50:
                fasta_N.write('>sequence_'+str(i)+'_'+key+'\n'+protein+'\n')
        bar.update(1)
fasta_N.close()
bar.close()

fasta_C = open(dom_path+'/'+ 'unknown_N_cutsequences.fasta', 'w')
bar = tqdm(total=sum([len(shared_Cterm[key]) for key in shared_Cterm.keys()]), position=0, leave=True)
for key in shared_Cterm.keys():
    for i, sequence in enumerate(shared_Cterm[key]):
        doms, _, _, ranges = pbu.gene_domain_scan(path, pfam_db, [sequence])
        index = [j for j, dom_match in enumerate(doms) if dom_match == key]
        if len(index) > 0:
            sequence_cut = min([ranges[j][0] for j in index])
            protein = str(Seq(sequence).translate())[:sequence_cut] # select N-terminus
            if 50 <= len(protein) < 250:
                fasta_C.write('>sequence_'+str(i)+'_'+key+'\n'+protein+'\n')
        bar.update(1)
fasta_C.close()
bar.close()
              
## Cluster sequences with CD-HIT
## -------------------------------
# first test 
cdpath = '/opt/anaconda3/pkgs/cd-hit-4.8.1-hd9629dc_0/bin'
input = dom_path+'/Phage_T7_tail_shared_sequences.fasta'
output = '/Users/Dimi/Desktop/T7test'
cdout, cderr = pbu.cdhit_python(cdpath, input, output)

# clustering all domain-grouped sequences 
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/clusters' 
cdpath = '/opt/anaconda3/pkgs/cd-hit-4.8.1-hd9629dc_0/bin'
input_N = dom_path+'/'+ 'unknown_C_cutsequences.fasta'
output_N = dom_path+'/'+ 'unknown_C_termini'
outN, errN = pbu.cdhit_python(cdpath, input_N, output_N)
input_C = dom_path+'/'+ 'unknown_N_cutsequences.fasta'
output_C = dom_path+'/'+ 'unknown_N_termini'
outC, errC = pbu.cdhit_python(cdpath, input_C, output_C)

# cluster C at 40%
input_N = dom_path+'/'+ 'unknown_C_cutsequences.fasta'
output_N = dom_path+'/'+ 'unknown_C_termini_40'
outN, errN = pbu.cdhit_python(cdpath, input_N, output_N, c=0.40, n=2)
    
## Make cluster FASTA files
## -------------------------------
clusters_unknown_N = open(dom_path+'/'+ 'unknown_N_termini.clstr')
input_file = dom_path+'/'+ 'unknown_N_cutsequences.fasta'
sequence_list = []
names_list = []
for record in SeqIO.parse(input_file, 'fasta'): # get sequences in a list
    sequence_list.append(str(record.seq))
    names_list.append(record.description) # sequence_list and names_list match indices
cluster_iter = -1
cluster_sequences = []
cluster_indices = []
for line in clusters_unknown_N.readlines():
    # new cluster
    if line[0] == '>':
        # finish old cluster if not the first one
        if (cluster_iter >= 0) & (len(cluster_sequences) >= 5):
            fasta = open(dom_path+'/'+'unknown_N_termini_cluster_'+str(cluster_iter)+'.fasta', 'w')
            for i, seq in enumerate(cluster_sequences):
                index = cluster_indices[i]
                fasta.write('>sequence'+str(index)+'\n'+seq+'\n')
            fasta.close()
        # initiate new cluster
        cluster_sequences = []
        cluster_indices = []
        cluster_iter += 1
    
    # in a cluster
    else:
        match = re.search('>sequence_[0-9]+_[A-Za-z_0-9]+', line).group(0)
        current_index = [i for i, description in enumerate(names_list) if match[1:] in description]
        current_sequence = sequence_list[current_index[0]]
        cluster_indices.append(current_index[0])
        cluster_sequences.append(current_sequence)

# Same for C-terminal sequences
clusters_unknown_C = open(dom_path+'/'+ 'unknown_C_termini.clstr')
input_file = dom_path+'/'+ 'unknown_C_cutsequences.fasta'
sequence_list = []
names_list = []
for record in SeqIO.parse(input_file, 'fasta'): # get sequences in a list
    sequence_list.append(str(record.seq))
    names_list.append(record.description)
cluster_iter = -1
cluster_sequences = []
cluster_indices = []
for line in clusters_unknown_C.readlines():
    # new cluster
    if line[0] == '>':
        # finish old cluster if not the first one
        if (cluster_iter >= 0) & (len(cluster_sequences) >= 5):
            fasta = open(dom_path+'/'+'unknown_C_termini_cluster_'+str(cluster_iter)+'.fasta', 'w')
            for i, seq in enumerate(cluster_sequences):
                index = cluster_indices[i]
                fasta.write('>sequence'+str(index)+'\n'+seq+'\n')
            fasta.close()
        # initiate new cluster
        cluster_sequences = []
        cluster_indices = []
        cluster_iter += 1
    
    # in a cluster
    else:
        match = re.search('>sequence_[0-9]+_[A-Za-z_0-9]+', line).group(0)
        current_index = [i for i, description in enumerate(names_list) if match[1:] in description]
        current_sequence = sequence_list[current_index[0]]
        cluster_indices.append(current_index)
        cluster_sequences.append(current_sequence)


# 8 - BUILDING NEW HMMs TO INCREASE DISCOVERY
# --------------------------------------------------
"""
STATUS: DONE

We see very few N/C combinations in the bar plot above. Which hints at a lot
of domains that are unknown and thus are not taken into account in the iterative
discovery. However, one option is to build our own HMMs from the sequences we have
already detected and have e.g. no known domain at the N or C terminus. To build
a profile HMM, the first thing that is needed is an alignment (multiple sequences)
from which to build a HMM. We would've expected a lot more combinations as RBPs 
are known to be modular!
   
Qs:
    - what clustering algorithm to use? CD-HIT
    - what threshold for sequence similarity is appropriate? 50%
    - what are the new domains we find?
    - how many sequences at minimum do we need? 5 per MSA
    
2/02/21
Looking at the MSA's of clusters manually. The alignment quality measure
in Jalview can be an interesting measure to look at, it is an ad-hoc measure describing
the likelihood of observing a mutation in an MSA.
- Cluster_N_28 looks like two separate subclusters, looking at the MSA it is clear. 
    Sequences 0-38 are not from the same cluster as the rest of the sequences. 
    And some at the end also don't appear to be from the same cluster (44, 63, 
    78, 88, 89, 91). We'll separate them into two subclusters and run MSA again 
    for those two new clusters. Actually, MSA for cluster 28a now still looks like
    multiple mini sub clusters within... Looks better but its percentage identity 
    is overall quite low... This will be a cluster to keep an eye on.
- Cluster_C_2: the last sequence seems liks an off one.
- Cluster_C_120: very low percentage identity overal... another one to keep an
    eye on.
- Cluster_C_175: last sequence (318) seems off one.
- Cluster_C_221: is a big (complex cluster), not a lot of highly conserved residues
    yet quality overall seems satisfactory. Hard to judge this one.
- Cluster_C_235: sequence 177 seems off (very short compared to rest, almost no
    matches).
- Cluster_C_271: another big cluster that is difficult to assess. Overal quality
    measure looks good but very few residues that are highly conserved.
- Cluster_C_277: again, large cluster. Also sequence 209 seems extraordinarily long.
- Cluster_C_292: sequence 108 seems like a long outlier. Quality overall seems good.
- Cluster_C_301: this one seems odd, overall the percentage identify is not terrible
    but also not quite high (except for the very end). Also the quality overall is
    very low! This rather seems like a C-terminal chaperone domain judging from the
    MSA.
- Cluster_C_321: overall low percentage identity but satisfactory quality, except
    for the start which is just sequence 108 (seems off).
- Cluster_C_331: seems low in quality overall.
- Cluster_C_338: identity is low but quality overall seems OK.

Overall, we see much more variation in these unknown C terminal ends, which is also
what we expect (the C-terminus to be less conserved than the N-terminus). 
In JalView, one indicator measure that we look at is the quality, which can give us
an estimate of how reliable the alignment/cluster is even though substitutions have
taken place.
All in all, we have 6 clusters of which the MSAs seem off and should keep in mind
when doing predictions.

11/02
we're done with checking the MSA's and have built HMMs off of them. Now it's time
to get HMMsearch going and see what new sequences we find (in the prophage genes). Those
hits can then be scanned to see whether or not other domains exist in it:
- if none: we'll check the sequence closely to see where this hit came from
- if one from the structural/binding/chap list: means we've already found that hit and
    thus we don't need to check it again
- if one from block list: then we're finding hits that we don't want -> should check which
    new HMM model led to this and if this is occurs more than once -> potentially delete
    that HMM model.

From the unknown N-terminal HMMs we get 159 unique hits, and 166 in total (multiple HMM detected the same
sequence). This duplicates can be the result of fragmented N-terminus or that these HMMs actually 
just represent the same domain. We can check this by printing the aln_range!
-> we see that 7 hits are dupped, and it always involves the N_termini_28a + N_termini_45 clusters. The
    latter HMM matches the entire N-termini (0-200), while 28a only 8-80. HMM 45 could be an overlapping 
    extension of 28a
    Now, finding dups is not a bad thing per se, we only don't want to miss hits! We could opt to delete cluster
    28a if (only if) it doesn't result in other hits as well!
-> indeed, no hits of 28a are not found by 45, so we could safely delete 28a, but then again, in future updates to
    the database this might not be the case anymore! So we'll leave it in anyway.

Furthermore, we find 4 C-terminal HMMs that are linked with a blocked domain (Phage_T4_gp36)
in one sequence ('C_termini_24', 'C_termini_205', 'C_termini_300', 'C_termini_340').
In any case, 329 of our hits get deleted anyway because they are already found! But this doesn't
mean we have to delete their HMMs, they can still be valid in future updates and make the dataframe
of predicted RBPs (see somain_RBP_predictor) more complete. These 329 hits don't have to be checked
again, the other 156 (including the hit related to blocked domain) do have to be checked again.
The question then becomes: how? Let's start with a simple BLAST.

BLAST through WIFI at home is not attaiable (~70hours...). With an ethernet it might be possible
as this is faster. Other option is to manually BLAST all those above 200AAs... or local BLAST
against the RBP database but that would be cheating i guess..

18/02
We will check the sequences manually, all those larger than 200AAs. But also, many sequences in those
218 filtered ones (still at DNA level) look very much alike! So we will translate and send them to CD-hit
to cluster at 90% to decrease the number of sequences that we have to check manually.

Why is this validation necessary?
All the known domains, we have manually checked on Pfam and curated them according to their relatedness
to RBPs, now we have made new HMMs that hopefully all also represent RBPs, but how can we be sure of
that? We can argue that because of the fact that we started with a known N- or C-terminus to construct
these HMMs for the unknown ends, that we already have a good probability that these new HMM also are
related to RBPs. Nevertheless, if we want to incorporate these HMMs into Pfam, we need to be sure.
We will incorporate a simple rule: if the top3 annotations indicate a phage RBP/tail protein, we keep the HMM.
We will then link this new information back to our earlier findings when we checked the clusters manually.

19/02 - UPDATED 9/04
Based on the results (see .xlsx), we conclude that the following clusters are unwanted: 
N_termini_1, C_termini_120, C_termini_143, C_termini_175, C_termini_277, C_termini_221,
C_termini_271, C_termini_234, C_termini_237, because of annotation related to flagellar 
biosynthesis and baseplates. All of the other clusters are either linked with hypothetical proteins 
(almost all of them), phage tail proteins or very occasionally tail fibers.

Two comments and questions remain:
- the C_termini clusters exhibit a load of overlap in the detected sequences, so these might actually be some
    duplicates of the same domain.
- the thing we still don't know is: what do these HMMs represent? A good extra thing to do is to compare HMMs
    (or their MSAs) with HHPred, but this failed for one example when I tried... should doublecheck.

Now the HMMs are added to Pfam-A_extended.hmm, we will try to press it again and predict. This means we also have
to extend our lists of structural and binding domains. We named every added domain as
phage_RBP_Nxx or phage_RBP_Cxx.

FINAL STEP TO DO: add the good HMMs to Pfam, press the DB again and rerun detections.

UPDATE 23/04
We will not preventively discard clusters anymore but see to what they are linked in RBP exploration.
Also, we are checking the N-terminal clusters for bitscores to perhaps change their threshold,
implemented a new function hmmsearch_thres_python that allows to fill in a threshold.

Based on raw results of HMM search for the N-terminal clusters and comparisons between the
obtained bitscores and E-values of hits, we conclude that for N-terminal (shorter) domains,
a threshold on bitscore between 18 and 20 seems appropriate.
"""

## Paths
## -------------------------------
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
clo_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/clustalo'
cluster_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/clusters' 
profile_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/profiles' 
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'

## Do MSA's from within Python
## -------------------------------
clusters_N = ['1', '4', '26', '28', '34', '45']
clusters_C = ['2', '10', '24', '43', '59', '60', '62', '67', '79', '97', '111', 
              '115', '120', '126', '138', '143', '157', '164', '175', '180', 
              '205', '217', '220', '221', '223', '234', '235','237', '249', '259',
              '267', '271', '277', '281', '292', '293', '296', '300', '301', '319', 
              '320', '321', '326', '331', '337', '338', '340']

# MSA's (takes a few minutes)
for cluster in clusters_N:
    infile = cluster_path+'/unknown_N_termini_cluster_'+cluster+'.fasta'
    outfile = clo_path+'/unknown_N_termini_MSA_clst_'+cluster+'.sto'
    Nout, Nerr = pbu.clustalo_python(clo_path, infile, outfile, out_format='st')
    
for cluster in clusters_C:
    infile = cluster_path+'/unknown_C_termini_cluster_'+cluster+'.fasta'
    outfile = clo_path+'/unknown_C_termini_MSA_clst_'+cluster+'.sto'
    Cout, Cerr = pbu.clustalo_python(clo_path, infile, outfile, out_format='st')

## Build new HMMs
## -------------------------------
for cluster in clusters_N:
    outfile = profile_path+'/unknown_N_termini_'+cluster+'.hmm'
    msafile = clo_path+'/unknown_N_termini_MSA_clst_'+cluster+'.sto'
    bout, berr = pbu.hmmbuild_python(path, outfile, msafile)

for cluster in clusters_C:
    outfile = profile_path+'/unknown_C_termini_'+cluster+'.hmm'
    msafile = clo_path+'/unknown_C_termini_MSA_clst_'+cluster+'.sto'
    bout, berr = pbu.hmmbuild_python(path, outfile, msafile)

## Search the new domains for hits
## -------------------------------
# load all genes
genes_df = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase_df.csv', index_col=0)
unique_genes = genes_df.columns
phage_genomes = genes_df.index

# translate genes and write to fasta
phageproteinbase = open(dom_path+'/PhageProteinBase.fasta', 'w')
for i, gene in enumerate(unique_genes):
    protein_sequence = str(Seq(gene).translate())[:-1] # don't need stop codon
    phageproteinbase.write('>protein_sequence_'+str(i)+'\n'+protein_sequence+'\n')
phageproteinbase.close()

new_gene_hits = {} # key:value = gene:cluster
new_gene_ranges = {} # key:value = gene:aln_range
for cluster in tqdm(clusters_N):
    hmmfile = profile_path+'/unknown_N_termini_'+cluster+'.hmm'
    hits, scores, biases, ranges = pbu.hmmsearch_thres_python(path, hmmfile, dom_path+'/PhageProteinBase.fasta', threshold=18)
    for i, hit in enumerate(hits):
        OM_score = math.floor(math.log(scores[i], 10)) # order of magnitude
        OM_bias = math.floor(math.log(biases[i]+0.00001, 10))
        if ((OM_score > OM_bias) and (hit not in new_gene_hits.keys())):
            new_gene_hits[hit] = ['N_termini_'+cluster]
            new_gene_ranges[hit] = [ranges[i]]
        elif ((OM_score > OM_bias) and (hit in new_gene_hits.keys())):
            new_gene_hits[hit].append('N_termini_'+cluster)
            new_gene_ranges[hit].append(ranges[i])

for cluster in tqdm(clusters_C):
    hmmfile = profile_path+'/unknown_C_termini_'+cluster+'.hmm'
    hits, scores, biases, ranges = pbu.hmmsearch_python(path, hmmfile, dom_path+'/PhageProteinBase.fasta')
    for i, hit in enumerate(hits):
        OM_score = math.floor(math.log(scores[i], 10)) # order of magnitude
        OM_bias = math.floor(math.log(biases[i]+0.00001, 10))
        if ((OM_score > OM_bias) and (hit not in new_gene_hits.keys())):
            new_gene_hits[hit] = ['C_termini_'+cluster]
            new_gene_ranges[hit] = [ranges[i]]
        elif (OM_score > OM_bias) and (hit in new_gene_hits.keys()):
            new_gene_hits[hit].append('C_termini_'+cluster)
            new_gene_ranges[hit].append(ranges[i])

#nghdf = pd.DataFrame.from_dict({'hits': new_gene_hits.keys(), 'clusters': new_gene_hits.values()})
#nghdf.to_csv('new_gene_hits2304.csv', index=False)

# lots of dups (same hit, multiple clusters)
for gene_hit in new_gene_hits.keys():
    if len(new_gene_hits[gene_hit]) > 1:
        print('dups found!')
        print(new_gene_hits[gene_hit])
        print(new_gene_ranges[gene_hit])

# go over all keys, scan genes & save those to check with BLAST
structs = ['Phage_T7_tail', 'Tail_spike_N', 'Prophage_tail', 'BppU_N', 'Mtd_N', 
           'Head_binding', 'DUF3751', 'End_N_terminal', 'phage_tail_N', 'Prophage_tailD1', 
           'DUF2163', 'Phage_fiber_2']
bindings = ['Lipase_GDSL_2', 'Pectate_lyase_3', 'gp37_C', 'Beta_helix', 'Gp58', 'End_beta_propel', 
            'End_tail_spike', 'End_beta_barrel', 'PhageP22-tail', 'Phage_spike_2', 
            'gp12-short_mid', 'Collar']
chaps = ['Peptidase_S74', 'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']

doublecheck = open(dom_path+'/sequences_to_doublecheck.fasta', 'w')
for gene_hit in tqdm(new_gene_hits.keys()):
    index = re.search('[0-9]+', gene_hit).group(0)
    gene_sequence = unique_genes[int(index)]
    hits, scores, biases, ranges = pbu.gene_domain_scan(path, pfam_db, [gene_sequence])
    if len(hits) == 0:
        # translate to protein and save if it's longer than 200AAs
        protein_sequence = str(Seq(gene_sequence).translate())[:-1] # don't need stop codon
        if len(protein_sequence) > 200:
            doublecheck.write('>'+gene_hit+'\n'+protein_sequence+'\n')
    else:
        for i, dom in enumerate(hits):
            if (dom not in structs) and (dom not in bindings) and (dom not in chaps):
                protein_sequence = str(Seq(gene_sequence).translate())[:-1] # don't need stop codon
                if len(protein_sequence) > 200:
                    doublecheck.write('>'+gene_hit+'\n'+protein_sequence+'\n')
doublecheck.close()

doublecheck = dom_path+'/sequences_to_doublecheck.fasta'
count = 0
for record in SeqIO.parse(doublecheck, 'fasta'):
    count += 1
count

# cluster the sequences at 90% to avoid duplicate work with BLAST
cdpath = '/opt/anaconda3/pkgs/cd-hit-4.8.1-hd9629dc_0/bin'
output_check = dom_path+'/clusters_to_doublecheck'
clust_out, clust_err = pbu.cdhit_python(cdpath, doublecheck, output_check, c=0.90, n=5)

doublecheck_filtered = dom_path+'/clusters_to_doublecheck.txt'
count = 0
for record in SeqIO.parse(doublecheck_filtered, 'fasta'):
    count += 1
count


# 9 - FINAL PREDICTIONS
# --------------------------------------------------
"""
To do:
v - Set final predictions (old ones and new ones)
v - Use that list to make a subset of PhageGeneBase
3 - Remove all data from local computer that is not needed anymore at this point
4 - check duplicates at RBP AND bacterial genome level (in particular, if two RBPs are the
        same, check the two corresponding bacterial genomes)
5 - delete the ones that are double duplicates (which are exactly the same datapoints)
6 - make pw matrix of (RBP - genome) combinations

DON'T add all bacterial genomes, 3560 is way too many, and a lot of dups.
We will first make a list(set()) of the bacterial genomes and then add them to a separate 
file. The unique ones are still 1129 genomes...
We will fix it in a different way, at the level of the feature engineering. We have the host
name so we can load the appropriate file of bacterial_genomes (PhageBase_1) in and select the
correct accession number. But untill the genomes are converted to either features or receptors,
the full genomes are too big to be out in a single file.
"""

# define paths and press new database
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
new_pfam = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A_extended.hmm'
output, err = pbu.hmmpress_python(path, new_pfam)

genes_df = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase_df.csv', index_col=0)
unique_genes = genes_df.columns
phage_genomes = genes_df.index

# redo the all_domains scan to update the json files (takes a long time!)
prodomains_dict = pbu.all_domains_scan(path, new_pfam, list(unique_genes))
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
preds = pbu.domain_RBP_predictor(path, new_pfam, prodomains_dict, structs, bindings, chaps, archive)
preds_df = pd.DataFrame.from_dict(preds)
preds_df.iloc[:50, 1:]

# faster way: old detections + new filtered detections from the steps above (new_gene_dict)
# run the code to get new_gene_dict first
domfile_pro = open(dom_path+'/alldomains_dict.json')
prodomains_dict = json.load(domfile_pro)
old_preds = pbu.domain_RBP_predictor(path, new_pfam, prodomains_dict, structs, bindings, chaps, archive)
old_preds_df = pd.DataFrame.from_dict(old_preds)
final_RBP_set = list(old_preds_df['gene_sequence'])

# How many C-terminals do we find
detected_C = {}
for cterm_list in old_preds_df.binding_domains:
    for item in cterm_list:
        if item in detected_C.keys():
            detected_C[item] += 1
        else:
            detected_C[item] = 1

detected_Cdf = pd.DataFrame(data={'domain':detected_C.keys(), 'counts':detected_C.values()}).sort_values(by='counts', ascending=False)
fig, ax = plt.subplots(figsize=(12, 8))
ax = sns.barplot(x='counts', y='domain', data=detected_Cdf)
ax.set_title('Detected known C-terminal domains')
fig.tight_layout()
#fig.savefig('detected_known_Cterminal_domains.png', dpi=400)

# get final RBP set
filter_list = []
#filter_list = ['N_termini_1', 'C_termini_120', 'C_termini_143', 'C_termini_175', 
#                'C_termini_221', 'C_termini_234', 'C_termini_237', 'C_termini_271', 
#                'C_termini_277'] # domains we don't want to include
for gene_hit in new_gene_hits.keys():
    index = re.search('[0-9]+', gene_hit).group(0)
    gene_sequence = unique_genes[int(index)]
    intersect = list(set(new_gene_hits[gene_hit]).intersection(filter_list))
    if (len(intersect) == 0) & (gene_sequence not in final_RBP_set):
        final_RBP_set.append(gene_sequence)
len(final_RBP_set) # 1407 RBPs / without filters: 1698 RBPs

# take subset of PhageGeneBase
final_RBP_set_indices = []
genebase = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase.csv')
for i, gene in enumerate(genebase['sequence']):
    if gene in final_RBP_set:
        final_RBP_set_indices.append(i)
genebase_subset = genebase.iloc[final_RBP_set_indices, :]
genebase_subset.shape
genebase_subset.to_csv(dom_path+'/RBPbase_230421.csv', index=False)

unique_bact_genomes = list(set(genebase_subset['host_accession']))
len(unique_bact_genomes) # still 1129

unique_combos = []
genebase_subset = genebase_subset.reset_index(drop=True)
for i, number in enumerate(genebase_subset.phage_nr):
    combo = (number, genebase_subset.host[i])
    if combo not in unique_combos:
        unique_combos.append(combo)
len(unique_combos)
    

# 10 - N-TERMINAL CLUSTER VALIDATION
# --------------------------------------------------
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'
clo_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/clustalo'
cluster_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/clusters' 
profile_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/profiles' 
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'

N_1s = cluster_path+'/unknown_N_termini_cluster_1.fasta'
N_28s = cluster_path+'/unknown_N_termini_cluster_28.fasta'
genes_df = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase_df.csv', index_col=0)
unique_genes = genes_df.columns

N1pieces = []
for record in SeqIO.parse(N_1s, 'fasta'):
    N1pieces.append(str(record.seq))
N28pieces = []
for record in SeqIO.parse(N_28s, 'fasta'):
    N28pieces.append(str(record.seq))

pie = 'MADYKLSELNSIDTIRSDDLLHVRVKKRPEMLGDEDRRMTYQDFLASFKLERFVQIAGSTMTGDLGIVKLLYGGKAVFDPTDSSEITMGDVLKAFKINANGLKLTIADASRSATVYHTLNKPSPNELGMRTNEENDARYARLAV'
lis = [x for x in unique_genes if pie in x]
N28genes = []
for gene in unique_genes:
    for piece in N28pieces:
        if piece in gene:
            N28genes.append(gene)

hmmfile = profile_path+'/unknown_N_termini_28.hmm'
for N28gene in N28genes:
    hits, scores, biases, ranges = pbu.gene_domain_search(path, hmmfile, N28gene)
    for i, hit in enumerate(hits):
        print(hit, scores[i])


# 7 - VALIDATION
# --------------------------------------------------
"""
STATUS: NOT STARTED

1) Check what genes are present before and after the predicted RBP -> should be
structural genes.

2) check the available annotation -> is the RBP already annotated?

3) BLAST it, see what the top hits are.

Qs & remarks:
    - it is my best guess that most of the detected genes will not have an annotation. And
        that only few genes in general will be annotated as RBP, which then also
        causes the double-checked list to be small.
    - Interesting: what domains are located in these unique_annotations proteins? What are
        the domains that should be related to RBPs but that we didn't find starting
        from T7_tail?
    - 10/12/20: When having to check 4 x 1022 genes in series, it will take about
        200-300 hours... We need to make this faster for sure. 
        - cable internet
        - multiple queries in one search
        - only checking 1 before and 1 after?
       Even doing multiple queries in one go takes very long... not a scalable solution
       The option I'll try is to make one big fasta and input that to the BLAST website
       and download the results afterwards for further processing.
       
       Of the 85000+ genes, only 16 (!) are annotated as tail fiber or spike. 
       11 of them contain the Peptidase_S74 domain, two contain a Caudo_TAP domain,
       one contains a Collar domain and one contains BppU_N, PTR and Collagen.
       10 of the 16 sequences overlap with the 1022 we've found in the iterative discovery.
       Caudo_TAP represents the domain of an assembly protein, which is not what we
       want to include in the database. The other domains are familiar to what we
       saw earlier and can be taken into account.
        
TO DO:
    - recheck the checked_detections now dna_sequences are compared instead of
        protein... if that does not match up, then we implement the hit in test or
        test in hit approach. Then below the unique annotations are implemented
        as well, finally these should be checked for domains and added if appropriate.
    - figure out plan B for BLAST checks. Even the 1022 hits will take a long time...
        make FASTA file of all previous/next sequences & input to BLAST website
"""

## STRUCTURAL GENE CLUSTER
## -----------------------
# get index of the detected RBP in question
# FIRST RUN ITERATIVE DISCOVERY
phagegenebase = pd.read_csv('/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/PhageGeneBase.csv')
minidetect = detections[:5]
validation_dict = {'gene': detections, 'previous_1':[], 'previous_2': [], 'next_1': [], 'next_2': []}
bar = tqdm(total=len(detections), position=0, leave=True)
dom_path = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection'

for gene in minidetect:
    gene_index = list(phagegenebase['sequence']).index(gene)
    host_number = phagegenebase['phage_nr'][gene_index]
    
    # collect the two genes before and after IF from the same genome (check number)
    temp_fasta = open(dom_path+'/sequences_validation.fasta', 'w')
    for i in range(gene_index-2, gene_index+2):
        if phagegenebase['phage_nr'][i] == host_number:
            temp_fasta.write('>sequence_validation'+str(i)+'\n'+phagegenebase['sequence'][i]+'\n')
    temp_fasta.close()
    
    # BLAST them and check the top hits
    mail = 'Dimitri.Boeckaerts@UGent.be'
    position_key = ['previous_1', 'previous_2', 'next_1', 'next_2']
    temp_fasta = open(dom_path+'/sequences_validation.fasta').read()
    result = NCBIWWW.qblast('blastx', 'nr', sequence=temp_fasta)
    for i, record in enumerate(NCBIXML.parse(result)): # parse multiple records (should be 4 in total)
        description = record.descriptions[0].title.split('>')[0] # top hit [0]
        print(i, description)
    #res_d = blast_descriptions(sequence=temp_fasta, email=mail)    
    #for i, testsubject in enumerate(sequences_to_test):
    #    res_d = blast_descriptions(sequence=testsubject, email=mail)
    #    validation_dict[position_key[i]].append(res_d['title'])
    
    bar.update(1)
bar.close()

## ANNOTATION
## -----------------------
# Get the annotated RBPs in the bacterial genomes
names = ['enterococcus_faecium', 'staphylococcus_aureus', 'klebsiella_pneumoniae', 
         'acinetobacter_baumannii', 'pseudomonas_aeruginosa', 'enterobacter_cloacae', 'enterobacter_aerogenes']
annotated_RBPs = {'accession': [], 'host': [], 'annotation': [], 'dna_sequence': [], 'protein_sequence': []}
directory = '/Users/Dimi/Desktop/PhageBase/'
for name in names:
    bact_genomes = pd.read_csv(directory+'phagebase1_'+name+'.csv')
    accession_codes = bact_genomes['accession']
    bar = tqdm(total=len(accession_codes), position=0, leave=True)
    for acc in accession_codes:
        handle = Entrez.efetch(db='nucleotide', rettype='gb', id=acc)
        for sequence in SeqIO.parse(handle, 'gb'):
            for feature in sequence.features:
                if (feature.type == 'CDS'):
                    try: # check whether there is a product
                        protein_name = feature.qualifiers['product'][0]
                        
                        # only add tail fibers or tail spikes
                        if re.search(r'tail.?(?:spike|fiber)', protein_name, re.IGNORECASE) is not None:
                            protein_seq = '-'; dna_seq = '-'
                            try:
                                protein_seq = feature.qualifiers['translation'][0]
                                dna_seq = str(feature.location.extract(sequence).seq)
                            except KeyError:
                                pass
                            
                            annotated_RBPs['accession'].append(acc)
                            annotated_RBPs['host'].append(name)
                            annotated_RBPs['annotation'].append(protein_name)
                            annotated_RBPs['dna_sequence'].append(dna_seq)
                            annotated_RBPs['protein_sequence'].append(protein_seq)
                        
                    except KeyError:
                        pass
        bar.update(1)
    bar.close()
        
# compare with detected RBP genes   
checked_detections = [gene for gene in detections if gene in annotated_RBPs['dna_sequence']]
print('number of double-checked detections:', len(checked_detections))
print('percentage that is double-checked:', len(checked_detections)/len(detections))

unique_annotations = [gene for gene in annotated_RBPs['protein_sequence'] if gene not in detections]
unique_domainRBPs = [gene for gene in detections if gene not in annotated_RBPs['protein_sequence']]

# Check the domains in these annotated sequences
path = '/opt/anaconda3/pkgs/hmmer-3.1b2-0/bin'
pfam_db = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/Pfam-A.hmm'
T7_tail_hmm = '/Users/Dimi/GoogleDrive/PhD/3_WP1_PHAGEBASE/32_DATA/RBP_detection/T7tail.hmm'

for gene in annotated_RBPs['dna_sequence']:
    hits, scores, biases, ranges = pbu.gene_domain_scan(path, pfam_db, [gene])
    print(hits, scores, biases, ranges)

for test in annotated_RBPs['dna_sequence']:
    hits, scores, biases, ranges = pbu.gene_domain_scan(path, pfam_db, [test])
    for hit in detections:
        if (test in hit) or (hit in test):
            print('check!')
        else:
            if 'Caudo_TAP' not in hits:
                detections.append(test)
                