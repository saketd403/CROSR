

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
import libmr

from sklearn.metrics import roc_auc_score

def calc_auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
   
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result   

def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    n_classes = openmax_fc8.size()[1]
    scores = []
    for category in range(n_classes):
        scores += [sp.exp(openmax_fc8[category])]
                
    total_denominator = sp.sum(sp.exp(openmax_fc8)) + sp.exp(sp.sum(openmax_score_u))
    prob_scores = scores/total_denominator 
    prob_unknowns = sp.exp(sp.sum(openmax_score_u))/total_denominator
    
    
    #modified_scores = [prob_unknowns] + prob_scores.tolist()
    #assert len(modified_scores) == (NCLASSES+1)
    #return modified_scores

    return prob_unknowns

def compute_distance(query_vector, mean_vec, distance_type = 'eucos'):
    """ 

    Output:
    --------
    query_distance : Distance between respective channels

    """

    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec, query_vector)/200. + spd.cosine(mean_vec, query_vector)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec, query_vector)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec, query_vector)
    else:
        print "distance type not known: enter either of eucos, euclidean or cosine"
    return query_distance
    
