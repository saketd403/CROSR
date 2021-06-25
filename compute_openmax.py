import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat

from openmax_utils import *
from evt_fitting import weibull_tailfitting, query_weibull

import numpy as np
import libmr

from sklearn.metrics import roc_auc_score


def recalibrate_scores(weibull_model, img_features,
                        alpharank = 10, distance_type = 'eucos'):

    img_features = img_features[0]
    NCLASSES = len(list(img_features))
    ranked_list = img_features.argsort().ravel()[::-1]

    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = np.zeros(NCLASSES)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    
    openmax_layer = []
    openmax_unknown = []
    
    for cls_indx in range(NCLASSES):

         
        category_weibull = query_weibull(cls_indx, weibull_model, distance_type = distance_type)
        distance = compute_distance(img_features, category_weibull[0],
                                            distance_type = distance_type)

        wscore = category_weibull[2].w_score(distance)
        modified_unit = img_features[cls_indx] * ( 1 - wscore*ranked_alpha[cls_indx] )
        openmax_layer += [modified_unit]
        openmax_unknown += [img_features[cls_indx] - modified_unit]

    openmax_fc8 = np.asarray(openmax_layer)
    openmax_score_u = np.asarray(openmax_unknown)

    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    """
    logits = [] 
    for indx in range(NCLASSES):
        logits += [sp.exp(img_features[indx])]
    den = sp.sum(sp.exp(img_features))
    softmax_probab = logits/den

    return np.asarray(openmax_probab), np.asarray(softmax_probab)
    """
    return openmax_probab

def get_scores(data_type,weibull_model,feature_path):

    results = []

    for cls_no in os.listdir(os.path.join(feature_path,data_type)):
        
        for filename in os.listdir(os.path.join(feature_path,data_type,cls_no)):

            img_features = np.load(os.path.join(feature_path,data_type,cls_no,filename))

            openmax =  recalibrate_scores(weibull_model, img_features)

            results.append(openmax)

    return np.array(results)

       
def get_args():
    parser = argparse.ArgumentParser(description='Get open max probability and compute AUROC')
    parser.add_argument('--MAV_path',default="./saved_MAVs/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--distance_scores_path',default="./saved_distance_scores/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--feature_dir',default="./saved_features/cifar10",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--alpha_rank',default=10,type=int,help="Alpha rank classes to consider")
    parser.add_argument('--weibull_tail_size',default=20,type=int,help="Tail size to fit")
    parser.set_defaults(argument=True)

    return parser.parse_args()

def main():

    args = get_args()

    distance_path = args.distance_scores_path
    mean_path = args.MAV_path
    alpha_rank = args.alpha_rank
    weibull_tailsize = args.weibull_tail_size

    weibull_model = weibull_tailfitting(mean_path, distance_path,
                                        tailsize = weibull_tail_size)

    in_dist_scores = get_scores("val",weibull_model,args.feature_dir)
    open_set_scores = get_scores("open_set",weibull_model,args.feature_dir)

    print("The AUROC is ",calc_auroc(in_dist_scores, open_set_scores))


if __name__ == "__main__":
    main()
