

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat

from openmax_utils import *
import numpy as np

import libmr


#---------------------------------------------------------------------------------
def weibull_tailfitting(meanfiles_path, distancefiles_path,
                        tailsize = 20, 
                        distance_type = 'eucos'):
                        

    
    weibull_model = {}
 
    for filename in os.listdir(meanfiles_path):
        category = filename.split(".")[0]
        weibull_model[category] = {}
        distance_scores = np.load(os.path.join(distancefiles_path,category+".npy"))[()][distance_type]
        
        meantrain_vec = np.load(os.path.join(meanfiles_path,category+".npy"))

        weibull_model[category]['distances_%s'%distance_type] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec

        distance_scores = distance_scores.tolist()

        mr = libmr.MR()

        tailtofit = sorted(distance_scores)[-tailsize:]

        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[category]['weibull_model'] = mr

    

    return weibull_model

def query_weibull(category_name, weibull_model, distance_type = 'eucos'):

    category_weibull = []
    category_weibull += [weibull_model[category_name]['mean_vec']]
    category_weibull += [weibull_model[category_name]['distances_%s' %distance_type]]
    category_weibull += [weibull_model[category_name]['weibull_model']]

    return category_weibull    

