# -*- coding: utf-8 -*-
###################################################################################################
# Copyright (c) 2016 , Regents of the University of Colorado on behalf of the University          #
# of Colorado Colorado Springs.  All rights reserved.                                             #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without modification,                #
# are permitted provided that the following conditions are met:                                   #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
# list of conditions and the following disclaimer.                                                #
#                                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice, this list          #
# of conditions and the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                                                 #
#                                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors may be            #
# used to endorse or promote products derived from this software without specific prior           #
# written permission.                                                                             #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY             #
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          #
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF     #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,           #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS           #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
#                                                                                                 #
# If you use this code, please cite the following works                                           #
#                                                                                                 #
# A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on                        #
# Computer Vision and Pattern Recognition (CVPR), 2016                                            #
#                                                                                                 #
# Notice Related to using LibMR.                                                                  #
#                                                                                                 #
# If you use Meta-Recognition Library (LibMR), please note that there is a                        #
# difference license structure for it. The citation for using Meta-Recongition                    #
# library (LibMR) is as follows:                                                                  #
#                                                                                                 #
# Meta-Recognition: The Theory and Practice of Recognition Score Analysis                         #
# Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult                     #
# IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695                                     #
#                                                                                                 #
# Meta recognition library is provided with this code for ease of use. However, the actual        #
# link to download latest version of LibMR code is: http://www.metarecognition.com/libmr-license/ #
###################################################################################################


import scipy as sp
import sys
import os, glob
import os.path as path
import scipy.spatial.distance as spd
from scipy.io import loadmat, savemat
import json
import torch
import numpy as np


def compute_channel_distances(mean_vector, features):

    mean_vector = mean_vector.data.numpy()

    eu, cos, eu_cos = [], [], []
    
    for feat in features:
        feat = feat[0].data.numpy()

        eu.append(spd.euclidean(mean_vector, feat))
        cos.append(spd.cosine(mean_vector, feat))
        eu_cos.append(spd.euclidean(mean_vector, feat)/200. +
                            spd.cosine(mean_vector, feat))
        
    eu_dist = np.array(eu)
  
    cos_dist = np.array(cos)

    eucos_dist = np.array(eu_cos)


    channel_distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean':eu_dist}
    return channel_distances
    

def compute_distances(cls_indx,mavfilepath,featurefilepath):
   
    mean_feature_vec = torch.from_numpy(np.load(os.path.join(mavfilepath,cls_indx+".npy")))
    
    featurefile_list = os.listdir(os.path.join(featurefilepath,cls_indx))

    correct_features = []
    for featurefile in featurefile_list:
        
        feature = torch.from_numpy(np.load(os.path.join(featurefilepath,cls_indx,featurefile)))

        predicted_category = torch.max(feature,dim=1)[1].item()
        
        if(predicted_category == category_index):
            correct_features.append(feature)


    distance_distribution = compute_channel_distances(mean_feature_vec, correct_features)
    return distance_distribution

def get_args():
    parser = argparse.ArgumentParser(description='Get activation vectors')
    parser.add_argument('--MAV_path',default="./saved_MAVs/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--save_path',default="./saved_distance_scores/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--feature_dir',default="./saved_features/cifar10",type=str,help="Path to save the ensemble weights")
    parser.set_defaults(argument=True)

    return parser.parse_args()

def main():

    args = get_args()

    for class_no in os.listdir(args.feature_dir):
        print("Class index ",class_no)
        distance_distribution = compute_distances(class_no,args.MAV_path,args.feature_dir)
        np.save(os.path.join(args.save_path,class_no+".npy"),distance_distribution)

if __name__ == "__main__":
    main()
