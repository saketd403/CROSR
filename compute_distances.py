
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
    parser.add_argument('--feature_dir',default="./saved_features/cifar10/train",type=str,help="Path to save the ensemble weights")
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
