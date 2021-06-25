
import os, sys
import glob
import time
import scipy as sp
from scipy.io import loadmat, savemat
import pickle
import os.path as path
import torch
import numpy as np


def compute_mean_vector(category_index,save_path,featurefilepath,):
    
    featurefile_list = os.listdir(os.path.join(featurefilepath,category_index))
    
    correct_features = []
    for featurefile in featurefile_list:
        
        feature = torch.from_numpy(np.load(os.path.join(featurefilepath,folder_name,featurefile)))

        predicted_category = torch.max(feature,dim=1)[1].item()
        
        if(predicted_category == category_index):
            correct_features.append(feature)
        
    correct_features = torch.cat(correct_features,0)

    mav = torch.mean(correct_features,dim=0)

    np.save(os.path.join(save_path,folder_name+".npy"),mav.data.numpy(),allow_pickle=False)

def get_args():
    parser = argparse.ArgumentParser(description='Get activation vectors')
    parser.add_argument('--save_path',default="./saved_MAVs/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--feature_dir',default="./saved_features/cifar10",type=str,help="Path to save the ensemble weights")
    parser.set_defaults(argument=True)

    return parser.parse_args()


def main():
    args = get_args()

    for class_no in os.listdir(args.feature_dir):
        print("Class index ",class_no)
        compute_mean_vector(class_no,args.save_path,args.feature_dir)

if __name__ == "__main__":
    main()

