import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import DHR_Net as models
import numpy as np
import pickle
import os
from PIL import Image

def epoch_anomalous(net,save_path,root):

    net.eval()

    with torch.no_grad():
        for folder in os.listdir(os.path.join(root,"open_set")):
            count=0
            
            for file_name in os.listdir(os.path.join(root,"open_set",str(folder))):

                if(count>=120):
                    break
                count = count + 1

                
                image = Image.open(os.path.join(root,"open_set",str(folder),file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image,0)
                image=image.cuda(non_blocking=True)
                logits, _, latent = net(image)

                squeezed_latent = []
                squeezed_latent.append(logits)
                for layer in latent:
                    m = nn.AdaptiveAvgPool2d((1,1))
                    new_layer = torch.squeeze(m(layer))
                    squeezed_latent.append(new_layer)
                
                feature = torch.cat(squeezed_latent,1)

                
                save_name = file_name.split(".")[0]
                print(save_name)
                np.save(os.path.join(save_path,"open_set",str(folder),save_name+".npy"),feature.cpu().data.numpy(),allow_pickle=False)
                


def epoch_train(net,save_path,root):

    net.eval()

    with torch.no_grad():
        for folder in os.listdir(os.path.join(root,"train")):

            
            for file_name in os.listdir(os.path.join(root,"train",str(folder))):

                image = Image.open(os.path.join(root,"train",str(folder),file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image,0)
                image=image.cuda(non_blocking=True)
                logits, _, latent = net(image)

                squeezed_latent = []
                squeezed_latent.append(logits)
                for layer in latent:
                    m = nn.AdaptiveAvgPool2d((1,1))
                    new_layer = torch.squeeze(m(layer))
                    squeezed_latent.append(new_layer)
                
                feature = torch.cat(squeezed_latent,1)

                
                save_name = file_name.split(".")[0]
                print(save_name)
                np.save(os.path.join(save_path,"train",str(folder),save_name+".npy"),feature.cpu().data.numpy(),allow_pickle=False)
                
                
           

def epoch_val(net,save_path,root):

    net.eval()

    with torch.no_grad():
        for folder in os.listdir(os.path.join(root,"val")):

           
            for file_name in os.listdir(os.path.join(root,"val",str(folder))):

                image = Image.open(os.path.join(root,"val",str(folder),file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image,0)
                image=image.cuda(non_blocking=True)
                logits, _, latent = net(image)

                squeezed_latent = []
                squeezed_latent.append(logits)
                for layer in latent:
                    m = nn.AdaptiveAvgPool2d((1,1))
                    new_layer = torch.squeeze(m(layer))
                    squeezed_latent.append(new_layer)
                
                feature = torch.cat(squeezed_latent,1)

                
                save_name = file_name.split(".")[0]
                print(save_name)
                np.save(os.path.join(save_path,"val",str(folder),save_name+".npy"),feature.cpu().data.numpy(),allow_pickle=False)
                 
def get_args():
    parser = argparse.ArgumentParser(description='Get activation vectors')
    parser.add_argument('--dataset_dir',default="./data/cifar10",type=str,help="Number of members in ensemble")
    parser.add_argument('--num_classes',default=10,type=int,help="Number of classes in dataset")
    parser.add_argument('--means',nargs='+',default=[0.4914, 0.4822, 0.4465],type=float,help="channelwise means for normalization")
    parser.add_argument('--stds',nargs='+',default=[0.2023, 0.1994, 0.2010],type=float,help="channelwise std for normalization")
    parser.add_argument('--save_path',default="./saved_features/cifar10",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--load_path',default="./saved_models/cifar10/50.pth",type=str,help="Path to save the ensemble weights")
    parser.set_defaults(argument=True)

    return parser.parse_args()

def main():

    args = get_args()

    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    num_classes = args.num_classes
    print("Num classes "+str(num_classes))


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.means, args.stds),
    ])

    root = args.dataset_dir

    net = models.DHRNet(num_classes)
    checkpoint = torch.load(args.load_path,map_location="cpu")
    net.load_state_dict(checkpoint)
    net.cuda()
 
    epoch_train(net,args.save_path,root)
    epoch_val(net,args.save_path,root)
    epoch_openset(net,args.save_path,root)


if __name__=="__main__":
    main()
    