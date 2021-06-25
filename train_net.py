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
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train DHR Net')
    parser.add_argument('--lr',default=0.05,type=float,help="learning rate")
    parser.add_argument('--epochs',default=500,type=int,help="Number of training epochs")
    parser.add_argument('--batch_size',default=128,type=int,help="Batch size")
    parser.add_argument('--dataset_dir',default="./data/cifar10",type=str,help="Number of members in ensemble")
    parser.add_argument('--num_classes',default=10,type=int,help="Number of classes in dataset")
    parser.add_argument('--means',nargs='+',default=[0.4914, 0.4822, 0.4465], type=float,help="channelwise means for normalization")
    parser.add_argument('--stds',nargs='+',default=[0.2023, 0.1994, 0.2010],type=float,help="channelwise std for normalization")
    parser.add_argument('--momentum',default=0.9,type=float,help="momentum")
    parser.add_argument('--weight_decay',default=0.0005,type=float,help="weight decay")
    parser.add_argument('--save_path',default="./save_models/cifar10",type=str,help="Path to save the ensemble weights")

    parser.set_defaults(argument=True)

    return parser.parse_args()

def epoch_train(epoch_no,net,trainloader,optimizer):
        
    net.train() 
    correct=0
    total=0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    for i,data in enumerate(trainloader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
    
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, reconstruct,_ = net(inputs)

        cls_loss = cls_criterion(logits, labels)

        reconst_loss = reconst_criterion(reconstruct,inputs)
      
        if(torch.isnan(cls_loss) or torch.isnan(reconst_loss)):
            print("Nan at iteration ",iter)
            cls_loss=0.0
            reconst_loss=0.0
            logits=0.0          
            reconstruct = 0.0  
            continue

        loss = cls_loss + reconst_loss

        loss.backward()
        optimizer.step()  

        total_loss = total_loss + loss.item()
        total_cls_loss = total_cls_loss + cls_loss.item()
        total_reconst_loss = total_reconst_loss + reconst_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]
    
def epoch_val(net,testloader):

    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    with torch.no_grad():
        for data in testloader:

            images, labels = data
            images=images.cuda(non_blocking=True)
            labels=labels.cuda(non_blocking=True)

            logits, reconstruct,_ = net(images)

            cls_loss = cls_criterion(logits, labels)

            reconst_loss = reconst_criterion(reconstruct,images)
        
            loss = cls_loss + reconst_loss

            total_loss = total_loss + loss.item()
            total_cls_loss = total_cls_loss + cls_loss.item()
            total_reconst_loss = total_reconst_loss + reconst_loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]
                 


def main():

    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    args = get_args()

    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    momentum= args.momentum
    weight_decay= args.weight_decay
    means = args.means
    stds = args.stds
    

    num_classes = args.num_classes
    print("Num classes "+str(num_classes))

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.RandomAffine(degrees=30,translate =(0.2,0.2),scale=(0.75,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means,stds),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means,stds),
    ])

    root = args.dataset_dir

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(root,"train"), 
                                        transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True,pin_memory=True,drop_last=True)

    testset = torchvision.datasets.ImageFolder(root=os.path.join(root,"val"), 
                                            transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False,pin_memory=True,drop_last=True)

    net = models.DHRNet(num_classes)
    net = torch.nn.DataParallel(net.cuda())


    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(epochs):  # loop over the dataset multiple times
    
        train_acc = epoch_train(epoch,net,trainloader,optimizer)
        test_acc = epoch_val(net,testloader)
        scheduler.step()
        print("Train accuracy and cls, reconstruct and total loss for epoch "+str(epoch)+" is "+str(train_acc))       
        print("Test accuracy and cls, reconstruct and total loss for epoch "+str(epoch)+" is "+str(test_acc))


        """
        torch.save({'epoch':epoch,
                     'model_state_dict':net.module.state_dict(),
                     'train_acc':train_acc[0],
                     'train_loss':train_acc[3],
                      'val_acc':test_acc[0] ,
                      'val_loss':test_acc[3]},
          "./save_models/vanilla_dhr/checkpoints/"+str(epoch)+".pth")
        """

if __name__ == "__main__":
    main()
    