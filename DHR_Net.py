import torch
import torch.nn as nn
import torch.nn.functional as F

class DHRNet(nn.Module):

    def __init__(self,num_classes):
        super(DHRNet,self).__init__()
        self.num_classes = num_classes

        self.conv1_1 =  nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 =  nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)


        self.conv2_1 =  nn.Conv2d(64, 128, kernel_size=3,
                               stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 =  nn.Conv2d(128, 128, kernel_size=3,
                               stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)


        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3,
                               stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3,
                               stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)


        self.btl1 = nn.Conv2d(64, 32, kernel_size=3,
                               stride=1, padding=1)
        self.btlu1 = nn.Conv2d(32, 64, kernel_size=3,
                               stride=1, padding=1)
        self.btl2 = nn.Conv2d(128, 32, kernel_size=3,
                               stride=1, padding=1)
        self.btlu2 = nn.Conv2d(32, 128, kernel_size=3,
                               stride=1, padding=1)
        self.btl3 = nn.Conv2d(256, 32, kernel_size=3,
                               stride=1, padding=1)
        self.btlu3 = nn.Conv2d(32, 256, kernel_size=3,
                               stride=1, padding=1)


        self.fc4 = nn.Linear(4*4*256, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.fc6 = nn.Linear(4096, self.num_classes)


        self.deconv1 = nn.ConvTranspose2d(64,3,kernel_size=2,stride=2,padding=0)
        self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,padding=0)
        self.deconv3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,padding=0)
    
    def forward(self,x):

        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        x1 = F.max_pool2d(x1,kernel_size=2,stride=2)
        x1 = F.dropout(x1,p=0.25)

        x2 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2 = F.max_pool2d(x2,kernel_size=2,stride=2)
        x2 = F.dropout(x2,p=0.25)

        x3 = F.relu(self.bn3_1(self.conv3_1(x2)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3)))
        x3 = F.max_pool2d(x3,kernel_size=2,stride=2)
        x3 = F.dropout(x3,p=0.25)
 
        x4 = x3.view(x3.size(0), -1)

        x5 = F.dropout(F.relu(self.fc4(x4)),p=0.5)
        x5 = F.dropout(F.relu(self.fc5(x5)),p=0.5)
        x5 = F.relu(self.fc6(x5))


        z3 = F.relu(self.btl3(x3))  
        z2 = F.relu(self.btl2(x2))  
        z1 = F.relu(self.btl1(x1)) 

        j3 = self.btlu3(z3)
        j2 = self.btlu2(z2)
        j1 = self.btlu1(z1)


        g2 = F.relu(self.deconv3(j3))
        g1 = F.relu(self.deconv2(j2+g2))
        g0 = F.relu(self.deconv1(j1+g1))

        return x5, g0, [z3,z2,z1]


    






