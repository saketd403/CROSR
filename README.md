# CROSR

PyTorch implementation for "Classification-Reconstruction Learning for Open-Set Recognition" CVPR 2019.
It is important to note that this repository borrows heavily from the repo - https://github.com/abhijitbendale/OSDN
as far as fitting weibull distribution and computing open max scores goes. It also borrows from the original repo for 
the paper - https://nae-lab.org/~rei/research/crosr (which is implemented in chainer). I think thsi repository is much 
cleaner and flexible to accomodate newer datasets and model architectures. Some part of code works on python 3 and 2 whereas
the other portion only works on python 2.7. Please follow the following steps to successfully run the code. 

## Usage

#### 1) Compiling LibMR

Compile LibMR and python interface to LibMR using following commands.
For pythong interfaces to work, you would require Cython to be pre-installed
on your machine
```bash
cd libMR/
chmod +x compile.sh
./compile.sh
```

#### 2) Train the DHRNet

```
python train_net.py
```

#### 3) Compute the activation vectors for images

```
python get_model_features.py
```

#### 4) Compute the MAV (mean activation vector) for each class category

```
python MAV_Compute.py
```

#### 5) Compute the distance scores for activation features of training set

```
python compute_distances.py
```

#### 6) Fit Weibull distribution for each category and calculate openmax scores (Note that this code needs to be run in Python 2.7.)

```
python compute_openmax.py
```

## Results

The AUROC score for CIFAR-10 with 6/4 split is 71.23.
