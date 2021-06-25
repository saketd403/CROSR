# CROSR

PyTorch implementation for "Classification-Reconstruction Learning for Open-Set Recognition" CVPR 2019.
It is important to note that this repository borrows heavily from the repo - https://github.com/abhijitbendale/OSDN
as far as fitting weibull distribution and computing open max scores goes. It also borrows from the original repo for 
the paper - https://nae-lab.org/~rei/research/crosr (which is implemented in chainer). I think thsi repository is much 
cleaner and flexible to accomodate newer datasets and model architectures.

## Usage

Compile LibMR and python interface to LibMR using following commands.
For pythong interfaces to work, you would require Cython to be pre-installed
on your machine
```bash
cd libMR/
chmod +x compile.sh
./compile.sh
```
