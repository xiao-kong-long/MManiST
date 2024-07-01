# Multi-Manifolds Fusing Hyperbolic Graph Network Balanced by Pareto Optimization for Identifying Spatial Domains of Spatial Transcriptomics


## Requirements

**Note:** due to rpy package, the project is just available on Linux system. For windows user, we design script version, which is able to achieve similar performance. We will provider later.

To install python environment:

```setup
conda env create -f env.yml
```
File ```env.yml``` recodes the path of anacond environment, default ```/root/anaconda3/envs```, if your anaconda path is different to it, please change corresponding content in ```env.yml```.

To install R environment:

1. install R 4.3.0 on linux system
```
sudo sh -c 'echo "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran43/" >> /etc/apt/sources.list.d/r.list'

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9

sudo apt update

sudo apt install r-base=4.3.0-1.2004.0
```
2. install mclust package on R
```
R
install.packages("mclust")
library(mclust)
quit()
```


## Training

To train the model(s) in the paper, run this command:

```train
python train.py
```
File ```train.py``` records default hyperparemeters that analyze DLPFC section 151507. If you want to analyze other sections or datasets, please download dataset and change hyperparameter defination on ```train.py```.


## Results
Due to limitation on size of supplementary materials, here we only provide section 151507 dataset for presentation. Details of more other datasets are available at supplementary document.
Our model achieves the following performance on :

### Clustering Performance on DLPFC section 151507

| Model name         | ARI             | NMI            |
| ------------------ |---------------- | -------------- |
|   MManiST          |     0.56        |      0.71      |

Visualization result is available in ```output/10x Visium/DLPFC/151507/comprehensive```



## Contributing

All contributions welcome! All content in this repository is licensed under the MIT license.