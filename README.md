### Install nauty 
We use version of nauty 26r11 (as of 12 Aug 2019). Full documentation is available here: http://pallini.di.uniroma1.it/. 

Archive with nauty is included in~~~~ the repo, so just clone this repo and do: 
```
cd nauty
tar xvzf nauty26r11.tar.gz
cd nauty26r11
./configure
make
```

### Download datasets for nauty
Most (if not all) of the datasets can be obtained from here: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
Download a zipped archive of the dataset of interest, for example `MUTAG.zip`. To keep folder clean, create a folder `datasets/` and unzip a dataset to it.

File `preprocessing.py` contains a function to transform this folder to necessary nauty format and also has a function to convert it to `networkx` graphs. 
Provide a name of the dataset in the script and run the file.  
```
python preprocessing.py 
```
At the end you should get `datasets/data_adj/MUTAG_adj/` with the graphs.
 
### Datasets for running models
Datasets for models are taken from PyTorch-Geometric: https://github.com/rusty1s/pytorch_geometric
A dataset should be downloaded the first time and then stored locally in your computer so that next run
it does not download it. 
`kernel` folder additionally converts these datasets to graphml format. 

### Run pairwise isomorphism test
**Dependencies**: 
* nauty26r11 (included archive)
* gcc compiler
* networkx 

You first need to get datasets in the necessary formats (see above).

File `nauty/main.py` contains experiments to (1) get generating groups of a given graph and (2) test isomorphism of two graphs. 
In experiment (1) the generating groups are saved to the file in a cyclic form. You can use `groups.py` to save discovered groups in a nice format.
In experiment (2) there are two ways to run test. One way is to run tests in parallel for all pairs in a dataset.
This works faster, but sometimes crashes. Another way is to test it sequentially, it works slower, but is easier to debug. 

File `postprocessing.py` cluster isomorphic graphs together to analyze isomorphic graphs easily. 

### Run NN model for graph classification
**Dependencies**:
* torch
* torch_geometric

`gin` folder includes implementation of NN model for graph classification
`train.py` runs the model to get k-fold cross-validation on evaluation. 

### Run Graph Kernel model for graph classification
**Dependencies**
* torch_geometric 
* sklearn
* igraph
* networkx
* gcc

Models output accuracies on Y_test, homogeneous and all Y_iso.  