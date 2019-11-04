### **Understanding Isomorphism Bias in Graph Data Sets**

### Intro
This code has two main components:
  1. Finding all isomorphism pairs in a graph data set.
  2. Running classification models. 
  
Given proper format of graphs, one can find all graph orbits in a data set efficiently (i.e. groups of graphs that are pairwise isomorphic). There are two types of classification models, Neural Networks (GIN) and Graph Kernels (WL, Histograms, RW). 

### Find isomorphic graphs with nauty
**Dependencies**: 
* nauty26r11 (included archive)
* gcc compiler
* networkx 

First **install nauty**.
We use version of nauty 26r11 (as of 12 Aug 2019). Full documentation is available here: http://pallini.di.uniroma1.it/. 

Archive with nauty is included in the repo, so just clone this repo and do: 
```
cd nauty
tar xvzf nauty26r11.tar.gz
cd nauty26r11
./configure
make
```

Graph data sets may contain isomorphic copies of graphs, in which case you may want to clean to run fair comparison of classification models. 
**Note:** for data sets from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets the cleaning procedure was already done and data sets can be found cleaned in diffierent formats (see section below). 
If you have your own data set, you should format it properly. Consult https://github.com/nd7141/graph_datasets on the format of graph data sets. 
Example of cleaning. Download a zipped archive of the dataset of interest, for example `MUTAG.zip`. To keep folder clean, create a folder `datasets/` and unzip a dataset to it.

File `preprocessing.py` contains a function to transform this folder to necessary nauty format and also has a function to convert it to `networkx` graphs. 
Provide a name of the dataset in the script and run the file.  
```
python preprocessing.py 
```
At the end you should get `datasets/data_adj/MUTAG_adj/` with the graphs that you can provide to nauty.

Next you can call `main.py` to find isomorphic providing the name for you data set. The program will generate all pairs that are isomorphic in a data set. The file uses `multiprocessing` to process many pairs in parallel. File `nauty/main.py` contains experiments to (1) get generating groups of a given graph (just for curiosity; no need for later purposes) and (2) test isomorphism of two graphs. 
In experiment (1) the generating groups are saved to the file in a cyclic form. You can use `groups.py` to save discovered groups in a nice format. In experiment (2) there are two ways to run test. One way is to run tests in parallel for all pairs in a dataset.
This works faster, but sometimes crashes. Another way is to test it sequentially, it works slower, but is easier to debug. 


Next, you can run `postprocessing.py`, which will generate human-readable file for orbits found in a data set, i.e. a group of graphs in a data set that are isomorphic to each other.  

 
### Run graph classification models

Clean data sets are provided in (https://github.com/nd7141/graph_datasets). Additionally, data sets are available in https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.TUDataset
A dataset should be downloaded the first time and then stored locally in your computer so that next run
it does not download it. 
`kernel` folder additionally converts these datasets to graphml format. 

#### Run NN model for graph classification
**Dependencies**:
* torch
* torch_geometric

`gin` folder includes implementation of NN model for graph classification
`train.py` runs the model to get k-fold cross-validation on evaluation. 

#### Run Graph Kernel model for graph classification
**Dependencies**
* torch_geometric 
* sklearn
* igraph
* networkx
* gcc

Models output accuracies on Y_test, homogeneous and all Y_iso.  
