### Install nauty 
We use latest version of nauty 26r11 (as of 12 Aug 2019). Full documentation is available here: http://pallini.di.uniroma1.it/. 

Archive with nauty is included in the repo, so just clone this repo and do: 
```
tar xvzf nauty26r11.tar.gz
cd nauty26r11
./configure
make
```

### Download datasets
Most (if not all) of the datasets can be obtained from here: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
Download a zipped archive of the dataset of interest, for example `MUTAG.zip`. To keep folder clean, create a folder `datasets/` and unzip a dataset to it.

File `preprocessing.py` contains a function to transform this folder to necessary nauty format. Provide a name of the dataset in the script and run the file.  
```
python preprocessing.py 
```
At the end you should get `datasets/data_adj/MUTAG_adj/` with the graphs. 

### Run pairwise isomorphism test

File `main.py` contains experiments to (1) get generating groups of a given graph and (2) test isomorphism of two graphs. 
In experiment (1) the generating groups are saved to the file in a cyclic form. You can use `groups.py` to save discovered groups in a nice format.
In experiment (2) there are two ways to run test. One way is to run tests in parallel for all pairs in a dataset.
This works faster, but sometimes crashes. Another way is to test it sequentially, it works slower, but is easier to debug. 

File `postprocessing.py` cluster isomorphic graphs together to analyze isomorphic graphs easily. 
