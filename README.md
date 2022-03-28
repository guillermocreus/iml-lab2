# Lab 2

The purpose of this report is to gain a better understanding of how the principal component analysis (PCA) al-
gorithm and the Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP) algorithm
work and obtain practical experience with these algorithms. The report will go through the main steps in the PCA
algorithm. Then, it will compare the PCA function developed for this report to the already available PCA functions
in the sklearn [4] library. These will then be compared to the UMAP function found in the umap-learn [2] library.

One of the main objectives of this report is to see how reducing a data set affects the performance, visualization, and
runtime of a clustering task done by our KMeans implementation (improved with KMeans++ [1]). Our reference
method will use the data set with d features, and we will compare it with our own implementation of PCA, sklearn
PCA, sklearn incremental PCA and UMAP. These dimensionality reduction methods will reduce the data set to 2
dimensions so we can plot it and, hopefully, cluster it better.

The data sets that used for this report were the Hypothyroid, Wines, and Cmc. This was due to a requirement of
using the same data sets as a previous report. At the end of the report, it is expected that a better understanding
of the PCA and UMAP algorithms is achieved. This includes a better understanding of how they work, what are
the expected outcomes from them, how they relate to previous reports that were written, and how they compare
to each other

## Report

A more detailed analysis of this task is shown in this [document](./report.pdf).

## Instructions

Steps to run the script

### Running script for the first time
This section shows how to create a virtual environment to run the scripts from this project
1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running the following command (installed dependencies should be displayed)
```bash
pip list
```

5. Close virtual env
```bash
deactivate
```

### Execute scripts

1.open virtual env
```bash
source venv/bin/activate
```
2. Running the script

	2.1.  For **wines dataset** execute
	```bash
	python3 main_wines.py
	```

	2.2. For **hypothyroid dataset** execute
	```bash
	python3 main_hypo.py
	```
 	2.3 For **cmc dataset** execute
	```bash
	python3 main_cmc.py
	```

3. Close virtual env
```bash
deactivate
```

# Authors

Nikita Belooussov, Valeriu Vicol and Guillermo Creus
