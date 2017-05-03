# FORKS
Finding Orderings Robustly using K-means and Steiner trees

Recent advances in single cell RNA-seq technologies have provided researchers with unprecedented details of transcriptomic variation across individual cells. However, it
has not been straightforward to infer differentiation trajectories from such data. Here, we present Finding Orderings Robustly
using K-means and Steiner trees (FORKS), an algorithm that pseudo-temporally orders cells and thereby infers bifurcating state
trajectories. FORKS, which is a generic method, can be applied to both single-cell or bulk differentiation data. It is a semi-supervised approach, in that it requires
the user to specify the starting point of the time course. We systematically benchmarked FORKS and 6 other pseudo-time estimation algorithms on 5 benchmark datasets, and
found it to be more accurate, more reproducible, faster and more memory-efficient than existing methods for pseudo-temporal ordering. Another major
advantage of our approach is that the algorithm requires no hyperparameter tuning.

## Code
The code is written in Python 3.5 and requires the following packages
* sklearn,
* scipy,
* seaborn,
* matplotlib, 
* numpy 
* pandas

All packages except seaborn can be found in Anaconda python installation.

## Examples
Here we present the algorithm and three of datasets namely, 
* arabidopsis
* deng_2014  
* guo_2010 
for users to test their code on.

## Citation
If you use the code please cite the following paper using the bibtex entry:
'''
@article {Sharma132811,
	author = {Sharma, Mayank and Li, Huipeng and Sengupta, Debarka and Prabhakar, Shyam and Jayadeva, Jayadeva},
	title = {FORKS: Finding Orderings Robustly using K-means and Steiner trees},
	year = {2017},
	doi = {10.1101/132811},
	publisher = {Cold Spring Harbor Labs Journals},
	URL = {http://biorxiv.org/content/early/2017/05/02/132811},
	eprint = {http://biorxiv.org/content/early/2017/05/02/132811.full.pdf},
	journal = {bioRxiv}
}
'''
## Research Paper
The paper for the same is available at:

http://biorxiv.org/content/early/2017/05/02/132811
