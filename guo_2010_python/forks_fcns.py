# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:57:55 2016

@author: mayank
"""

import numpy as np
import pandas as pd
import scipy as sp
#from ggplot import *
#del chopsticks
#del diamonds
#del meat
#del movies
#del mpg
#del mtcars
#del pageviews
#del pigeons
#del salmon
import os
from sklearn.decomposition import PCA,FactorAnalysis
from k_medoids import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score,pairwise,calinski_harabaz_score
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn import manifold
from time import time
from sklearn.cluster.bicluster import SpectralBiclustering
import hdbscan
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import queue
import seaborn as sns
from scipy.special import expit
#from fancyimpute import BiScaler, KNN, AutoEncoder, SoftImpute
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
sns.set_style("whitegrid")
#from tsne import *
#from bhtsne import *
#print os.getcwd()
#path1="D:\Dropbox\phd\pseudotemporal\preimplant_python"
#os.chdir(path1)
#print os.getcwd()
#def clear_all():
#    gl=globals().copy()
#    for var in gl:
#        if var[0]=='_':continue
#        if 'func' in str(globals()[var]):continue
#        if 'module' in str(globals()[var]):continue
#        del globals()[var]
#clear_all()
#data=np.array([[5,2,3,4],[4,1,4,2],[3,4,6,8]])

#%%
def find_unique_gene_cells(gene_name,cell_names):
    #gene_names and cell_names are pd.Series type
    gene_names=list(gene_name)
    cell_names=list(cell_names)
    unique_genes=list(pd.Series(pd.unique(gene_names)))
    unique_cells=list(pd.Series(pd.unique(cell_names)))
    idx_genes=list([])
    idx_cells=list([])               
    for item1 in list(unique_genes):
        for i, item2 in enumerate(list(gene_names)):
            if(item1==item2):
                idx_genes.append(i)
                break
    for item1 in list(unique_cells):
        for i, item2 in enumerate(list(cell_names)):
            if(item1==item2):
                idx_cells.append(i)
                break
    return idx_genes,idx_cells
#%%
def find_closest_cluster_center(cluster_centers,data,labels,starting_cell):
    neigh = NearestNeighbors(n_neighbors=1,algorithm='auto', leaf_size=30)
    neigh.fit(cluster_centers) 

    if(labels.size!=0):
        temp=labels==np.unique(labels)[0]
        starting_cells=data[temp,:]
        starting_cell=starting_cells[0,:]
        distances,indices=neigh.kneighbors([starting_cell])
    elif(starting_cell.size!=0):
        distances,indices=neigh.kneighbors([starting_cell])
    else:
        indices=np.zeros((0,0))
        print('Wrong imputs!')
    return indices
#%%
def select_marker_genes(data,gene_name,marker_genes):
    ##select only the data belonging to the marker genes
#    gene_name and marker genes is pd.series
    gene_idx=[i for i, item in enumerate(list(gene_name)) if item in set(list(marker_genes.iloc[0:,0]))]
    data2=data[:,gene_idx]
    return data2,gene_idx
#%%
def gene_selection(data,fig_title,frac_of_cells,is_std,std_frac):
    #data should be of the form samples X features
    #select all the genes whose count is >0 across all cells and convert it into a vector
    data_vec=np.log2(data[data>0.0])
    # the histogram of data
    xlabel='gene expression'
    ylabel='counts'
    hist_plot(data_vec,fig_title,xlabel,ylabel)
    quantile_25_gene_expr=np.power(2,np.percentile(data_vec,25))
    std_gene_expr=np.power(2,np.std(data_vec))
    std_expressed_genes=np.zeros((data.shape[1],))
    temp_quantile=np.zeros((data.shape[1],),dtype='bool')
    temp_std=np.zeros((data.shape[1],),dtype='bool')
    for i in range(0,data.shape[1]):
        std_expressed_genes[i]=np.std(data[data[:,i]>0.0,i])
        temp_quantile[i]=np.divide(np.sum(data[:,i]>quantile_25_gene_expr),data.shape[0],dtype='float64')>frac_of_cells 
        temp_std[i]= (std_expressed_genes[i]>std_frac*std_gene_expr)
    if(is_std==True):
        genes_selected=temp_quantile*temp_std #select highly varying and highly expressed genes
    else:
        genes_selected=temp_quantile
    print ('genes selected = %d'%np.sum(genes_selected))
    return genes_selected
    
#%%
def index_to_val(statistics,data_rank):
    temp=np.zeros(data_rank.shape)
    for i in range(0,data_rank.shape[0]):
        temp[i,]=statistics[data_rank[i,]]
    return temp
def quantile_normalization(data,use_mean,use_median,use_quantile,quantile):
    #data should be of the form samples X features
    data_sorted=np.sort(data,axis=1)#sort in increasing order across features for each sample
    data_rank=np.argsort(np.argsort(data,axis=1),axis=1)#get the rank of individual features for each sample
    if(use_mean):
        me=np.mean(data_sorted,axis=0)
        data_final=index_to_val(me,data_rank)
    elif(use_median):
        med=np.median(data_sorted,axis=0)
        data_final=index_to_val(med,data_rank)
    elif(use_quantile):
        Q=np.percentile(data,quantile,axis=0)
        data_final=index_to_val(Q,data_rank)
    else:
        data_final=data
        print('wrong options provided')
    return data_final
    
#%%
def remove_zero_rc(data):
    #data should be in the form samples X features
    #remove genes with zero count in all cells
    gene_list=np.sum(data>0, axis=0)>0.0
    data=data[:,gene_list]
    #find the cells that have genes greater than th+1
    cell_list=np.sum(data>0, axis=1)>0.0
    data=data[cell_list,]  
    return data,gene_list,cell_list

def hist_plot(data,fig_title,xlabel,ylabel):
    n, bins, patches = plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(fig_title)
    plt.grid(True)
    plt.show()
    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
    plt.close()
    return
#%%
def normalization_spherize(data):
    # we scale the data to have zero mean and unit variance
    #data should be of the form samples X features
    return preprocessing.scale(data)   
#%%
def normalization_maxscale(data):
    # we scale the data to be between zero and one
    # data should be of the form samples X features
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)
#%%
def normalization_pQ(data,use_mean,use_median,use_quantile,quantile):
    #we use pQ normalization to discard cells and genes and then do normalization
    # data  should be of the form samples X features
    # cells whose detected gene count was < Q_1 - frac x IQR (Inter Quartile range) are discarded as low quality
    # expression profile of remaining cells are then quantile normalized by ranking genes in each cell according to their expression
    # and then assigning to each gene the median across all cells of genes with same rank
    # then define the number of detected genes as d_min 
    # genes with lower rank than d_min are assigned a pseudocount equal to quantile normalized expression of d_min ranked gene
    frac=0.5
    if(data.shape[0]>500):
        hard_outlier=500
    else:
        hard_outlier=50
    th=threshold_pQ(data,frac,hard_outlier)
    print('data shape before discarding (%d,%d)'%data.shape)
    print ('threshold for detected genes is %d'%th)
    #remove genes with zero count in all cells
    gene_list=np.sum(data>0, axis=0)>0.0
    data=data[:,gene_list]
    #find the cells that have genes greater than th+1
    if(th<0.9*data.shape[1]):#just to prevent discarding a lot of cells
        print('here')
        cell_list=np.sum(data>0, axis=1)>(th+1)      
        data=data[cell_list,]
    else:
        cell_list=np.sum(data>0, axis=1)>0.0
    
    #once we discarded cells we need to discard genes
#    d_min=np.min(np.sum(data>0, axis=1))
    #do a Quantile normalization
#    robust_scaler = preprocessing.RobustScaler()
#    data = robust_scaler.fit_transform(data)
    data=quantile_normalization(data,use_mean,use_median,use_quantile,quantile=75)

    #for each cell sort the genes in desc order
    # for each sorted cell [i], for index th+1, find the gene value denoted by key[i]
    # replace all the genes in the cell [i] with values less than key to key
    if(th<0.9*data.shape[1]):#just to prevent discarding a lot of cells
        key=-np.sort(-data,axis=1)[:,th+1]
        for i in range(0,len(key)):
            data[i,data[i,:]<key[i]]=key[i]
    #remove zero std genes
    data_std=np.std(data, axis=0)
    data=data[:,data_std>0.0]
    gene_list=data_std>0.0
     
    print('data shape after discarding (%d,%d)'%data.shape)
    return data,cell_list,gene_list
#%%    
def threshold_pQ(data,frac,hard_outlier):
    #get number of detected genes
    detected_genes=np.sum(data>0,axis=1)
    Q1=np.percentile(detected_genes,25)
    Q3=np.percentile(detected_genes,75)
    IQR=Q3-Q1
    th=int(np.floor(Q1-frac*IQR))
    if(th<hard_outlier):
        th=hard_outlier    
    return th
#%%
#def add_legend(colormap,labels):
#    cmap = getattr(mpl.cm, colormap)
#    unique_labels=np.unique(labels)
#    n_colors=unique_labels.shape[0]
#    bins = np.linspace(0, 1, n_colors + 2)[1:-1]
#    palette = list(map(tuple, cmap(bins)[:, :4]))
#    legend1_line2d=list()
#    for color in palette:
#        legend1_line2d.append(mlines.Line2D([0], [0], linestyle='none', marker='o', alpha=1, markersize=5, markerfacecolor=color))
#    return legend1_line2d,unique_labels
def add_legend(colormap,labels):
    #labels is pd.Series
    cmap = getattr(mpl.cm, colormap)
    unique_labels=np.unique(labels)
    n_colors=unique_labels.shape[0]
    bins = np.linspace(0, 1, n_colors + 2)[1:-1]
    palette = list(map(tuple, cmap(bins)[:, :4]))
    legend_line2d=list()
    for color in palette:
        legend_line2d.append(mlines.Line2D([0], [0], linestyle='none', marker='o', alpha=1, markersize=5, markerfacecolor=color))
    scatter_colors=list([])
    for i in range(labels.shape[0]):
        for j in range(unique_labels.shape[0]):
            if(labels[i]==unique_labels[j]):
                scatter_colors.append(palette[j])
    return legend_line2d,unique_labels,palette,n_colors,scatter_colors
    
def get_corr_with_det_genes(data,detected_genes,labels,fig_title1,fig_title2):
    #data should be of the form samples X features
    min_comp=np.min(data.shape)
    pca = PCA(n_components=min_comp,whiten=True)
    pca.fit(data.T)
#    pca_score = pca.explained_variance_ratio_
    V = pca.components_
#    data_reduced=pca.fit_transform(data.T)
#    detected_genes=np.sum(data>0,axis=1)
    corr=np.zeros((min_comp))
    for i in range(0,V.shape[0]):
        corr[i]=sp.stats.pearsonr(V[i,], detected_genes)[0]
##    plt.plot(corr)
    fig = plt.figure()
#    fig.patch.set_alpha(0.5)
    ax = fig.add_subplot(111)
    plt.bar(range(1,V.shape[0]+1), corr)
    ax.xaxis.set_major_locator(MaxNLocator(11))
    plt.xlabel('PCA components')
    plt.ylabel('Pearson correlation')
    plt.title(fig_title1)
    plt.grid(True)
    plt.show()
    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title1+'.pdf', bbox_inches='tight')
    plt.close()
    #now reduce the gene space to 2
    pca = PCA(n_components=2,whiten=True)
    pca.fit(data)
    data_reduced=pca.fit_transform(data)
    print('size of labels is %f'%(labels.size))
#    print(labels.size)
    if(labels.size!=0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colormap='rainbow' 
        shadow=False
        fancybox=False
        frameon=True
        print([shadow,fancybox,frameon])
        labels=pd.Series(labels)
        legend_line2d,unique_labels,palette,n_colors,scatter_colors=add_legend(colormap,labels)
        scplot=ax.scatter(data_reduced[:,0], data_reduced[:,1], c=list(scatter_colors),s=20,alpha=1.0,edgecolors='face')
        legend1 = ax.legend(legend_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=shadow,fancybox=fancybox,frameon=frameon)
    else:
        scplot=ax.scatter(data_reduced[:,0], data_reduced[:,1], s=20,alpha=1.0,edgecolors='face')
    ax.set_title(fig_title2)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y_axis')
    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title2+'.pdf', bbox_inches='tight')
    plt.close()
    return corr
    
#%%
def remove_outlier_genes(data,min_cluster_size,metric):
    #we run HDBSCAN algorithm to find the outlier genes
    #data should be of the form samples X features
    # we then take transpose of the data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,metric=metric)
    cluster_labels = clusterer.fit_predict(data.T)
    data=data[:,cluster_labels!=-1]
    return data,cluster_labels
#%%
def normalization_binarization(data):
    # we comvert the matrix to a binary matrix
    # data should be of the form samples X features
    binarizer = preprocessing.Binarizer().fit(data)
    return binarizer.transform(data)
#%%
#def impute_data(data, impute_params):
#    knn=impute_params['knn']
#    impute_type=impute_params['type']
#    frac=0.5
#    hard_outlier=500
#    #remove low quality cells
#    th=threshold_pQ(data,frac,hard_outlier)
#    #find the cells that have genes greater than th+1
#    cell_list=np.where(np.sum(data>0, axis=1)>th+1)
#    data=data[cell_list[0],]
#    data[data==0.0]=np.nan
#    if(impute_type=='knn'):
#        # Use 3 nearest rows which have a feature to fill in each row's missing features
#        data_reconstructed = KNN(k=knn).complete(data)
#    elif(impute_type=='BiScaler'):
#        # Instead of solving the nuclear norm objective directly, instead
#        # induce sparsity using singular value thresholding
#        data_reconstructed=BiScaler().fit_transform(data)
#    elif(impute_type=='SoftImpute'):
#        #do a soft impute
#        data_reconstructed = SoftImpute().complete(data)
#    elif(impute_type=='AutoEncoder'):
#        #this will be painfully slow so don't use
#        solver = AutoEncoder(hidden_layer_sizes=None, hidden_activation="tanh",optimizer="adam",recurrent_weight=0.0)
#        data_reconstructed = solver.complete(data)
#    else:
#        print('wrong parameter choice!')
#        data_reconstructed=data        
#    return data_reconstructed
#%%
def f_statistics(X,n_clusters):
#    N=X.shape[0]
    M=X.shape[1]
    if(n_clusters==1):
        f=1.0
        return f
    clusterer = KMeans(n_clusters=n_clusters-1, random_state=0)
    cluster_labels = clusterer.fit_predict(X)
    S_k_minus_1=-clusterer.score(X)
    
    if(S_k_minus_1==0.0):
        f=1.0
        return f
        
    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(X)
    S_k=-clusterer.score(X)
    
    if(n_clusters==2 and M>1):
        alpha=(1.0-(3.0/(4.0*M)))
    else:
        alpha_old=(1.0-(3.0/(4.0*M)))
        for K in range(3,n_clusters+1):
            alpha=alpha_old+ (1-alpha_old)/6.0
            alpha_old=alpha
    
    f=S_k/(alpha*S_k_minus_1)
    return f
    
    
#%%
def find_nclusters(X,range_clusters):
    #X should be of the form samples (cells)X features(genes)
    sil_scores=np.zeros((len(range_clusters),1))
    i=0
    for n_clusters in range_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        sil_scores[i]=silhouette_avg
        i=i+1
    return np.argmax(sil_scores),np.max(sil_scores),sil_scores
def find_nclusters_CH(X,range_clusters):
    #X should be of the form samples (cells)X features(genes)
    sil_scores=np.zeros((len(range_clusters),1))
    i=0
    for n_clusters in range_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = calinski_harabaz_score(X, cluster_labels)
        sil_scores[i]=silhouette_avg
        i=i+1
    return np.argmax(sil_scores),np.max(sil_scores),sil_scores
def find_nclusters_F(X,range_clusters):
    #X should be of the form samples (cells)X features(genes)
    sil_scores=np.zeros((len(range_clusters),1))
    i=0
    for n_clusters in range_clusters:
        silhouette_avg = f_statistics(X, n_clusters)
        sil_scores[i]=silhouette_avg
        i=i+1
    return np.argmin(sil_scores),np.min(sil_scores),sil_scores
#%%all mappings in 2D for now
def mapping(X,actual_time,mapping_params):
    mapping_type=mapping_params['mapping_type']
    n_neighbors=mapping_params['n_neighbors']
    n_components=mapping_params['n_components']
    isplot=mapping_params['isplot']
    gene_preprocessing_type=mapping_params['gene_preprocessing_type']
    if(mapping_type=='Isomap'):
        t0 = time()
        X_reduced = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)+'_nn'+str(n_neighbors)
        print("Isomap: %.2g sec" % (t1 - t0))
    elif(mapping_type=='MDS'):
        t0 = time()
        mds = manifold.MDS(n_components, max_iter=100, n_init=1)
        X_reduced = mds.fit_transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)
        print("MDS: %.2g sec" % (t1 - t0))
    elif(mapping_type=='PCA'):
        t0 = time()
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X_reduced = pca.transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)
        print("PCA: %.2g sec" % (t1 - t0))
    elif(mapping_type=='RandomForest'):
        t0 = time()
        hasher = RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
        X_transformed = hasher.fit_transform(X)
        # Visualize result after dimensionality reduction using truncated SVD
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X_transformed)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)+'_nEst'+str(200)
        print("RandomForest: %.2g sec" % (t1 - t0))
    elif(mapping_type=='SpectralEmbedding'):
        t0 = time()
        se = manifold.SpectralEmbedding(n_components=n_components,n_neighbors=n_neighbors)
        X_reduced = se.fit_transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)+'_nn'+str(n_neighbors)
        print("SpectralEmbedding: %.2g sec" % (t1 - t0))
    elif(mapping_type=='LLE_standard'):
        t0 = time()
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,method='standard')
        X_reduced = clf.fit_transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)+'_nn'+str(n_neighbors)
        print("LLE_standard: %.2g sec" % (t1 - t0))
    elif(mapping_type=='LLE_modified'):
        t0 = time()
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,method='modified')
        X_reduced = clf.fit_transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)+'_nn'+str(n_neighbors)
        print("LLE_modified: %.2g sec" % (t1 - t0))
    elif(mapping_type=='LLE_hessian'):
        t0 = time()
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,method='hessian')
        X_reduced = clf.fit_transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)+'_nn'+str(n_neighbors)
        print("LLE_hessian: %.2g sec" % (t1 - t0))
    elif(mapping_type=='LLE_LTSA'):
        t0 = time()
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,method='ltsa')
        X_reduced = clf.fit_transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)+'_nn'+str(n_neighbors)
        print("LLE_LTSA: %.2g sec" % (t1 - t0))
    elif(mapping_type=='tSNE'):
        t0 = time()
#        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        if(X.shape[0]<20):
            perplexity=5
        else:
            perplexity=20        
        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0,perplexity=perplexity,method='exact',verbose=0)
        X_reduced =  tsne.fit_transform(X)
        t1 = time()
        fig_title=mapping_type+'_nc'+str(n_components)
        print("tSNE: %.2g sec" % (t1 - t0))            
    else:
        print('wrong input')
        return
    if(isplot==True):
        plot_params={}
        plot_params['mapping_type']=mapping_type
        if(actual_time.size!=0):
            plot_params['label_type']='actual_time'
        else:
            plot_params['label_type']=''
        plot_params['clustering_type']=''
        plot_params['init_method']=0 #0:default, 1:kmeans, 2:kmedoids, 3:knn, 4:multiple_run
        plot_params['plot_dimensions']=2 #2 or 3 only
        plot_params['isplot']=isplot
        plot_params['gene_preprocessing_type']=gene_preprocessing_type
        plotting_data(np.array([]),X_reduced,actual_time,plot_params)
    return X_reduced

def plotting_data(x,X_reduced,labels,plot_params):
    mapping_type=plot_params['mapping_type']
    label_type=plot_params['label_type']
    clustering_type=plot_params['clustering_type']
    init_method=plot_params['init_method'] #1:kmeans, 2:kmedoids, 3:knn, 4:multiple_run
    orig_dimensions=X_reduced.shape[1]
    plot_dimensions=plot_params['plot_dimensions']
    isplot=plot_params['isplot']
    gene_preprocessing_type=plot_params['gene_preprocessing_type']
    if(isplot==True):
        if(x.size==0):
            #if cluster centres are not given
            if(labels.size!=0):
                #do a scatter plot in 3D

                fig_title='%s %dD vis_dim %dD %s preproc %s init %d with %s' %(mapping_type, orig_dimensions,plot_dimensions,clustering_type,gene_preprocessing_type,init_method,label_type)

                if(orig_dimensions>=3 and plot_dimensions==3):
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection='3d')
                    colormap='rainbow'
#                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2],c=labels, cmap=colormap,s=20,alpha=1)
#                    legend1_line2d,unique_labels=add_legend(colormap,labels)
                    labels=pd.Series(labels)
                    legend_line2d,unique_labels,palette,n_colors,scatter_colors=add_legend(colormap,labels)
                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2], c=list(scatter_colors),s=20,alpha=1.0,edgecolors='face')
                    legend1 = ax.legend(legend_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=shadow,fancybox=fancybox,frameon=frameon)

                    ax.set_title(fig_title)
                    ax.set_xlabel('x-axis')
                    ax.set_ylabel('y_axis')
                    ax.set_zlabel('z_axis')
                    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
                    plt.close()
                elif(orig_dimensions>=2 and plot_dimensions==2):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    colormap='rainbow'
                    labels=pd.Series(labels)
                    legend_line2d,unique_labels,palette,n_colors,scatter_colors=add_legend(colormap,labels)
                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1], c=list(scatter_colors),s=20,alpha=1.0,edgecolors='face')
                    legend1 = ax.legend(legend_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=shadow,fancybox=fancybox,frameon=frameon)
                    
#                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1],c=labels, cmap=colormap,s=20,alpha=1)
#                    legend1_line2d,unique_labels=add_legend(colormap,labels)
#                    legend1 = ax.legend(legend1_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=shadow,fancybox=fancybox,frameon=frameon)
                    ax.set_title(fig_title)
                    ax.set_xlabel('x-axis')
                    ax.set_ylabel('y_axis')
                    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
                    plt.close()
                else:
                    print('Wrong inputs!')
            else:
                                #do a scatter plot in 2D without labels

                fig_title='%s %dD vis_dim %dD %s init %d' %(mapping_type, orig_dimensions,plot_dimensions,clustering_type,init_method)

                if(orig_dimensions>=3 and plot_dimensions==3):
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection='3d')
                    colormap='rainbow'
                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2],s=20,alpha=1)
                    ax.set_title(fig_title)
                    ax.set_xlabel('x-axis')
                    ax.set_ylabel('y_axis')
                    ax.set_zlabel('z_axis')
                    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
                    plt.close()
                elif(orig_dimensions>=2 and plot_dimensions==2):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    colormap='rainbow'
                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1],s=20,alpha=1)
                    ax.set_title(fig_title)
                    ax.set_xlabel('x-axis')
                    ax.set_ylabel('y_axis')
                    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
                    plt.close()
                else:
                    print('Wrong inputs!')
        else:
            #if cluster centres are provided
            dist_mat,MST_km1,MST_orig,w_MST=calculateMST(x)
            if(labels.size!=0):
                #do a scatter plot  with labels and cluster centers
             

                fig_title='%s %dD with MST vis_dim %dD %s init %d with %s' %(mapping_type, orig_dimensions,plot_dimensions,clustering_type,init_method,label_type)

                if(orig_dimensions>=3 and plot_dimensions==3):
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection='3d')
                    colormap='rainbow'
                    labels=pd.Series(labels)
                    legend_line2d,unique_labels,palette,n_colors,scatter_colors=add_legend(colormap,labels)
                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2], c=list(scatter_colors),s=20,alpha=1.0)
                    legend1 = ax.legend(legend_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=shadow,fancybox=fancybox,frameon=frameon)

                    
#                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2],c=labels, cmap=colormap,s=20,alpha=1)
                    for i,j in MST_orig:
                        ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],[x[i,2],x[j,2]],'-',alpha=1,color='blue')
                        ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],[x[i,2],x[j,2]],'o',color='black')
#                    legend1_line2d,unique_labels=add_legend(colormap,labels)
#                    legend1 = ax.legend(legend1_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=shadow,fancybox=fancybox,frameon=frameon)
                    ax.set_title(fig_title)
                    ax.set_xlabel('x-axis')
                    ax.set_ylabel('y_axis')
                    ax.set_zlabel('z_axis')
                    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
                    plt.close()

                elif(orig_dimensions>=2 and plot_dimensions==2):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    colormap='rainbow'
                    labels=pd.Series(labels)
                    legend_line2d,unique_labels,palette,n_colors,scatter_colors=add_legend(colormap,labels)
                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1], c=list(scatter_colors),s=20,alpha=1.0)
                    legend1 = ax.legend(legend_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=shadow,fancybox=fancybox,frameon=frameon)

                    
#                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1],c=labels, cmap=colormap,s=20,alpha=1)
                    for i,j in MST_orig:
                        ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],'-',alpha=1,color='blue')
                        ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],'o',color='black')
#                    legend1_line2d,unique_labels=add_legend(colormap,labels)
#                    legend1 = ax.legend(legend1_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=shadow,fancybox=fancybox,frameon=frameon)
                    ax.set_title(fig_title)
                    ax.set_xlabel('x-axis')
                    ax.set_ylabel('y_axis')
                    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
                    plt.close()
                    

                else:
                    print('Wrong inputs!')
            else:
                                #do a scatter plot  with no label information
#                datafr1=pd.DataFrame(X_reduced,columns=map(str, range(1,X_reduced.shape[1] + 1)))
                fig_title='%s %dD with MST vis_dim %dD %s init %d' %(mapping_type, orig_dimensions,plot_dimensions,clustering_type,init_method)

                if(orig_dimensions>=3 and plot_dimensions==3):
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection='3d')
                    colormap='rainbow'
                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2],s=20,alpha=1)
                    for i,j in MST_orig:
                        ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],[x[i,2],x[j,2]],'-',alpha=1,color='blue')
                        ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],[x[i,2],x[j,2]],'o',color='black')
                    ax.set_title(fig_title)
                    ax.set_xlabel('x-axis')
                    ax.set_ylabel('y_axis')
                    ax.set_zlabel('z_axis')
                    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
                    plt.close()
                    

                elif(orig_dimensions>=2 and plot_dimensions==2):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    colormap='rainbow'
                    scplot=ax.scatter(X_reduced[:,0], X_reduced[:,1],s=20,alpha=1)
                    for i,j in MST_orig:
                        ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],'-',alpha=1,color='blue')
                        ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],'o',color='black')
                    ax.set_title(fig_title)
                    ax.set_xlabel('x-axis')
                    ax.set_ylabel('y_axis')
                    plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
                    plt.close()
                    

                else:
                    print('Wrong inputs!')
        
    else:
        print('isPlot is set to False. \n Nothing will be plotted!')







#%% 
def calculateMST(cluster_centers):
    dist_mat=pairwise.pairwise_distances(cluster_centers, Y=None, metric='euclidean')
    dist_mat=csr_matrix(dist_mat)
    MST_km = minimum_spanning_tree(dist_mat)
    MST_km=MST_km.toarray()
    w_MST=np.sum(MST_km)
    MST_orig=np.zeros((0,2),dtype='int64')
    for i in range(0,MST_km.shape[0]):
        temp=np.where(MST_km[i]>0.0)[0]
        if(len(temp)>0):
            for j in range(0,len(temp)):
                MST_orig=np.append(MST_orig,[[i,temp[j]]],axis=0)        
    MST_km1=(MST_km+MST_km.T)
    dist_mat=dist_mat.toarray()
    return dist_mat,MST_km1,MST_orig,w_MST


#%%
def searchpathends(cluster_centers,MST_km):
    #first convert MST_km to a dict of indices
    MST_dict={}
    end_idxs=[]
    end_vals=[]
    for i in range(0,MST_km.shape[0]):
        MST_dict[i]=np.where(MST_km[i]>0.0)[0]
    for key in MST_dict.keys():
        if len(MST_dict[key])==1:
            end_vals.append(cluster_centers[MST_dict[key]])
            end_idxs.append(key)
    return end_vals,end_idxs,MST_dict

#%%
def searchlongestpath(end_idxs,MST_dict,parent_node_in_MST,num_points_per_cluster):
    #parent node is the starting node for the tree traversal
    #if parent node is not empty then start from parent node and do a tree traversal from that node
    M=len(MST_dict)
    all_paths={}
    all_points=np.array([])
    all_path_len=np.array([])
    if(len(parent_node_in_MST)!=0):
#         each node is a list of 3 numbers,  [node_index, distance from root, parent]
#         for each node initialize it with [i, np.inf, -1]
#               for each node n in Graph:
#                   n.distance = INFINITY
#                   n.parent = NIL
        i=parent_node_in_MST[0]
        nodes=np.zeros((M,3))
        for j in range(0,M):
            nodes[j,]=np.array([j,np.inf,-1])
        nodes[i,]=np.array([i,0,-1])
        Q=queue.Queue()
        Q.put(tuple(nodes[i,]))
        #now search till we reach one of end_idxs
        #        current = Q.dequeue()
        #          for each node n that is adjacent to current:
        #              if n.distance == INFINITY:
        #                  n.distance = current.distance + 1
        #                  n.parent = current
        #                  Q.enqueue(n)
        while(Q.empty()==False):        
            current=Q.get()
            neighbors=MST_dict[current[0]]
            for k in range(0,len(neighbors)):
                if(nodes[neighbors[k],1]==np.inf):
                    nodes[neighbors[k],1]=current[1]+1
                    nodes[neighbors[k],2]=current[0]
                    Q.put(tuple(nodes[neighbors[k],]))
        
#        from any node now we have a path till the root node.
#        path can be found from nodes matrix, by moving up the parents
#        first find the nodes except the current root in the terminal set of nodes 
        R=np.where(np.array(end_idxs)!=i)
        for j in range(0,len(R[0])):
            path=np.array([])
            points=0
            path_len=0
            node=end_idxs[R[0][j]]
            path_len=path_len+1
            points=points+num_points_per_cluster[node]
            while(node!=parent_node_in_MST[0]):
                path=np.append(path,node)
                node=int(nodes[node,2])
                points=points+num_points_per_cluster[node]
                path_len=path_len+1
            path=np.append(path,parent_node_in_MST)
            all_paths[j]= path
            all_points=np.append(all_points,points)
            all_path_len=np.append(all_path_len,path_len)
    else:
        for i in range(0,len(end_idxs)):
            nodes=np.zeros((M,3))
            for j in range(0,M):
                nodes[j,]=np.array([j,np.inf,-1])
            nodes[end_idxs[i],]=np.array([end_idxs[i],0,-1])
            Q=queue.Queue()
            Q.put(tuple(nodes[end_idxs[i],]))
          
            while(Q.empty()==False):        
                current=Q.get()
                neighbors=MST_dict[current[0]]
                for k in range(0,len(neighbors)):
                    if(nodes[neighbors[k],1]==np.inf):
                        nodes[neighbors[k],1]=current[1]+1
                        nodes[neighbors[k],2]=current[0]
                        Q.put(tuple(nodes[neighbors[k],]))
            
            R=np.where(np.array(end_idxs)!=end_idxs[i])
           
            for j in range(0,len(R[0])):
                path=np.array([])
                points=0
                path_len=0
                node=end_idxs[R[0][j]]
                path_len=path_len+1
                points=points+num_points_per_cluster[node]
                while(node!=end_idxs[i]):
                    path=np.append(path,node)
                    node=int(nodes[node,2])
                    points=points+num_points_per_cluster[node]
                    path_len=path_len+1
                path=np.append(path,parent_node_in_MST)
                all_paths[i*len(R)+j]   =path
                all_points=np.append(all_points,points)
                all_path_len=np.append(all_path_len,path_len)
    #find the path with max length
    val_path=np.max(np.array(all_path_len))
#    idx_path=np.argmax(np.array(all_path_len))
    R_path=np.where(np.array(all_path_len)==val_path)
#   if more than two paths have same lengths  then choose the one with max
#   number of cells
    if(len(R_path[0])>2):
        cells_in_path=all_points[R_path[0]]
        val_cell=np.max(np.array(cells_in_path))
#        idx_cell=np.argmax(np.array(cells_in_path))
        R_cell=np.where(np.array(cells_in_path)==val_cell)
        p=np.random.rand(len(R_cell[0]))
        p=np.divide(p,np.sum(p))
        prob_idx=np.argmax(p)
        longest_path=all_paths[R_path[0][R_cell[0][prob_idx]]]
    elif(len(R_path[0])>1 and len(R_path[0])<=2):
        p=np.random.rand()
        if(p>0.5):
            longest_path=all_paths[R_path[0][0]]
        else:
            longest_path=all_paths[R_path[0][1]]
    else:
        longest_path=all_paths[R_path[0][0]]
    return all_paths,longest_path
       
#%%
#cluster_centers=connected_nodes_to_node
#X_reduced=temp_nodes
def clusterassignment(cluster_centers,X_reduced):
    dist_mat1=pairwise.pairwise_distances(X_reduced, cluster_centers, metric='euclidean')
    cluster_assigned_to1=np.argmin(dist_mat1,axis=1)
    N=X_reduced.shape[0]
    M=cluster_centers.shape[0]
    r=np.zeros((N,M),dtype='int64')
    r[range(N),cluster_assigned_to1]=1
    #check for empty clusters
    r_sum=np.sum(r,axis=0)
    #find the index of the entries which have zero elements
    idx_zeros=np.where(r_sum==0)
    if(idx_zeros[0].size!=0):
        idx_max=np.argmax(r_sum)#only the forst occurance of max values is returned
        if(idx_max.size==1):
            idx_max=np.array([idx_max],dtype='int32')
        #find the nearest neighbors where  labels == idx_max[0]
        
        for idx in idx_zeros[0]:
            temp1=X_reduced[cluster_assigned_to1==idx_max[0],]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(temp1)
            distances, indices = nbrs.kneighbors(cluster_centers[[idx],])
            temp=cluster_assigned_to1[cluster_assigned_to1==idx_max[0]]
#            print idx
            temp[indices]=idx
            cluster_assigned_to1[cluster_assigned_to1==idx_max[0]]=temp
            r=np.zeros((N,M),dtype='int64')
            r[range(N),cluster_assigned_to1]=1
                
    return r,cluster_assigned_to1
#%%
#cluster_centers=cluster_centers_steiner
#MST_dict=MST_dict_steiner
#all_paths=all_paths_steiner
#longest_path=longest_path_steiner
#MST_orig=MST_orig_steiner
#cluster_labels=cluster_labels_steiner
def pseudotemporalordering(X_reduced,cluster_centers,MST_dict,all_paths,longest_path,MST_orig,cluster_labels):
#    the cells belonging to end nodes, are projected onto end clusters i.e,
#    cells in cluster C_1 are placed on C_1-C_2, cells in cluster C_M are placed in C_(M-1)-C_M.
#    If no bifurcation, then cells belonging to intermediate cluster C_i are divided into
#    two parts in a way such that cells closer to C_(i-1) are projected on
#    C_(i-1)-C_i and cells closer to C_(i+1) are projected onto C_i- C_(i+1)
#    if bifurcation occurs at C_i then cells are divided as per neighbors of C_i
#    M=cluster_centers.shape[0]
    N=X_reduced.shape[0]
    start_node=longest_path[-1]
    parent_node_paths={}
    k=0
    for i in range(0,len(all_paths)):
        if(all_paths[i][-1]==start_node):
            parent_node_paths[k]=all_paths[i]
            k=k+1
    
#    for each of the edges, find the projection of cells onto those
#    then according to the paths order them
    nodes_to_be_projected={}
    edge_visited=np.zeros((MST_orig.shape[0],1))
    end_pseudotime_edge=np.zeros((MST_orig.shape[0],1))
    for i in range(0,MST_orig.shape[0]):
        temp_node_idx_final=np.array([],dtype='int64')
        for j in range(0,2):
            node=MST_orig[i,j]
            degree_node=len(MST_dict[node])
            connected_node_idx_to_node=MST_dict[node]
            if(degree_node==1):
                temp_node_idx=np.where(cluster_labels==node)
                temp_node_idx_final=np.append(temp_node_idx_final,temp_node_idx[0])
            else:
                temp_node_idx1=np.where(cluster_labels==node)#this returns numpy array
                temp_nodes=X_reduced[temp_node_idx1[0],]#points belonging to the current node
                connected_nodes_to_node=cluster_centers[connected_node_idx_to_node,]#nodes connected to the current node
                # now divide the points beloinging to current node among all the nodes it is connected to by an edge
#                print (i,j)
                r,cluster_assigned_to_temp=clusterassignment(connected_nodes_to_node,temp_nodes)
                if(j==0):
                    node_connected_to_current_node=MST_orig[i,1]
                    R=np.where(connected_node_idx_to_node==node_connected_to_current_node)
                    temp_node_idx2=np.where(cluster_assigned_to_temp==np.array(R[0]))
                    temp_node_idx3=temp_node_idx1[0][temp_node_idx2[0],]
                    temp_node_idx_final=np.append(temp_node_idx_final,temp_node_idx3)
                else:
                    node_connected_to_current_node=MST_orig[i,0]
                    # find the index of the node_connected_to_current_node i.e., node connected in the current edge
                    # in all the nodes connected to current_node
                    R=np.where(connected_node_idx_to_node==node_connected_to_current_node)
                    #find which all points belong to the node which connected by an edge to the current node
                    temp_node_idx2=np.where(cluster_assigned_to_temp==np.array(R[0]))
                    #find the actual indices of those points among all points
                    temp_node_idx3=temp_node_idx1[0][temp_node_idx2[0],]
                    temp_node_idx_final=np.append(temp_node_idx_final,temp_node_idx3)
        nodes_to_be_projected[i]=temp_node_idx_final
        edge_visited[i]=False
        end_pseudotime_edge[i]=np.inf
    ordering=np.zeros(N,dtype='int64')
    init_time=0
    for i in range(0,len(parent_node_paths)):
        path=np.array(parent_node_paths[i],dtype='int64')
        path=path[::-1]#rerverse the path
        for j in range(0,len(path)-1):
            idx_edge=0
            idx_edge_prev=0
            #find the index of this edge in MST_orig
            for k in range(0,MST_orig.shape[0]):
                if((path[j]==MST_orig[k,0] and path[j+1]==MST_orig[k,1]) or (path[j]==MST_orig[k,1] and path[j+1]==MST_orig[k,0])):
                    idx_edge=k
            
            if(j>=2):
                for k in range(0,MST_orig.shape[0]):
                    if((path[j]==MST_orig[k,0] and path[j-1]==MST_orig[k,1]) or (path[j]==MST_orig[k,1] and path[j-1]==MST_orig[k,0])):
                        idx_edge_prev=k
                init_time=end_pseudotime_edge[idx_edge_prev]
            if(edge_visited[idx_edge]==False):
#                 if the nodes on that edge have not been assigned a pseudotime
#                 then we project the cells onto that node and give them a
#                 pseudotime
                cells=X_reduced[np.array(nodes_to_be_projected[idx_edge],dtype='int64'),]
                v=cluster_centers[path[j+1],]-cluster_centers[path[j],]
                v=np.divide(v,np.linalg.norm(v))
                projection=np.dot(cells,v.T)
                sort_idx=np.argsort(projection)
                P=cells.shape[0]
                node_idx=np.array(nodes_to_be_projected[idx_edge],dtype='int64')
                ordering[node_idx[sort_idx]]=init_time+np.array(range(1,P+1),dtype='int64')
                init_time=init_time+P
                edge_visited[idx_edge]=True
                end_pseudotime_edge[idx_edge]=init_time
    return ordering                    
#%%                        
def correlation(x,y,corr_type):
    if(corr_type=='pearson'):
        corr,p_val=sp.stats.pearsonr(x, y)
    elif(corr_type=='spearman'):
        corr,p_val=sp.stats.spearmanr(x, y)
    else:
        print('wrong inputs!')
    return corr,p_val
#%%
def steinercost(cluster_centers,X_reduced,C,MST_orig,r):
    dist_mat=pairwise.pairwise_distances(X_reduced, cluster_centers, metric='euclidean')
    dist_mat=dist_mat**2
    D1=r*dist_mat
    sum1=np.sum(D1)
    term1=0.0
    for i in range(0,MST_orig.shape[0]):
        term1=term1+np.linalg.norm(cluster_centers[MST_orig[i,0],]-cluster_centers[MST_orig[i,1],])**2
    cost=term1+C*sum1
    return cost
#%%
def calculategradients(x,X_reduced,MST_orig,r,C):
    M=x.shape[0]
    d=x.shape[1]
    gradx=np.zeros((M,d))
    for i in range(0,M):
        for j in range(0,2):
            temp=np.where(MST_orig[:,j]==i)#find all the indices in MST_orig, where the jth index is i
            if(temp[0].size!=0):
                if(j==0):
                    for k in range(0,len(temp)):
                        idx1=MST_orig[temp[0][k],0]
                        idx2=MST_orig[temp[0][k],1]
                        gradx[i,:]=gradx[i,:]+2*(x[idx1,:]-x[idx2,:])
                else:
                    for k in range(0,len(temp)):
                        idx1=MST_orig[temp[0][k],0]
                        idx2=MST_orig[temp[0][k],1]
                        gradx[i,:]=gradx[i,:]-2*(x[idx1,:]-x[idx2,:])
        temp=np.subtract(X_reduced,x[i,:])
        temp1=np.multiply(temp.T,r[:,i].T).T
        sum1=np.sum(temp1,axis=0)
        sum1=2.0*C*sum1
        gradx[i,:]=np.subtract(gradx[i,:],sum1)                        
    return gradx
                    
                    
#%%     
def steiner_map(x,X_reduced,C,eta,iterMax):
    eta_zero=eta
    dist_mat,MST_km,MST_orig,w_MST=calculateMST(x)
    r,cluster_assigned_to=clusterassignment(x,X_reduced)            
    iters=1
    w_MST_prev=1.0
    cost_best=steinercost(x,X_reduced,C,MST_orig,r)
    all_costs=cost_best
    x_best=x
    w_MST_best=w_MST
    MST_best=MST_orig
    r_best=r
    cluster_assigned_to_best=cluster_assigned_to
    iter_best=iters
    
    all_costs_orig=all_costs
    cost_best_orig=cost_best
    cost_orig=cost_best
    x_orig=x
    w_MST_orig=w_MST
    MST_orig_orig=MST_orig
    r_orig=r
    cluster_assigned_to_orig=cluster_assigned_to
    iter_orig=iters
    eta_zero_orig=eta_zero
    while(iters<iterMax and (np.abs(np.divide(w_MST-w_MST_prev,w_MST_prev))>1e-05)):
        iters=iters+1
        w_MST_prev=w_MST
        gradx=calculategradients(x,X_reduced,MST_orig,r,C)
        x=np.subtract(x,eta*gradx)
        dist_mat,MST_km,MST_orig,w_MST=calculateMST(x)
        eta=np.divide(eta_zero,iters)
        r,cluster_assigned_to=clusterassignment(x,X_reduced)
        cost=steinercost(x,X_reduced,C,MST_orig,r)
#        print cost
        if(cost<cost_best):
            cost_best=cost
            x_best=x
            w_MST_best=w_MST
            MST_best=MST_orig
            r_best=r
            cluster_assigned_to_best=cluster_assigned_to
            iter_best=iters
        elif(cost>cost_best and iters<5):
            print('here')
            cost_best=cost_best_orig
            cost=cost_orig
            x=x_orig
            w_MST=w_MST_orig
            MST_orig=MST_orig_orig
            r=r_orig
            cluster_assigned_to=cluster_assigned_to_orig
            iters=iter_orig
            eta_zero=eta_zero/10
            all_costs=all_costs_orig
            w_MST_prev=1.0
        all_costs=np.append(all_costs,cost)
        
    return x_best,w_MST_best,MST_best,r_best,cluster_assigned_to_best,cost_best,iter_best,all_costs

        
        
        
        
        