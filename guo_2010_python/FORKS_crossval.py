# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 19:25:10 2016

@author: mayank
"""

#%%
import os
print (os.getcwd())
path1="D:\Dropbox\phd\pseudotemporal\guo_2010_python"
os.chdir(path1)
print (os.getcwd())
#%%
from forks_fcns import *
#%%
dataframe=pd.read_csv("data/guo2010Data.txt",sep="\t")
#data in form samples X features
data=dataframe.iloc[1:,1:].as_matrix().T
data,gene_list,cell_list=remove_zero_rc(data) 
#gene_markers=pd.read_csv('data/df_gene_markers_norm_data.csv',header=None)

#%%
#cell_names=pd.Series(dataframe.columns.values[1:])
cell_names=pd.Series(dataframe.columns.values[1:])
actual_time=np.array(dataframe.iloc[0,1:],dtype="int64")

#%%
gene_name=dataframe.iloc[1:,0]
#%%
actual_time=actual_time[cell_list]
gene_name=gene_name[gene_list]
#%%
idx_genes,idx_cells=find_unique_gene_cells(gene_name.copy(),cell_names.copy())
#%%
gene_name=gene_name.iloc[idx_genes]
actual_time=actual_time[idx_cells]
cell_names=cell_names.iloc[idx_cells]
data=data[idx_cells,]
data=data[:,idx_genes]
#%%
#gene selection step
#frac_of_cells=0.1
#is_std=True
#std_frac=0.5
#fig_title='histogram of expressed genes'
#gene_list=gene_selection(data,fig_title,frac_of_cells,is_std,std_frac)
#gene_name=gene_name[gene_list]
#data=data[:,gene_list]
#%%
#get mean and variance of data across all cells before preprocessing
mean=np.mean(data,axis=0)
median=np.median(data,axis=0)
std=np.std(data,axis=0)

#%%
# the histogram of the mean of data
fig_title='histogram of mean gene expression before preprocessing LT1_0_LT2_0'
xlabel='mean'
ylabel='counts'
hist_plot(mean,fig_title,xlabel,ylabel)

# the histogram of the median of data
fig_title='histogram of median gene expression before preprocessing LT1_0_LT2_0'
xlabel='median'
ylabel='counts'
hist_plot(median,fig_title,xlabel,ylabel)

# the histogram of the standard deviation of data
fig_title='histogram of std gene expression before preprocessing LT1_0_LT2_0'
xlabel='std_dev'
ylabel='counts'
hist_plot(std,fig_title,xlabel,ylabel)

#%%
fig_title1='correlation of detected genes before preproc with PCs LT1_0_LT2_0'
fig_title2='PCA 2D plot before preproc with PCs LT1_0_LT2_0'
detected_genes=np.sum(data>0.0,axis=1)
actual_time1=actual_time[:]
cell_names1=cell_names.copy()
corr_before=get_corr_with_det_genes(data,detected_genes,actual_time1, fig_title1,fig_title2)
#%% 
#pQ normalization
use_mean=False
use_median=False
use_quantile=False
quantile=75
data1,cell_list,gene_list=normalization_pQ(data[:],use_mean,use_median,use_quantile,quantile)
detected_genes=detected_genes[cell_list]

actual_time1=actual_time[:]
actual_time1=actual_time1[cell_list]
cell_names1=cell_names.copy()
cell_names1=cell_names1.iloc[cell_list,]
#pass this way to prevent data mutation inside function: like pass by value
#%%
fig_title1='correlation of detected genes after preproc with PCs LT1_0_LT2_0'
fig_title2='PCA 2D plot after preproc with PCs LT1_0_LT2_0'
corr_after=get_corr_with_det_genes(data1,detected_genes,actual_time1, fig_title1,fig_title2)
#%%
#get mean and variance of data across all cells after preprocessing
mean1=np.mean(data1,axis=0)
median1=np.median(data1,axis=0)
std1=np.std(data1,axis=0)

#%%
# the histogram of the mean of data
fig_title='histogram of mean gene expression after preprocessing LT1_0_LT2_0'
xlabel='mean'
ylabel='counts'
hist_plot(mean1,fig_title,xlabel,ylabel)

# the histogram of the median of data
fig_title='histogram of median gene expression after preprocessing LT1_0_LT2_0'
xlabel='median'
ylabel='counts'
hist_plot(median1,fig_title,xlabel,ylabel)

# the histogram of the standard deviation of data
fig_title='histogram of std gene expression after preprocessing LT1_0_LT2_0'
xlabel='std_dev'
ylabel='counts'
hist_plot(std1,fig_title,xlabel,ylabel)
#%%
#spherize the data
data1=normalization_spherize(data1)
#%%save the processed_data 

np.savetxt(path1+"\\data\\processed_data.csv",data1,delimiter=',')
np.savetxt(path1+"\\data\\processed_time.csv",actual_time1,delimiter=',')
gene_name.to_csv(path1+"\\data\\processed_genes.csv",header=None)
cell_names1.to_csv(path1+"\\data\\processed_cells.csv",header=None)
#%% find the number of components so as to explain the variance in the data
#pca = PCA(whiten=True)
pca = PCA()
pca.fit(data1)
mappedData=pca.fit_transform(data1)
explained_var_ratio=pca.explained_variance_ratio_
cum_sum_exp_var=np.cumsum(explained_var_ratio)
inds = np.where(cum_sum_exp_var > 0.9)
red_dim=inds[0][0]
mappedData=mappedData[0:,0:red_dim]

plt.plot(cum_sum_exp_var)
plt.xlabel('PCA components')
plt.ylabel('Cumulative variance')
fig_title='Cumulative variance of PCs'
plt.title(fig_title)
plt.grid(True)
plt.show()
plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
plt.close()
if(red_dim>=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap='rainbow'    
    labels=pd.Series(actual_time1)
    legend_line2d,unique_labels,palette,n_colors,scatter_colors=add_legend(colormap,labels)
    scplot=ax.scatter(mappedData[:,0], mappedData[:,1], mappedData[:,2], c=list(scatter_colors),s=20,alpha=1.0,edgecolors='face')
    legend1 = ax.legend(legend_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=False,fancybox=False,frameon=True)
    fig_title='3D PCA plot of data'
    ax.set_title(fig_title)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')
else:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormap='rainbow'
    labels=pd.Series(actual_time1)
    legend_line2d,unique_labels,palette,n_colors,scatter_colors=add_legend(colormap,labels)
    scplot=ax.scatter(mappedData[:,0], mappedData[:,1], c=list(scatter_colors),s=20,alpha=1.0,edgecolors='face')
    legend1 = ax.legend(legend_line2d,unique_labels,numpoints=1, fontsize=10,loc='upper right',shadow=False,fancybox=False,frameon=True)
    fig_title='2D PCA plot of data'
    ax.set_title(fig_title)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y_axis')
    
plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
plt.close()


#%%find number of clusters
range_clusters=range(4,31,1)
n_clus,max_sil,sil_scores=find_nclusters(mappedData,range_clusters)
M=range_clusters[n_clus]

#%%
#generate cross val indices
nfolds=8
skf = StratifiedKFold(n_splits=nfolds,random_state=1)
cv_idx=np.zeros(actual_time1.shape)
i=0
for train_ind, test_idx in skf.split(mappedData, actual_time1):
    cv_idx[test_idx]=i
    i=i+1
np.savetxt(os.getcwd()+'/data'+"/cv_idx.csv", cv_idx, delimiter=",")
for i in range(0,nfolds): print (np.sum(cv_idx!=i))
#%%
#all_mapping_type=['Isomap','MDS','PCA','RandomForest','SpectralEmbedding','LLE_standard','tSNE']
all_mapping_type=['Isomap','MDS','PCA','RandomForest','SpectralEmbedding','tSNE']
#all_mapping_type=['tSNE']
init_cell=list([])

for mapping_type in all_mapping_type:
    result_kmea=np.zeros((1,3))
    result_kmed=np.zeros((1,3))
    result_steiner=np.zeros((1,3))
#    if(mapping_type=='tSNE'):
#        red_dim=8
#    if(red_dim<10 and mapping_type=='LLE_modified'):
#        n_neighbors=red_dim
    if(mapping_type=='LLE_modified'):
        n_neighbors=red_dim        
    else:
        n_neighbors=10
    for j in range(0,nfolds):
        data2=data1[cv_idx!=j,:]
        actual_time2=actual_time1[cv_idx!=j]
        
        #############################################################################
        #reduce the dimensionality of the data
        plot_dimensions=2
#        n_neighbors=3
        n_components=red_dim
        isplot=False
        gene_preprocessing_type='none'
        mapping_params={}
        mapping_params['mapping_type']=mapping_type
        mapping_params['n_neighbors']=n_neighbors
        mapping_params['n_components']=n_components
        mapping_params['isplot']=isplot
        mapping_params['gene_preprocessing_type']=gene_preprocessing_type
        if(mapping_type=='tSNE'):
            #we first reduce the initial dimension to 50 then use tSNE on the reduced dataset            
            if(data2.shape[1]>50 and red_dim<50):
                pca = PCA()
                pca.fit(data2)
                mappedData=pca.fit_transform(data2)
                mappedData=mappedData[0:,0:50]            
                X_reduced=mapping(mappedData,actual_time2,mapping_params)
            elif(data2.shape[1]>2*red_dim and red_dim>50):                
                pca = PCA()
                pca.fit(data2)
                mappedData=pca.fit_transform(data2)
                mappedData=mappedData[0:,0:2*red_dim]            
                X_reduced=mapping(mappedData,actual_time2,mapping_params)
            else:
                X_reduced=mapping(data2,actual_time2,mapping_params)
        else:
            X_reduced=mapping(data2,actual_time2,mapping_params)
        print ('2 %d,%d'%X_reduced.shape)
        ##############################################################################
        #first run k-means on the data
        
        initialization=0
        print('mapping-> %s, nfold->%d, type->%s, init->%d'%(mapping_type,j,'kmeans',initialization))
        
        t0=time()
        kmeans = KMeans(n_clusters=M, random_state=0).fit(X_reduced)
        cluster_centers_kmea=kmeans.cluster_centers_
#        r_kmea,cluster_labels_kmea=clusterassignment(cluster_centers_kmea,X_reduced)
        cluster_labels_kmea=kmeans.labels_
        
        num_points_per_cluster_kmea=np.zeros((M,1))
        for i in range(0,M):
            num_points_per_cluster_kmea[i]=np.sum(cluster_labels_kmea==i)
            
        plot_params={}
        plot_params['mapping_type']=mapping_type
        if(actual_time2.size!=0):
            plot_params['label_type']='actual_time'
        else:
            plot_params['label_type']=''
        plot_params['clustering_type']='kmeans'
        plot_params['init_method']=initialization #0:default (for kmeans and kmedoids), 1:kmeans, 2:kmedoids, 3:knn, 4:multiple_run
        plot_params['plot_dimensions']=plot_dimensions #2 or 3 only
        plot_params['isplot']=isplot
        plot_params['gene_preprocessing_type']=gene_preprocessing_type
        plotting_data(cluster_centers_kmea,X_reduced,actual_time2,plot_params)
        
        #find the MST connecting the k-means cluster centers        
        dist_mat_kmea,MST_kmea,MST_orig_kmea,w_MST_kmea=calculateMST(cluster_centers_kmea)
        #search for the path ends, this step is mainly used when we do not know the starting point
        end_vals_kmea,end_idxs_kmea,MST_dict_kmea=searchpathends(cluster_centers_kmea,MST_kmea)
        
        #find the index in cluster_centers closest to the starting cell
        starting_cell=np.zeros((0,))
        #if you have starting cell, put labels and data as empty matrices else put starting cell as empty matrix
        indices_kmea=find_closest_cluster_center(cluster_centers_kmea,X_reduced,actual_time2,starting_cell)
        #search all the paths going from the cluster center nearest to starting cell
        all_paths_kmea,longest_path_kmea=searchlongestpath(end_idxs_kmea,MST_dict_kmea,indices_kmea[0],num_points_per_cluster_kmea) 
        #perform pseudotemporal ordering from that starting cell
        ordering_kmea=pseudotemporalordering(X_reduced,cluster_centers_kmea,MST_dict_kmea,all_paths_kmea,longest_path_kmea,MST_orig_kmea,cluster_labels_kmea)
        print ('3 %d,%d'%X_reduced.shape)
        #find spearman correlation with actual labels
        spearman_corr_kmea,p_val_kmea=correlation(actual_time2,ordering_kmea,'spearman')
        t1=time()
        result_kmea=np.append(result_kmea,np.array([[spearman_corr_kmea,t1-t0,initialization]]),axis=0)
        
        #######################################################################
        # then run k-medoids        
        
        initialization=0
        print('mapping-> %s, nfold->%d, type->%s, init->%d'%(mapping_type,j,'kmedoids',initialization))
        t0=time()
        kmedoids = KMedoids(n_clusters=M, random_state=0).fit(X_reduced)
        cluster_centers_kmed=kmedoids.cluster_centers_
#        r_kmed,cluster_labels_kmed=clusterassignment(cluster_centers_kmed,X_reduced)
        cluster_labels_kmed=kmedoids.labels_
        
        num_points_per_cluster_kmed=np.zeros((M,1))
        for i in range(0,M):
            num_points_per_cluster_kmed[i]=np.sum(cluster_labels_kmed==i)
            
        plot_params={}
        plot_params['mapping_type']=mapping_type
        if(actual_time2.size!=0):
            plot_params['label_type']='actual_time'
        else:
            plot_params['label_type']=''
        plot_params['clustering_type']='kmedoids'
        plot_params['init_method']=initialization #0:default, 1:kmeans, 2:kmedoids, 3:knn, 4:multiple_run
        plot_params['plot_dimensions']=plot_dimensions #2 or 3 only
        plot_params['isplot']=isplot
        plot_params['gene_preprocessing_type']=gene_preprocessing_type
        plotting_data(cluster_centers_kmed,X_reduced,actual_time2,plot_params)
        
        #find the MST connecting the k-means cluster centers        
        dist_mat_kmed,MST_kmed,MST_orig_kmed,w_MST_kmed=calculateMST(cluster_centers_kmed)
        #search for the path ends, this step is mainly used when we do not know the starting point
        end_vals_kmed,end_idxs_kmed,MST_dict_kmed=searchpathends(cluster_centers_kmed,MST_kmed)
        
        #find the index in cluster_centers closest to the starting cell
        starting_cell=np.zeros((0,))
        #if you have starting cell, put labels and data as empty matrices else put starting cell as empty matrix
        indices_kmed=find_closest_cluster_center(cluster_centers_kmed,X_reduced,actual_time2,starting_cell)
        #search all the paths going from the cluster center nearest to starting cell
        all_paths_kmed,longest_path_kmed=searchlongestpath(end_idxs_kmed,MST_dict_kmed,indices_kmed[0],num_points_per_cluster_kmed) 
        #perform pseudotemporal ordering from that starting cell
        ordering_kmed=pseudotemporalordering(X_reduced,cluster_centers_kmed,MST_dict_kmed,all_paths_kmed,longest_path_kmed,MST_orig_kmed,cluster_labels_kmed)
        print ('4 %d,%d'%X_reduced.shape)
        #find spearman correlation with actual labels
        spearman_corr_kmed,p_val_kmed=correlation(actual_time2,ordering_kmed,'spearman')
        t1=time()
        result_kmed=np.append(result_kmed,np.array([[spearman_corr_kmed,t1-t0,initialization]]),axis=0)
        ###########################################################################
        #Run steiner tree algo
       
        for initialization in range(1,3):
            if(initialization==1):
                # run steiner with k-means initialization
                print('mapping-> %s, nfold->%d, type->%s, init->%d'%(mapping_type,j,'steiner',initialization))
                t0=time()                
                kmeans = KMeans(n_clusters=M, random_state=0).fit(X_reduced)
                cluster_centers_init_st=kmeans.cluster_centers_
                cluster_labels_init_st=kmeans.labels_
            elif(initialization==2):
                # run steiner with k-medoids initialization
                print('mapping-> %s, nfold->%d, type->%s, init->%d'%(mapping_type,j,'steiner',initialization))
                t0=time()
                kmedoids = KMedoids(n_clusters=M, random_state=0).fit(X_reduced)
                cluster_centers_init_st=kmedoids.cluster_centers_
                cluster_labels_init_st=kmedoids.labels_
            else:
                #select M random points from the data
                print('mapping-> %s, nfold->%d, type->%s, init->%d'%(mapping_type,j,'steiner',initialization))
                t0=time()
                np.random.seed(1)
                indices=np.random.choice(actual_time2.shape[0], M, replace=False)
                cluster_centers_init_st=X_reduced[indices,:]
            print ('5 %d,%d'%X_reduced.shape)
            iterMax=1000
            eta=1e-02
            C=1.0
            cluster_centers_steiner,w_MST_steiner,MST_steiner,r_steiner,cluster_labels_steiner,cost_steiner,iter_steiner,all_costs_steiner= steiner_map(cluster_centers_init_st,X_reduced,C,eta,iterMax) 
            plt.plot(all_costs_steiner)
            plt.xlabel('iterations')
            plt.ylabel('steiner_cost')
            fig_title='steiner cost init %d nfold %d'%(initialization,j)
            plt.title(fig_title)
            plt.grid(True)
            plt.show()
            plt.savefig(os.getcwd()+'/graphs'+'/'+fig_title+'.pdf', bbox_inches='tight')
            plt.close()
            print ('6 %d,%d'%data1.shape)
            num_points_per_cluster_steiner=np.zeros((M,1))
            for i in range(0,M):
                num_points_per_cluster_steiner[i]=np.sum(cluster_labels_steiner==i)
                
            plot_params={}
            plot_params['mapping_type']=mapping_type
            if(actual_time2.size!=0):
                plot_params['label_type']='actual_time'
            else:
                plot_params['label_type']=''
            plot_params['clustering_type']='steiner'
            plot_params['init_method']=initialization #0:default, 1:kmeans, 2:kmedoids, 3:knn, 4:multiple_run
            plot_params['plot_dimensions']=plot_dimensions #2 or 3 only
            plot_params['isplot']=isplot
            plot_params['gene_preprocessing_type']=gene_preprocessing_type
            plotting_data(cluster_centers_steiner,X_reduced,actual_time2,plot_params)
            
            #find the MST connecting the steiner cluster centers        
            dist_mat_steiner,MST_steiner,MST_orig_steiner,w_MST_steiner=calculateMST(cluster_centers_steiner)
            #search for the path ends, this step is mainly used when we do not know the starting point
            end_vals_steiner,end_idxs_steiner,MST_dict_steiner=searchpathends(cluster_centers_steiner,MST_steiner)
            
            #find the index in cluster_centers closest to the starting cell
            starting_cell=np.zeros((0,))
            #if you have starting cell, put labels and data as empty matrices else put starting cell as empty matrix
            indices_steiner=find_closest_cluster_center(cluster_centers_steiner,X_reduced,actual_time2,starting_cell)
            #search all the paths going from the cluster center nearest to starting cell
            all_paths_steiner,longest_path_steiner=searchlongestpath(end_idxs_steiner,MST_dict_steiner,indices_steiner[0],num_points_per_cluster_steiner) 
            #perform pseudotemporal ordering from that starting cell
            ordering_steiner=pseudotemporalordering(X_reduced,cluster_centers_steiner,MST_dict_steiner,all_paths_steiner,longest_path_steiner,MST_orig_steiner,cluster_labels_steiner)
            #find spearman correlation with actual labels
            spearman_corr_steiner,p_val_steiner=correlation(actual_time2,ordering_steiner,'spearman')
            
            t1=time()
            result_steiner=np.append(result_steiner,np.array([[spearman_corr_steiner,t1-t0,initialization]]),axis=0)
    #once all the results of folds are out, we calulate the statistics of the results and the we then calculate the best among all three
    result_kmea=result_kmea[1:,]
    result_kmed=result_kmed[1:,]
    result_steiner=result_steiner[1:,]

    result_kmea_pd   = pd.DataFrame(result_kmea,index=range(0,result_kmea.shape[0]),columns=['Correlation', 'Run time','initialization'])
    result_kmed_pd   = pd.DataFrame(result_kmed,index=range(0,result_kmed.shape[0]),columns=['Correlation', 'Run time','initialization'])
    result_steiner_pd= pd.DataFrame(result_steiner,index=range(0,result_steiner.shape[0]),columns=['Correlation', 'Run time','initialization'])
    
    file_name_kmeans='result_'+mapping_type+'_kmeans.csv'
    file_name_kmedoids='result_'+mapping_type+'_kmedoids.csv'
    file_name_steiner='result_'+mapping_type+'_steiner.csv'
    
    result_kmea_pd.to_csv(path1+"\\results\\"+file_name_kmeans)
    result_kmed_pd.to_csv(path1+"\\results\\"+file_name_kmedoids)
    result_steiner_pd.to_csv(path1+"\\results\\"+file_name_steiner)
    
    result_kmea_stat=[[np.mean(result_kmea[:,0],axis=0),np.std(result_kmea[:,0],axis=0),np.mean(result_kmea[:,1],axis=0),np.std(result_kmea[:,0],axis=0),'kmeans','default',mapping_type,int(red_dim),M,n_neighbors,nfolds]]
    result_kmed_stat=[[np.mean(result_kmed[:,0],axis=0),np.std(result_kmed[:,0],axis=0),np.mean(result_kmed[:,1],axis=0),np.std(result_kmed[:,0],axis=0),'kmedoids','default',mapping_type,red_dim,M,n_neighbors,nfolds]]
#    result_steiner_stat=[[np.mean(result_steiner[result_steiner[:,2]==1,0],axis=0),np.std(result_steiner[result_steiner[:,2]==1,0],axis=0),np.mean(result_steiner[result_steiner[:,2]==1,1],axis=0),np.std(result_steiner[result_steiner[:,2]==1,1],axis=0),'steiner','kmeans',mapping_type,red_dim,M,nfolds],
#                         [np.mean(result_steiner[result_steiner[:,2]==2,0],axis=0),np.std(result_steiner[result_steiner[:,2]==2,0],axis=0),np.mean(result_steiner[result_steiner[:,2]==2,1],axis=0),np.std(result_steiner[result_steiner[:,2]==2,1],axis=0),'steiner','kmedoids',mapping_type,red_dim,M,nfolds],
#                         [np.mean(result_steiner[result_steiner[:,2]==3,0],axis=0),np.std(result_steiner[result_steiner[:,2]==3,0],axis=0),np.mean(result_steiner[result_steiner[:,2]==3,1],axis=0),np.std(result_steiner[result_steiner[:,2]==3,1],axis=0),'steiner','random',mapping_type,red_dim,M,nfolds] ]
    result_steiner_stat=[[np.mean(result_steiner[result_steiner[:,2]==1,0],axis=0),np.std(result_steiner[result_steiner[:,2]==1,0],axis=0),np.mean(result_steiner[result_steiner[:,2]==1,1],axis=0),np.std(result_steiner[result_steiner[:,2]==1,1],axis=0),'steiner','kmeans',mapping_type,red_dim,M,n_neighbors,nfolds],
                         [np.mean(result_steiner[result_steiner[:,2]==2,0],axis=0),np.std(result_steiner[result_steiner[:,2]==2,0],axis=0),np.mean(result_steiner[result_steiner[:,2]==2,1],axis=0),np.std(result_steiner[result_steiner[:,2]==2,1],axis=0),'steiner','kmedoids',mapping_type,red_dim,M,n_neighbors,nfolds]]
                         
    result_kmea_stat_pd   = pd.DataFrame(result_kmea_stat,index=range(0,len(result_kmea_stat)),columns=['mean Correlation', 'std Correlation','mean Run time','std Run time','method','initialization','mapping_type','reduced dim','# clusters','n_neigh','nfolds'])
    result_kmed_stat_pd   = pd.DataFrame(result_kmed_stat,index=range(0,len(result_kmed_stat)),columns=['mean Correlation', 'std Correlation','mean Run time','std Run time','method','initialization','mapping_type','reduced dim','# clusters','n_neigh','nfolds'])
    result_steiner_stat_pd= pd.DataFrame(result_steiner_stat,index=range(0,len(result_steiner_stat)),columns=['mean Correlation', 'std Correlation','mean Run time','std Run time','method','initialization','mapping_type','reduced dim','# clusters','n_neigh','nfolds'])
    
    file_name_kmeans_stat='result_'+mapping_type+'_kmeans_stat.csv'
    file_name_kmedoids_stat='result_'+mapping_type+'_kmedoids_stat.csv'
    file_name_steiner_stat='result_'+mapping_type+'_steiner_stat.csv'
    
    result_kmea_stat_pd.to_csv(path1+"\\results\\"+file_name_kmeans_stat)
    result_kmed_stat_pd.to_csv(path1+"\\results\\"+file_name_kmedoids_stat)
    result_steiner_stat_pd.to_csv(path1+"\\results\\"+file_name_steiner_stat)
                       
    #%%

   
    