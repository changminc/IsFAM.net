# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import functions as fn
import re
import networkx as nx
import json

# model_1: OUR_Train2, model_2:CHEN_Train6
model_path = 'model_1'
train_path = 'train/' + model_path + '/'
label_path = train_path + 'label/'
train_subg = 'Train_Cutoff4_CORKMAX_n2.subg'
# Read Train set's Subgraph
searched_TR, cnt_TR = fn.get_subg( train_path + train_subg )

# Label order: 1 = nonfamily, 2 = family
NonFam_Img_cnt = fn.get_num_jpg( label_path + 'PersonData_' + 'NonFam' + '.txt')
Fam_Img_cnt = fn.get_num_jpg( label_path + 'PersonData_' + 'Fam' + '.txt')
labels_cnt = NonFam_Img_cnt + Fam_Img_cnt
NonFam_cnt_TR = int( NonFam_Img_cnt * 0.8 )
Fam_cnt_TR = int( Fam_Img_cnt * 0.8 )
Img_cnt_TR = NonFam_cnt_TR + Fam_cnt_TR    # number of train images

   
"""
Build matrix of frequent subgraphs from train set
Row: Image-ID, Column: Subgraph-ID, Value: Frequency
"""
G = nx.Graph()
with open( train_path + train_subg, 'r' ) as m:
    f_list = m.readlines()   
    matrix_TR = np.zeros( (Img_cnt_TR, cnt_TR) )
    # 초기설정: (Image 개수, Subgraph 개수) 형태로 0을 채움    
    # row index ===> Img_list index,  col index ===> subGraph index
    for line in f_list[:]:       
        line_list = line.rstrip().split(' ')    # Read a line
        if line_list[0] == 't':
            label_list = []    
            subG_idx = line_list[2]
            subG_freQ = line_list[4]    
            # subG_freQ !== Documnet Frequency. subG_freQ == sum of Term Frequency
        elif line_list[0] == 'v':
            subG_vertex = line_list[1]
            subG_vertexLabel = line_list[2]
            label_list.append( int(subG_vertexLabel) )
            G.add_node( int(subG_vertex), label=int(subG_vertexLabel) )
        elif line_list[0] == 'e':
            subG_edge1 = line_list[1]
            subG_edge2 = line_list[2]
            subG_edgeLabel = line_list[3]
            G.add_edge( int(subG_edge1), int(subG_edge2), weight=int(subG_edgeLabel) )
        elif line.startswith(' {'):
            regx_ = re.compile('\d*\:\d*')    
            Gid_Freq = regx_.findall( line )
            for g_f in Gid_Freq:
                img_idx, Freq = g_f.split(':')
                matrix_TR[ int( img_idx ), int(subG_idx) ] += int( Freq )
            
            with open( train_path + 'graph/' + 'ID_' + subG_idx.zfill(3) + '.nodelabel', 'w' ) as nl:
                json.dump( label_list, nl )
            nx.write_gpickle( G, train_path + 'graph/' + 'ID_' + subG_idx.zfill(3) + '.g' )
            G.clear()
           
                    
# TF-IDF normalization of SubGraph-Frequency
col_sums_TR = matrix_TR.sum( axis=1 )
col_df_TR = ( matrix_TR !=0 ).sum(0)
col_log_tf_TR = np.log10( matrix_TR + 1 )
col_idf_TR = np.log10( (1.0*Img_cnt_TR) / (1.0*col_df_TR + 1) )
matrix_TR = col_log_tf_TR * col_idf_TR
matrix_TR[ np.isinf( matrix_TR ) ] = 0   # when the sum of feature vector == 0, NaN type is returned

                   
"""
Create Train-Model using SVM-Linear
"""
from svmutil import *
# Create Labels
# label Order: nonfamily first!, family followed
labels_NonFam_TR = np.full( NonFam_cnt_TR, 0.0, dtype=float )
labels_Fam_TR = np.full( Fam_cnt_TR, 1.0, dtype=float)
train_labels = np.concatenate( (labels_NonFam_TR, labels_Fam_TR), axis=0 )
#### convert to lists for libsvm
train_set = map( list, matrix_TR ) 
train_labels = list( train_labels )
# create SVM
prob = svm_problem( train_labels, train_set )
param = svm_parameter('-t 0')
# train SVM on data
model = svm_train( prob, param )

with open( train_path + 'train_values.pkl', 'w') as t:
    cPickle.dump( col_idf_TR, t)
    cPickle.dump( searched_TR, t)
    cPickle.dump( cnt_TR, t)
    
svm_save_model( train_path + 'train.model', model )


    