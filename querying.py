# -*- coding: utf-8 -*-
import numpy as np
import functions as fn
from svmutil import *
import cPickle
import projection as pr
# model_1: OUR_TrainSet, model_2:CHEN_TrainSet
model_path = 'model_1'
train_path = 'train/' + model_path + '/' 
train_g_path = train_path + 'graph/'
query_path = 'query/'

# Load Train Model and variables
model = svm_load_model( train_path + 'train.model' )
with open( train_path + 'train_values.pkl', 'r' ) as t:
    col_idf_TR = cPickle.load( t )
    search_TR = cPickle.load( t )
    cnt_TR = cPickle.load( t )

# Build the Martix of Query Subgraphs
face_attr = pr.faceAPI( query_path )
G_q = pr.build_facegraph( face_attr )
freq_lst = pr.get_frequency( G_q, train_g_path )
matrix_TE = np.zeros( (1, cnt_TR ) )
for BoFG_ID, freq in freq_lst:
    matrix_TE[ 0 ][ BoFG_ID ] = freq

"""
# Comparing subgraphs from train set  and  subgraphs from test set
"""
# TF-IDF normalization of SubGraph-Frequency advised by JaePil
col_sums_TE = matrix_TE.sum( axis=1 )
col_df_TE = (matrix_TE !=0).sum(0)
col_log_tf_TE = np.log10( matrix_TE + 1 )
matrix_TE = col_log_tf_TE * col_idf_TR
matrix_TE[ np.isinf( matrix_TE ) ] = 0   
matrix_TE[ np.isnan( matrix_TE ) ] = 0
            
   
"""
Predicting query's label using SVM-Linear
"""
#### convert to lists for libsvm
query = map( list, matrix_TE )
# Accuracy, Mean Squared Error, Squared Correlation Coefficient
pred_labels, (acc, mse, scc), pred_values = svm_predict( [0], query, model )
if pred_labels == [0.0]:
    print 'non-famly'
else:
    print 'family'