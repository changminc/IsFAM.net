# -*- coding: utf-8 -*-
import urllib2
import os, time
import json
import networkx as nx
import functions as fn
import itertools
from networkx.algorithms import isomorphism as iso

def faceAPI( query_path ):
    """ # Send Query image to Oxford AI's API
    # https://www.projectoxford.ai/demo/face#detection
    """
    img_files = [ f for f in os.listdir( query_path ) if f.endswith('.jpg') ]
    url = 'https://api.projectoxford.ai/face/v0/detections?analyzesFaceLandmarks=true&analyzesAge=true&analyzesGender=true&analyzesHeadPose=true'
    
    for img in img_files:
        data = open( query_path + img, "rb")
        length = os.path.getsize( query_path + img )
        request = urllib2.Request(url=url, data=data)
        request.add_header('Cache-Control', 'no-cache')
        request.add_header('Content-Length', '%d' % length)
        request.add_header('Content-Type', 'application/octet-stream')
        request.add_header('Ocp-Apim-Subscription-Key', 'FILL YOUR SUBSCRIPTION KEY of OXFORD API')
        
        result = urllib2.urlopen(request).read().strip()
        res_obj = json.loads( result )
        data.close()
        
        with open( query_path + 'attr/' + img[:-4] + '.json', 'w' ) as f:
            json.dump( res_obj, f, indent=4 )
    
    return res_obj
    
def build_facegraph( json_obj ):
    """
    # Build Face Graph from query image 
    """
    # create Graph of Query & BoFG
    G_q = nx.Graph(); G_bofg = nx.Graph()
    # from Query
    faces = json_obj
    age_range = [ (1,3), (3,8), (8,13), (13,20), (20,37), (37,66), (66,85) ]
    age_cate = [ range(s, e) for s, e in age_range ]
    v_label_dict = { (1, 1):0, (5, 1):1, (10, 1):2, (16, 1):3, (28, 1):4, (51, 1):5, (75, 1):6,
                     (1, 2):7, (5, 2):8, (10, 2):9, (16, 2):10, (28, 2):11, (51, 2):12, (75, 2):13 }
    
    f_cen_lst = []
    for idx, face in enumerate( faces ):   
        f_eye_l = face['faceLandmarks']['eyeLeftInner']
        f_eye_r = face['faceLandmarks']['eyeRightInner']
        f_cen = ( (f_eye_l['x'] + f_eye_r['x'])/2.0, (f_eye_l['y'] + f_eye_r['y'])/2.0 )
        f_gen = [ 1 if face['attributes']['gender'] else 0 ][0]
        f_age_s = face['attributes']['age']
        f_age = [ int(fn.median(a_c)) for a_c in age_cate if f_age_s in a_c ][0]
        v_label = v_label_dict[ f_age, f_gen ]
        G_q.add_node( idx, label=v_label )
        f_cen_lst.append( f_cen )
        
    f_cen_combi = list( itertools.combinations( f_cen_lst, 2 ) )
    for a, b in f_cen_combi:
        px_dist = fn.euclidean_dist( a, b )
        a_idx = f_cen_lst.index( a )
        b_idx = f_cen_lst.index( b )
        G_q.add_edge( a_idx, b_idx, weight=px_dist )
    
    # Minimum Spanning Tree
    G_q_mst = nx.minimum_spanning_tree( G_q )   
    # Set initial number of Order Distance as 0 For MST edges
    for v1, v2 in G_q_mst.edges():
        G_q_mst[ v1 ][ v2 ][ 'weight' ] = 0
    # Assign Order Distance for All Edges    
    G_q_od = nx.all_pairs_shortest_path_length( G_q_mst, cutoff=4 )
    # Remove the edges visiting itself
    # Decrease Order Distances with -1 
    # Copy MST's Order Distance to Edges Attrs. of Graph G
    for v1 in G_q_od.keys():
        for v2 in G_q_od[ v1 ].keys():
            if v1 == v2:    # Remove Nodes self to self
                del G_q_od[ v1 ][ v2 ]
            else:    # Modify Weights of Other Nodes & Assign them to Graph G
                # Edge Label                        
                order_dist = G_q_od[ v1 ][ v2 ]                                               
                order_dist -= 1
                G_q.add_edge( v1, v2, weight=order_dist )

    return G_q

def get_frequency( G_q, train_g_path ):
    """
    # Projecting a Query Graph to BoFG histogram
    """
    train_lst = fn.get_file_list( train_g_path, '.g' )
    freq_lst = []
    for g_tr in train_lst:
        G_tr = nx.read_gpickle( train_g_path + g_tr )
        # Count the frequency of each G_tr in G_q
        # Check that Node labels of G_tr belongs to that of G_q
        with open( train_g_path + g_tr[:-2] + '.nodelabel', 'r') as nl:
            G_tr_nlabels = json.load( nl )
        G_q_nlabels = [ int(n_tup[1]['label']) for n_tup in G_q.nodes(data=True)]
        if fn.is_subset( G_tr_nlabels, G_q_nlabels ):
            # print G_tr_nlabels, G_tr.edges(data=True)
            nm = iso.numerical_node_match('label', 1)
            em = iso.numerical_edge_match('weight', 0)
            GM = iso.GraphMatcher( G_q, G_tr, node_match=nm, edge_match=em )
    
            frequency = len( list(GM.subgraph_isomorphisms_iter()) )        
            if frequency > 0:
                BoFG_ID = int( g_tr[-5:-2] )
                freq_lst.append( (BoFG_ID, frequency) )
    
    return freq_lst
    
