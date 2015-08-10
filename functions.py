# -*- coding: utf-8 -*-
import re
import numpy
import math 
import os 
from collections import Counter

def get_file_list( path, ext ):
    return [ f for f in os.listdir(path) if f.endswith( ext ) ]
    
def get_subg( subg_txt ):
    with open( subg_txt, 'r' ) as s:
        s_txt = s.read()        
        pattern_ = re.compile('(t\s#\s\d+\s\*\s\d+\n)((v\s\d+\s\d+\n)+)((e\s\d+\s\d+\s\d+\n)+)(\s\{.*\}\n)')    
        searched = pattern_.findall( s_txt )        
        cnt = len( searched )
    return searched, cnt

def get_num_jpg( jpg_txt ):
    with open( jpg_txt, 'r' ) as j:
        j_txt = j.read()        
        pattern_ = re.compile('.+jpg\n')    
        searched = pattern_.findall( j_txt )
        cnt = len( searched )
    return cnt
    
def euclidean_dist(xy1, xy2):    
    px_dist = math.sqrt((xy1[0] - xy2[0])**2 + (xy1[1] - xy2[1])**2)
    return px_dist
    
def median( lst ):
    return numpy.median( numpy.array(lst) )
    
def is_subset( lst1, lst2 ):
    c1, c2 = Counter( lst1 ), Counter( lst2 )
    for k, n in c1.items():
        if n > c2[k]:
            return False
    return True