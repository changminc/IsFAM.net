# -*- coding: utf-8 -*-
import urllib2
import os, time
import json
path = 'query/'
img_files = [ f for f in os.listdir( path ) if f.endswith('.jpg') ]
url = 'https://api.projectoxford.ai/face/v0/detections?analyzesFaceLandmarks=true&analyzesAge=true&analyzesGender=true&analyzesHeadPose=true'

for img in img_files:
    data = open( path + img, "rb")
    length = os.path.getsize( path + img )
    request = urllib2.Request(url=url, data=data)
    request.add_header('Cache-Control', 'no-cache')
    request.add_header('Content-Length', '%d' % length)
    request.add_header('Content-Type', 'application/octet-stream')
    request.add_header('Ocp-Apim-Subscription-Key', 'FILL YOUR SUBSCRIPTION KEY of OXFORD API')
    
    result = urllib2.urlopen(request).read().strip()
    res_obj = json.loads( result )
    
    with open( path + 'attr/' + img[:-4] + '.json', 'w' ) as f:
        json.dump( res_obj, f, indent=4 )

    
    

