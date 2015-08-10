# IsFAM.net
<Discriminative Subgraphs for Discovering Family Photos>
- We apply CORK algorithm for classifying group photos into family and non-family.

<Requirements on Python Libraries>
* networkx==1.9.1
* numpy==1.9.2
* Pillow==2.9.0
* requests==2.7.0
* LibSVM==3.20: you may install manually ( https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download )
* also, CORK algorithm by Marisa Thoma: you can use one of them. 
  - (original) http://www.dbs.ifi.lmu.de/~thoma/pub/sam2010/sam2010.zip
  - (extension) https://code.google.com/p/grad-proj/source/browse/#svn%2Ftrunk%2Fgspancpp%2Fbin%253Fstate%253Dclosed


<Example of querying>
- First of all, you need to get a subsrciption key of face detection API from Project Oxford AI: https://www.projectoxford.ai/face
  - it returns a json with age and gender and face positions in an image.  
  - Fill your subscription key into 'projection.py',
  
    ```
    request.add_header('Ocp-Apim-Subscription-Key', 'FILL YOUR SUBSCRIPTION KEY at OXFORD API')
    ```
- Then, **execute querying.py**




* If you want to test your own image, replace the q1.jpg with yours in 'query' folder.
* If you want to change the train model, chage the model number in querying.py,
  ```
  model_path = 'model_1'
  ```
