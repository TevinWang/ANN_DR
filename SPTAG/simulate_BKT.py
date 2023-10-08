import time
import subprocess
import numpy as np
import os
import sys

# adding path
sys.path.insert(0, '/home/jingyuah/SPTAG')
# importing SPTAG
from Release import SPTAG


def testBuild(algo, distmethod, x, out):
    i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
    i.SetBuildParam("NumberOfThreads", '4', "Index")
    i.SetBuildParam("DistCalcMethod", distmethod, "Index")
    if i.Build(x, x.shape[0], False):
        i.Save(out)
    else:
        print("CANNOT BUILD INDEX")


def testAdd(index, x, out, algo, distmethod):
    if index != None:
        i = SPTAG.AnnIndex.Load(index)
    else:
        i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
    i.SetBuildParam("NumberOfThreads", '4', "Index")
    i.SetBuildParam("DistCalcMethod", distmethod, "Index")
    if i.Add(x, x.shape[0], False):
        i.Save(out)
    else:
        print("CANNOT BUILD INDEX")


def testDelete(index, x, out):
   i = SPTAG.AnnIndex.Load(index)
   ret = i.Delete(x, x.shape[0])
   print("deleted and now is: ", ret)


def testSearch(index, q, k):
    j = SPTAG.AnnIndex.Load(index)
    for t in range(q.shape[0]):
        result = j.Search(q[t], k)
        print (result[0]) # ids
        print (result[1]) # distances




def Test(algo, distmethod, data, query, k):
    '''
    Build and search  
    Parameters: 
           algo (str): Graph algorithm ('BKT' or 'KDT') - see Parameters.md for more info
           distmethod (str): Distance comparison method. ('L2' or 'Cosine') 
    '''
    testBuild(algo, distmethod, data, 'testindices')
    result = testSearch('testindices', query, k)


vector_dim = 100                           # dimension
vector_num = 1000                      # dataset size
nq = 1                     # nb of queries
np.random.seed(1234)             # make reproducible

randomData = np.random.random((vector_num, vector_dim)).astype('float32')
randomData.shape


np.save('random_BKT.npy', randomData) # Save the embeddings to .npy
arr = np.load('random_BKT.npy') # Load in the saved embeddings into the notebook


query = arr[:10] # The query vector
k = 5 # Number of results to return, in this case the 5 most similar results

Test('BKT', 'L2', arr, query, k)