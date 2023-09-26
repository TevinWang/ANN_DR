import time
import subprocess
import numpy as np
import os
import sys
import shutil

# adding path
sys.path.insert(0, '/home/jingyuah/SPTAG')
# importing SPTAG
from Release import SPTAG


def testBuild(algo, distmethod, x, out):

    # i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
    # i.SetBuildParam("NumberOfThreads", '4', "Index")
    # i.SetBuildParam("DistCalcMethod", distmethod, "Index")
    # i.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
    # if i.Build(x, x.shape[0], False):
    #     i.Save(out) # Save the index to the disk
    # else:
    #     print("CANNOT BUILD INDEX")
    
    index = SPTAG.AnnIndex('SPANN', 'Float', x.shape[1])

    # Set the thread number to speed up the build procedure in parallel 
    index.SetBuildParam("IndexAlgoType", "BKT", "Base")
    index.SetBuildParam("IndexDirectory", "spann_index", "Base")
    index.SetBuildParam("DistCalcMethod", "L2", "Base")

    index.SetBuildParam("isExecute", "true", "SelectHead")
    index.SetBuildParam("NumberOfThreads", "4", "SelectHead")
    index.SetBuildParam("Ratio", "0.2", "SelectHead") # index.SetBuildParam("Count", "200", "SelectHead")

    index.SetBuildParam("isExecute", "true", "BuildHead")
    index.SetBuildParam("RefineIterations", "3", "BuildHead")
    index.SetBuildParam("NumberOfThreads", "4", "BuildHead")

    index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
    index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
    index.SetBuildParam("PostingPageLimit", "12", "BuildSSDIndex")
    index.SetBuildParam("SearchPostingPageLimit", "12", "BuildSSDIndex")
    index.SetBuildParam("NumberOfThreads", "4", "BuildSSDIndex")
    index.SetBuildParam("InternalResultNum", "32", "BuildSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", "64", "BuildSSDIndex")

    if (os.path.exists(out)):
        shutil.rmtree(out)

    if index.Build(x, x.shape[0], False):
        index.Save(out) # Save the index to the disk


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


vector_dim = 10                          # dimension
vector_num = 100                      # dataset size
nq = 1                     # nb of queries
np.random.seed(1234)             # make reproducible

randomData = np.random.random((vector_num, vector_dim)).astype('float32')
randomData.shape

print("---------------Saving Data--------------------")
np.save('random_spann_1.npy', randomData) # Save the embeddings to .npy

print("---------------Loading Input --------------------")
arr = np.load('random_spann_1.npy') # Load in the saved embeddings into the notebook


query = arr[:1] # The query vector
k = 5 # Number of results to return, in this case the 5 most similar results


print("----------------------BUILD---------------")
build_start = time.time()
testBuild('SPANN', 'L2', arr, 'spann_index')
build_end = time.time()
print("-------------BUILD-----------------")

print("----------------------TEST SEARCH---------------")
test_start = time.time()
result = testSearch('spann_index', query, k)
test_end = time.time()
print("-------------TEST SEARCH-------------------")


# print("BUILD INDEX for", vector_num, "samples with dimension", vector_dim, "took: ", str(build_end - build_start))
print("TEST SEARCH ", str(len(query)), "took: ", str(test_end - test_start))