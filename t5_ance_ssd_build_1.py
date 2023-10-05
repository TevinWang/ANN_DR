import time
import datetime
import numpy as np
import random
import os
import sys
import shutil
import pickle

# adding path
sys.path.insert(0, '/home/jingyuah/SPTAG')
# importing SPTAG
from Release import SPTAG


def testBuild(algo, distmethod, x, out):
    
    index = SPTAG.AnnIndex('SPANN', 'Float', x.shape[1])

    # Set the thread number to speed up the build procedure in parallel 
    index.SetBuildParam("IndexAlgoType", algo, "Base")
    index.SetBuildParam("IndexDirectory", "spann_index", "Base")
    index.SetBuildParam("DistCalcMethod", distmethod, "Base")

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


def testBuildWithMetaData(algo, distmethod, x, s, out):

    index = SPTAG.AnnIndex('SPANN', 'Float', x.shape[1])
    # Set the thread number to speed up the build procedure in parallel 
    index.SetBuildParam("IndexAlgoType", algo, "Base")
    index.SetBuildParam("IndexDirectory", "spann_index", "Base")
    index.SetBuildParam("DistCalcMethod", distmethod, "Base")

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
    if index.BuildWithMetaData(x, s, x.shape[0], False, False):
       index.Save(out)


def testAdd(i, x, out, algo, distmethod):
    if i != None:
        index = SPTAG.AnnIndex.Load(i)
    else:
        index = SPTAG.AnnIndex('SPANN', 'Float', x.shape[1])

    # Set the thread number to speed up the build procedure in parallel 
    index.SetBuildParam("IndexAlgoType", algo, "Base")
    index.SetBuildParam("IndexDirectory", "spann_index", "Base")
    index.SetBuildParam("DistCalcMethod", distmethod, "Base")

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

    if index.Add(x, x.shape[0], False):
        index.Save(out)
    else:
        print("CANNOT BUILD INDEX")
 

def testAddWithMetaData(i, x, s, out, algo, distmethod):
    
    if i != None:
        index = SPTAG.AnnIndex.Load(i)
    else:
        index = SPTAG.AnnIndex('SPANN', 'Float', x.shape[1])

    # Set the thread number to speed up the build procedure in parallel 
    index.SetBuildParam("IndexAlgoType", algo, "Base")
    index.SetBuildParam("IndexDirectory", "spann_index", "Base")
    index.SetBuildParam("DistCalcMethod", distmethod, "Base")

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
    if index.AddWithMetaData(x, s, x.shape[0], False, False):
        index.Save(out)
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


# vector_dim = 10                          # dimension
# vector_num = 100                      # dataset size
# nq = 1                     # nb of queries
# np.random.seed(1234)             # make reproducible

# randomData = np.random.random((vector_num, vector_dim)).astype('float32')
# randomData.shape

# Load Data from T5 ANCE


fname = '/data/group_data/cx_group/efficient-dr/embeddings/t5-ance-ms.pickle'
with open(fname, "rb") as f:
    encs = pickle.load(f)

vector_num  = encs[0].shape[0] # 8841823
vector_dim = encs[0].shape[1] # 768

embeds = encs[0] #(8841823, 768)
meta = encs[1]
print(embeds.shape) # embeddings
print(len(meta)) # list of IDs
print("--------------Input Loaded-----------------------")

'''
save and load random data
'''
'''
vector_dim = 10                          # dimension
vector_num = 100                      # dataset size
nq = 1                     # nb of queries
np.random.seed(1234)             # make reproducible

randomData = np.random.random((vector_num, vector_dim)).astype('float32')
randomData.shape

print("---------------Saving Data--------------------")
np.save('t5_ance.npy', embeds) # Save the embeddings to .npy
print("---------------Loading Input --------------------")
arr = np.load('random_spann_1.npy') # Load in the saved embeddings into the notebook


query = arr[:1] # The query vector
k = 5 # Number of results to return, in this case the 5 most similar results
'''

f = open("/home/jingyuah/build_1.txt", 'w+') # create if non exist

print("----------------------BUILD---------------")
f.write("----------------------BUILD---------------\n")
build_start = time.time()


i = 100000
# Prepare metadata for each vectors, separate them by '\n' <- only separator supported
m = ""
for j in range(0, i):
    m += str(meta[j]) + '\n'
testBuildWithMetaData('BKT', 'Cosine', embeds[:i], m, 'spann_index')

i = i + 100000
while i < vector_num:

    # Prepare metadata for each vectors, separate them by '\n' <- only separator supported
    m = ""
    for j in range(i-100000, i):
        m += str(meta[j]) + '\n'

    testAddWithMetaData('spann_index', embeds[i-100000:i], m, 'spann_index', 'BKT', 'Cosine')
    # testAdd('spann_index', embeds[i-100000:i], 'spann_index', 'SPANN', 'L2')
    i = i + 100000

m = ""
for j in range(i-100000, vector_num):
    m += str(meta[j]) + '\n'  
# testAdd('spann_index', embeds[i-100000:], 'spann_index', 'SPANN', 'L2')
testAddWithMetaData('spann_index', embeds[i-100000:], m, 'spann_index', 'BKT', 'L2')


build_end = time.time()
print("BUILD ", str(vector_num), "  with dimension ", str(vector_dim), " took:", str(datetime.timedelta(seconds=build_end-build_start)))
f.write("BUILD " +  str(vector_num) + " with dimension " +  str(vector_dim) + "took: "+ str(datetime.timedelta(seconds=build_end-build_start)) + "\n")
print("-------------BUILD-----------------")
f.write("----------------------BUILD---------------\n")


print("----------------------TEST SEARCH---------------")
f.write("----------------------TEST SEARCH---------------\n")
K = [85, 90, 95, 100]
search_num = 100
for k in K:
    test_start = time.time()
    query_idx = np.random.choice(embeds.shape[0], search_num, replace=False) # select 100 terms
    query = embeds[query_idx]
    result = testSearch('spann_index', query, k)
    test_end = time.time()
    print("TEST SEARCH ", str(search_num), "queries with k =", str(k), "took: ", str(datetime.timedelta(seconds=test_end - test_start)))
    print("   that is ", str(datetime.timedelta(seconds=test_end - test_start)/k), "on average")
    f.write("TEST SEARCH " + str(search_num) + " queries with k = " +  str(k) + " took: " + str(datetime.timedelta(seconds=test_end - test_start)) + "\n")
    f.write("   that is " + str(datetime.timedelta(seconds=test_end - test_start)/k) +  " on average\n")
print("-------------TEST SEARCH-------------------")
f.write("----------------------TEST SEARCH---------------\n")

f.close()