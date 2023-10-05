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

    # [Base]
    index.SetBuildParam("IndexAlgoType", algo, "Base")
    index.SetBuildParam("IndexDirectory", out, "Base")
    index.SetBuildParam("DistCalcMethod", distmethod, "Base")

    # [SelectHead]
    index.SetBuildParam("isExecute", "true", "SelectHead") # arbitrary tree number
    index.SetBuildParam("NumberOfThreads", "45", "SelectHead") ### -> 4
    index.SetBuildParam("Ratio", "0.2", "SelectHead") # index.SetBuildParam("Count", "200", "SelectHead")\
    # ##
    # index.SetBuildParam("BKTKmeansK", "32", "SelectHead") 
    # index.SetBuildParam("BKTLeafSize", "8", "SelectHead") 
    # index.SetBuildParam("BKTKmeansK", "32", "SelectHead") 
    # index.SetBuildParam("SelectThreshold", "10", "SelectHead") 
    # index.SetBuildParam("SplitFactor", "6", "SelectHead") 
    # index.SetBuildParam("SplitThreshold", "25", "SelectHead") 

    # [BuildHead]
    index.SetBuildParam("isExecute", "true", "BuildHead")
    index.SetBuildParam("RefineIterations", "3", "BuildHead")
    index.SetBuildParam("NumberOfThreads", "45", "BuildHead")
    # ##
    # index.SetBuildParam("NeighborhoodSize", "32", "BuildHead")
    # index.SetBuildParam("TPTNumber", "32", "BuildHead")
    # index.SetBuildParam("TPTLeafSize", "2000", "BuildHead") # dont set maxCheck

    # [BuildSSDIndex]
    index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
    index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
    index.SetBuildParam("PostingPageLimit", "12", "BuildSSDIndex") # <- 3
    index.SetBuildParam("SearchPostingPageLimit", "12", "BuildSSDIndex") # <- 3
    index.SetBuildParam("NumberOfThreads", "45", "BuildSSDIndex") # -> 4
    index.SetBuildParam("InternalResultNum", "64", "BuildSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", "32", "BuildSSDIndex")    
    ##
    # index.SetBuildParam("ReplicaCount", "8", "BuildSSDIndex") # dont set maxCheck and maxDistRatio
    index.SetBuildParam("HashTableExponent", "4", "SearchSSDIndex") # default 2

    if (os.path.exists(out)):
        shutil.rmtree(out)

    if index.Build(x, x.shape[0], False):
        index.Save(out) # Save the index to the disk
    else: 
        print("first build: CANNOT BUILD")
        assert True==False


def testBuildWithMetaData(algo, distmethod, x, s, out):

    index = SPTAG.AnnIndex('SPANN', 'Float', x.shape[1])
    # Set the thread number to speed up the build procedure in parallel 

    # [Base]
    index.SetBuildParam("IndexAlgoType", algo, "Base")
    index.SetBuildParam("IndexDirectory", out, "Base")
    index.SetBuildParam("DistCalcMethod", distmethod, "Base")

    # [SelectHead]
    index.SetBuildParam("isExecute", "true", "SelectHead") # arbitrary tree number
    index.SetBuildParam("NumberOfThreads", "45", "SelectHead") ### -> 4
    index.SetBuildParam("Ratio", "0.2", "SelectHead") # index.SetBuildParam("Count", "200", "SelectHead")\
    ##
    # index.SetBuildParam("BKTKmeansK", "32", "SelectHead") 
    # index.SetBuildParam("BKTLeafSize", "8", "SelectHead") 
    # index.SetBuildParam("BKTKmeansK", "32", "SelectHead") 
    # index.SetBuildParam("SelectThreshold", "10", "SelectHead") 
    # index.SetBuildParam("SplitFactor", "6", "SelectHead") 
    # index.SetBuildParam("SplitThreshold", "25", "SelectHead") 

    # [BuildHead]
    index.SetBuildParam("isExecute", "true", "BuildHead")
    index.SetBuildParam("RefineIterations", "3", "BuildHead")
    index.SetBuildParam("NumberOfThreads", "45", "BuildHead")
    ##
    # index.SetBuildParam("NeighborhoodSize", "32", "BuildHead")
    # index.SetBuildParam("TPTNumber", "32", "BuildHead")
    # index.SetBuildParam("TPTLeafSize", "2000", "BuildHead") # dont set maxCheck

    # [BuildSSDIndex]
    index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
    index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
    index.SetBuildParam("PostingPageLimit", "12", "BuildSSDIndex") # <- 3
    index.SetBuildParam("SearchPostingPageLimit", "12", "BuildSSDIndex") # <- 3
    index.SetBuildParam("NumberOfThreads", "45", "BuildSSDIndex") # -> 4
    index.SetBuildParam("InternalResultNum", "64", "BuildSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", "32", "BuildSSDIndex")    
    ##
    # index.SetBuildParam("ReplicaCount", "8", "BuildSSDIndex") # dont set maxCheck and maxDistRatio
    index.SetBuildParam("HashTableExponent", "4", "SearchSSDIndex") # default 2

    if index.BuildWithMetaData(x, s, x.shape[0], False, False):
       index.Save(out)
    else:
        print("first build: CANNOT BUILD")
        assert True==False


def testSearch(index, q, k):
    j = SPTAG.AnnIndex.Load(index)
    result = []
    for t in range(q.shape[0]):
        retrieved = j.Search(q[t], k)
        result.append(retrieved[0])
    return result

def testSearchSingle(i, q, k):
    idx = SPTAG.AnnIndex.Load(i)

    # [SearchSSDIndex]
    idx.SetBuildParam("isExecute", "true", "SearchSSDIndex")
    idx.SetBuildParam("BuildSsdIndex", "false", "SearchSSDIndex")
    idx.SetBuildParam("InternalResultNum", "96", "SearchSSDIndex")
    idx.SetBuildParam("NumberOfThreads", "4", "SearchSSDIndex") # <- 1
    idx.SetBuildParam("HashTableExponent", "4", "SearchSSDIndex") # default 2
    idx.SetBuildParam("SearchPostingPageLimit", "3", "SearchSSDIndex")

    result = []
    k_doc = idx.Search(q, k) # 3*k
    result.append(k_doc[0]) # k_doc is a tuple
    return np.array(result)


def testSearchWithMetaData(index, q, k):
    j = SPTAG.AnnIndex.Load(index)
    j.SetSearchParam("MaxCheck", '1024', "Index")
    result = []
    for t in range(q.shape[0]):
        retrieved = j.SearchWithMetaData(q[t], k)
        result.append(retrieved)
    return result


# Load Data from T5 ANCE

print("-------------load embeddings------------")

fname = '/data/group_data/cx_group/efficient-dr/embeddings/t5-ance-ms.pickle'
with open(fname, "rb") as f:
    encs = pickle.load(f)

vector_num  = encs[0].shape[0] # 1000000 # encs[0].shape[0] # 8 841 823
vector_dim = encs[0].shape[1] # 768

embeds = encs[0] #(8841823, 768)
meta = encs[1]
print(embeds.shape) # embeddings
print(len(meta)) # list of IDs
print("--------------Input Loaded-----------------------")

idx_path = '/home/jingyuah/SPANN_sim/index'

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

f = open("/home/jingyuah/SPANN_sim/t5_ance_stats.txt", 'w+') # create if non exist

print("----------------------BUILD---------------")
f.write("----------------------BUILD---------------\n")

build_start = time.time()

m = ""
for i in range(len(meta[:vector_num])):
    m += str(meta[i]) + '\n'

testBuildWithMetaData('BKT', 'Cosine', embeds[:vector_num], m, idx_path)

build_end = time.time()

print("BUILD ", str(vector_num), "  with dimension ", str(vector_dim), " took:", str(datetime.timedelta(seconds=build_end-build_start)))
f.write("BUILD " +  str(vector_num) + " with dimension " +  str(vector_dim) + " took: "+ str(datetime.timedelta(seconds=build_end-build_start)) + "\n")
print("-------------BUILD-----------------")
f.write("----------------------BUILD---------------\n")


result = testSearch(idx_path, embeds[:2, :], k=10)
r = np.array(result)
print("search: ", r)

result = testSearchWithMetaData(idx_path, embeds[:2, :], k=10)
r = np.array(result)
print("search with meta: ", r)


# print("----------------------TEST SEARCH---------------")
# f.write("----------------------TEST SEARCH---------------\n")

# q = embeds[0]
# result = testSearchSingle(idx_path, 1, k=10)
# print("result: ", result)

# # qs = embeds[:10]
# # result = testSearch(idx_path, 1, k=10)
# # print(result)

# K = [1, 5, 10]
# search_num = 100
# for k in K:
#     test_start = time.time()
#     query_idx = np.random.choice(embeds.shape[0], search_num, replace=False) # select 100 terms
#     query = embeds[query_idx]
#     result = testSearch(idx_path, query, k)
#     test_end = time.time()
#     print("TEST SEARCH ", str(search_num), "queries with k =", str(k), " took: ", str(datetime.timedelta(seconds=test_end - test_start)))
#     print("   that is ", str(datetime.timedelta(seconds=test_end - test_start)/k), "on average")
#     f.write("TEST SEARCH " + str(search_num) + " queries with k = " +  str(k) + " took: " + str(datetime.timedelta(seconds=test_end - test_start)) + "\n")
#     f.write("   that is " + str(datetime.timedelta(seconds=test_end - test_start)/k) +  " on average\n")

# print("-------------TEST SEARCH-------------------")
# f.write("----------------------TEST SEARCH---------------\n")

f.close()