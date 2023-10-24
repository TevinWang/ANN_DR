import time
import datetime
import numpy as np
import random
import os
import sys
import shutil
import pickle
import yaml
import argparse

# adding path
sys.path.insert(0, '/home/jingyuah/SPTAG')
# importing SPTAG
from Release import SPTAG


def augment_train(data): # tranform the data to 769 dimensional
    norms = (data ** 2).sum(axis=1) # point-wise sq; sum by row
    phi = norms.max() 
    extracol = np.sqrt(phi - norms)
    return np.hstack((data, extracol.reshape(-1, 1))) # vertically stack at the end

def augment_q(queries):
    extracol = np.zeros(len(queries), dtype='float32') # zero queries
    return np.hstack((queries, extracol.reshape(-1, 1)))    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help='path to config file')
    args = parser.parse_args()

    with open(args.config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())


    print("----------------HYPERPARAM-----------------")
    # process config file

    idx_path = config['idx_path']
    algo = config['Algo']
    distmethod = config['distM']
    num_cores = config['NumberOfThreads']

    SH_BKTKmeansK = config['SelectHead']['BKTKmeansK']
    SH_BKTLeafSize = config['SelectHead']['BKTLeafSize']
    SH_SamplesNumber = config['SelectHead']['SamplesNumber']
    SH_SelectThreshold = config['SelectHead']['SelectThreshold']
    SH_SplitThreshold = config['SelectHead']['SplitThreshold']
    SH_Ratio = config['SelectHead']['Ratio']

    BH_TPTNumber = config['BuildHead']['TPTNumber']
    BH_TPTLeafSize = config['BuildHead']['TPTLeafSize']
    BH_MaxCheck = config['BuildHead']['MaxCheck']
    BH_MaxCheckForRefineGraph = config['BuildHead']['MaxCheckForRefineGraph']
    BH_NeighborhoodSize = config['BuildHead']['NeighborhoodSize']

    BSSD_PostingPageLimit = config['BuildSSDIndex']['PostingPageLimit']
    BSSD_MaxCheck = config['BuildSSDIndex']['MaxCheck']
    BSSD_SearchPostingPageLimit = config['BuildSSDIndex']['SearchPostingPageLimit']
    BSSD_MaxDistRatio = config['BuildSSDIndex']['MaxDistRatio']
    BSSD_SearchInternalResultNum = config['BuildSSDIndex']['SearchInternalResultNum']

    SI_MaxCheck = config['SearchIndex']['MaxCheck']
    SI_SearchPostingPageLimit = config['SearchIndex']['SearchPostingPageLimit']
    SI_SearchInternalResultNum = config['SearchIndex']['SearchInternalResultNum']
    SI_MaxDistRatio = config['SearchIndex']['MaxDistRatio']


    print("-----------------DOCUMENTS LOADED-------------------")

    # load input data
    fname = '/data/group_data/cx_group/efficient-dr/embeddings/t5-ance-ms.pickle'
    with open(fname, "rb") as f:
        encs = pickle.load(f)

    vector_num  = encs[0].shape[0] # 1000000 # encs[0].shape[0] # 8 841 823
    vector_dim = encs[0].shape[1] # 768

    embeds = encs[0] #(8841823, 768)
    meta = encs[1]
    print("Embedding Shape: ", embeds.shape) # embeddings
    print("Length of Metadata: ", len(meta)) # list of id's = index of embeds

    # vector_dim = 768                        # dimension
    # vector_num = 1000                      # dataset size
    # nq = 1                     # nb of queries
    # np.random.seed(1234)             # make reproducible
    # embeds = np.random.random((vector_num, vector_dim)).astype('float32')
    # print("Vector Shape: ", embeds.shape)


    # prepare meta data
    m = ""
    for i in range(vector_num):
        m += str(meta[i]) + '\n'
        # m += str(i) + '\n'

    print("-----------------INPUT LOADED-------------------")


    print("-----------------TRANSFORMATION-------------------")
    # tranform input data
    embeds = augment_train(embeds)
    print("Updated Embeds Shape: ", embeds.shape)
    vector_dim = embeds.shape[1]

    print("-----------------TRANSFORMATION-------------------")


    
    print("-----------------START BUILDING-------------------")
    # build with meta data
    index = SPTAG.AnnIndex('SPANN', 'Float', embeds.shape[1])
    # Set the thread number to speed up the build procedure in parallel 

    # [Base]
    index.SetBuildParam("IndexAlgoType", algo, "Base")
    index.SetBuildParam("IndexDirectory", idx_path, "Base")
    index.SetBuildParam("DistCalcMethod", distmethod, "Base")

    # [SelectHead]
    index.SetBuildParam("isExecute", "true", "SelectHead")
    index.SetBuildParam("TreeNumber", "1", "SelectHead")
    index.SetBuildParam("BKTKmeansK", str(SH_BKTKmeansK), "SelectHead")
    index.SetBuildParam("BKTLeafSize", str(SH_BKTLeafSize), "SelectHead")
    index.SetBuildParam("SamplesNumber", str(SH_SamplesNumber), "SelectHead")
    index.SetBuildParam("SelectThreshold", str(SH_SelectThreshold), "SelectHead")  # default 10
    index.SetBuildParam("SplitFactor", "6", "SelectHead")    
    index.SetBuildParam("SplitThreshold", str(SH_SplitThreshold), "SelectHead")  # default 25
    index.SetBuildParam("Ratio", str(SH_Ratio), "SelectHead")   
    index.SetBuildParam("NumberOfThreads", str(num_cores), "SelectHead") ### -> 4
    index.SetBuildParam("BKTLambdaFactor", "-1", "SelectHead")

    # [BuildHead]
    index.SetBuildParam("isExecute", "true", "BuildHead")
    index.SetBuildParam("NeighborhoodSize", str(BH_NeighborhoodSize), "BuildHead")
    index.SetBuildParam("TPTNumber", str(BH_TPTNumber), "BuildHead") # default 32
    index.SetBuildParam("TPTLeafSize", str(BH_TPTLeafSize), "BuildHead") # default 2000
    index.SetBuildParam("MaxCheck", str(BH_MaxCheck), "BuildHead") # default 4096
    index.SetBuildParam("MaxCheckForRefineGraph", str(BH_MaxCheckForRefineGraph), "BuildHead") # default 8192
    index.SetBuildParam("RefineIterations", "3", "BuildHead")
    index.SetBuildParam("NumberOfThreads", str(num_cores), "BuildHead")
    index.SetBuildParam("BKTLambdaFactor", "-1", "BuildHead")

    # [BuildSSDIndex]
    index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
    index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
    index.SetBuildParam("InternalResultNum", "64", "BuildSSDIndex") 
    index.SetBuildParam("ReplicaCount", "8", "BuildSSDIndex")
    index.SetBuildParam("PostingPageLimit", str(BSSD_PostingPageLimit), "BuildSSDIndex") # default 12
    index.SetBuildParam("NumberOfThreads", str(num_cores), "BuildSSDIndex")
    index.SetBuildParam("MaxCheck", str(BSSD_MaxCheck), "BuildSSDIndex") # default 4096
    index.SetBuildParam("SearchPostingPageLimit", str(BSSD_SearchPostingPageLimit), "BuildSSDIndex") # default 12
    index.SetBuildParam("SearchInternalResultNum", str(BSSD_SearchInternalResultNum), "BuildSSDIndex")
    index.SetBuildParam("MaxDistRatio", str(BSSD_MaxDistRatio), "BuildSSDIndex")

    # [SearchIndex]
    index.SetBuildParam("MaxCheck", str(SI_MaxCheck), "SearchSSDIndex")
    index.SetBuildParam("NumberOfThreads", str(num_cores), "SearchSSDIndex")
    index.SetBuildParam("SearchPostingPageLimit", str(SI_SearchPostingPageLimit), "SearchSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", str(SI_SearchInternalResultNum), "SearchSSDIndex")
    index.SetBuildParam("MaxDistRatio", str(SI_MaxDistRatio), "SearchSSDIndex") # default 8.0


    t1 = time.time()

    if index.BuildWithMetaData(embeds, m, embeds.shape[0], False, False):
        index.Save(idx_path)
    else:
        print("first build: CANNOT BUILD")
        assert True==False

    t2 = time.time()
    print("BUILDING TOOK: ", str(t2 - t1), "seconds")


    print("-------------FINISHED BUILDING-------------")

    print("---------------TRIAL SERACH------------------")

    result = index.Search(embeds[0], 10)
    r = np.array(result[0])
    print("NN 10 for embeds[0]: ", r)

    print("---------------BUILD FILE ENDED------------------")