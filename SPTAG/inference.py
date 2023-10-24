import pickle
import numpy as np
import sys
import fileinput
import time
import json
import argparse
import yaml

# adding path
sys.path.insert(0, '/home/jingyuah/SPTAG')
# importing SPTAG
from Release import SPTAG

def augment_q(queries):
    extracol = np.zeros(len(queries), dtype='float32') # zero queries
    return np.hstack((queries, extracol.reshape(-1, 1)))   

def testSearch(index, q, k):
    j = SPTAG.AnnIndex.Load(index)
    result = []
    for t in range(q.shape[0]):
        retrieved = j.Search(q[t], k)
        result.append(retrieved[0])
    return result



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help='path to config file')
    args = parser.parse_args()

    with open(args.config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())


    print("----------------HYPERPARAM-----------------")
    # process config file

    idx_path = config['idx_path']
    stats_path = config['inf_path']
    num_cores = config['NumberOfThreads']
    max_dist_ratio = config['SearchIndex']['MaxDistRatio']

    print("-------- retrieved ground truth documents ---------------")
    # retrieved documents of T5 ANCE
    rname = '/data/group_data/cx_group/efficient-dr/retrieved.train.trec'
    retrieved = {}
    for line in fileinput.input([rname]):
        ids = line.split(' ')
        if ids[0] not in retrieved.keys():
            retrieved[ids[0]] = [[int(ids[2]), int(ids[3])]] # q_id: [d_id, rank]
        else:
            retrieved[ids[0]].append([int(ids[2]), int(ids[3])])
    # need to exclude those with less than 100 documents when doing recall
    queries_id = retrieved.keys() 


    print("-------- parsing queries---------------")
    # parse queries
    name = '/data/group_data/cx_group/efficient-dr/embeddings/embeddings.query.rank.'
    for i in range(8): # 0-7
        fname = name + str(i)
        with open(fname, "rb") as fTruth:
            encs = pickle.load(fTruth)
        if i == 0:
            queries = encs[0]
            q_ids = encs[1]
        else:
            queries = np.concatenate((queries, encs[0]))
            q_ids = np.concatenate((q_ids, encs[1]))

    # print("quries shape: ", queries.shape)
    # print("q_ids shape", q_ids.shape)
    # print("q_ids[0] type", type(q_ids[0]))

    # print("is q_ids[0] in retrieved keys: ", (q_ids[0] in queries_id))
    # r_for_q =  retrieved[q_ids[0]]
    # print("r_for_q[0]", r_for_q[0])
    # print("array shape: ", np.array(r_for_q).shape)

    print("-----------------transform queries------------------")
    queries = augment_q(queries)
    print("Updated Queries Shape: ", queries.shape)

    print("-------------------load index-----------------------")
    # load ANN index
    index = SPTAG.AnnIndex.Load(idx_path)
    index.SetBuildParam("NumberOfThreads", str(num_cores), "SearchSSDIndex")
    index.SetBuildParam("MaxDistRatio", str(max_dist_ratio), "SearchSSDIndex")
    

    print("-------- computing recall and time ---------------")
    # num of quries to be searched
    search_num = queries.shape[0]

    K = [10, 80, 85, 90, 95, 100]
    recall_stats = {}
    time_stats = {}
    qps_stats = {}

    for k in K: 

        recalls = []
        time_total = 0

        for i in range(search_num):
            
            t0 = time.time()
            r = index.Search(queries[i], k) # quries and q_ids share index # 2*k, id is enough
            t1 = time.time()

            time_total += (t1-t0)
            k_doc = np.array(r[0]) # spann retrieved ids

            r_for_q =  np.array(retrieved[q_ids[i]]) # q_id: [d_id, rank]
            gt = r_for_q[:k, 0] # list of the d_id's
            set_truth = set(gt)
            set_got = set(k_doc)
            # recall rate for this query
            TP = set_got.intersection(set_truth)
            recall_i =  len(TP)/len(set_truth) # first col is id, sec is rank
            recalls.append(recall_i)

        recall_stats[k] = sum(recalls)/len(recalls)
        # time stats
        time_stats[k] = time_total/search_num # convert to ms
        qps_stats[k] = search_num/time_total # queryes per sec

    print("recall: ", recall_stats)
    print("time: ", time_stats)
    print("qps: ", qps_stats)

    with open(stats_path, 'w') as convert_file: 
        convert_file.write(json.dumps(recall_stats))
        convert_file.write(json.dumps(time_stats))
        convert_file.write(json.dumps(qps_stats))