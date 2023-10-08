
from faiss import write_index, read_index
import time 
import pickle
import numpy as np
import fileinput
import json


index = read_index('/data/group_data/cx_group/ann_index/hnsw/faiss_hnsw_sq_marco.bin')
index.efSearch = 500

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


search_num = q_ids.shape[0] # search for all

K = [10, 80, 85, 90, 95, 100]
recall_stats = {}
time_stats = {}
for k in K: 

    recalls = []

    start_time = time.time()
    D, I = index.search(queries[:search_num], k) # I = nq * k
    end_time = time.time()

    for i in range(search_num): # the searched items
        r_for_q =  np.array(retrieved[q_ids[i]]) # q_id: [d_id, rank]
        gt = r_for_q[:k, 0] # list of the d_id's
        set_truth = set(gt)
        set_got = set(I[i])
        
        TP = set_got.intersection(set_truth)
        recall_i =  len(TP)/len(set_truth) # first col is id, sec is rank
        recalls.append(recall_i)

    recall_stats[k] = sum(recalls)/len(recalls)
    # time stats
    time_stats[k] = (end_time-start_time)/search_num

print(recall_stats)
print(time_stats)

with open('/home/jingyuah/benchmarks/faiss_hnsw/hnsw_pq_stats_ef200.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(recall_stats))
     convert_file.write(json.dumps(time_stats))
