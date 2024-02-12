
import faiss
from faiss import write_index, read_index
import time 
import pickle
import numpy as np
import fileinput
import json
import os 
import sys
import argparse


def augment_train(data, aug_num=1): # tranform the data to 769 dimensional
    norms = (data ** 2).sum(axis=1) # point-wise sq; sum by row
    phi = norms.max() 
    extracol = np.sqrt(phi - norms)
    # stack more
    for i in range(aug_num):
        data = np.hstack((data, extracol.reshape(-1, 1)))
    print("after reshape, data has shape: ", data.shape)
    return data # vertically stack at the end

def augment_q(queries, aug_num=1):
    extracol = np.zeros(len(queries), dtype='float32') # zero queries
    for i in range(aug_num):
        queries = np.hstack((queries, extracol.reshape(-1, 1)))   
    print("after reshape, data has shape: ", queries.shape) 
    return queries  


def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        print("number of queries: ", nvecs)
        print("dimension: ", dim)
        f.seek(4+4)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    print("n: ", n)
    print("d: ", d)
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D





parser = argparse.ArgumentParser()
parser.add_argument("--index_name")
parser.add_argument("--index_dir")
parser.add_argument("--embed_path")
parser.add_argument("--query_path")
parser.add_argument("--gt_path")
parser.add_argument("--stats_path")
parser.add_argument("--efSearch", default=200)
parser.add_argument("--nprobes", default=1200)
parser.add_argument("--refine_frac", default=5)
args = parser.parse_args()

# print configs
print(f'--- --- \n searching {args.index_name} \n \
    with queries at {args.query_path} \n \
    against gt at {args.gt_path} \n \
    and output to {args.stats_path} \n --- ---')


# ################## LOADING 

print("------------------- Load Embeddings -----------------")
name = args.embed_path
embeds = read_fbin(name)
print("embedding shape: ", embeds.shape)
vector_num = embeds.shape[0]
vector_dim = embeds.shape[1]


print("-----------------TRANSFORMATION-------------------")
# tranform input data
embeds = augment_train(embeds, aug_num=0)
print("Updated Embeds Shape: ", embeds.shape)
sys.stdout.flush()
vector_dim = embeds.shape[1]


query_name = args.query_path
gt_name = args.gt_path

queries = read_fbin(query_name)

queries = augment_q(queries=queries, aug_num=0)
print("queries shape: ", queries.shape)

I_gt, D_gt = knn_result_read(gt_name)


print("---------------load index------------------")
dest_dir = args.index_dir
index_name = args.index_name
output_name = f'{index_name}.bin'
index_file = os.path.join(dest_dir, output_name) 
print("loading: ", index_file)
index = read_index(index_file)

index.efSearch = int(args.efSearch)
index.nprobe = int(args.nprobes)


refine_index = faiss.IndexFlatL2(vector_dim)

K = [10, 100]
search_num = queries.shape[0] # search all queries 
recall_stats = {}
time_stats = {}
qps = {}

for k in K: 

    recalls = []

    
    start_time = time.time()
    
    D_c, I_c = index.search(queries[:search_num], int(args.refine_frac)*k) # I = nq * k
    print("retrieved.shape: ", I_c.shape) ##
    
    end_time = time.time()   
    

    for i in range(search_num): # the searched items
        
        # process mapping for rerank
        map = {}
        for idx in range(I_c.shape[1]): # map from the id in the new index to id in the original index  
            map[idx] = I_c[i][idx] # all the retrieved ids for query i 
        
        refine_index.reset()
        refine_index.add(embeds[I_c[i]]) # add the returned result into fine search for this specific 
        D_r, I_nr = refine_index.search(queries[i:i+1], k) # I_nr: (1, k)
        
        # map the result back 
        I_r = []
        for idx in range(I_nr.shape[1]):
            I_r.append(map[I_nr[0][idx]])
            
        # just a progress check    
        if i % 1000 == 0:
            print(f"{i}th reranked retrieved len: {len(I_r)}")
    
        
        gt =  I_gt[i, :k] # need to find the top k
        set_truth = set(gt)
        set_got = set(I_r)
        
        TP = set_got.intersection(set_truth)
        recall_i =  len(TP)/len(set_truth) # first col is id, sec is rank
        recalls.append(recall_i)

    recall_stats[k] = sum(recalls)/len(recalls)
    # time stats
    time_stats[k] = (end_time-start_time)/search_num # in sec
    qps[k] = search_num/(end_time-start_time) # in sec


print(recall_stats)
print(time_stats)

with open(args.stats_path, 'w') as convert_file: 
     convert_file.write(json.dumps(recall_stats))
     convert_file.write(json.dumps(time_stats))
     convert_file.write(json.dumps(qps))
