
import faiss
from faiss import write_index, read_index
import time 
import pickle
import numpy as np
import fileinput


print("-------------load embeddings------------")

fname = '/data/group_data/cx_group/efficient-dr/embeddings/t5-ance-ms.pickle'
with open(fname, "rb") as f:
    encs = pickle.load(f)

embeds = encs[0] #(8841823, 768)
meta = encs[1]
print(embeds.shape) # embeddings
print(len(meta)) # list of IDs
print("sample meta: ", meta[:10])

vector_num  = encs[0].shape[0] # 8 841 823
vector_dim = encs[0].shape[1] # 768

# vector_dim = 10                          # dimension
# vector_num = 10000                      # dataset size # > centroids=256; >=9984
# nq = 1                     # nb of queries
# np.random.seed(1234)             # make reproducible

# embeds = np.random.random((vector_num, vector_dim)).astype('float32')
# meta = np.arange(vector_num)


assert vector_dim % 2 == 0
pq_m = vector_dim // 2

print("----------------Building HNSW Flat-----------")
index_hnsw = faiss.IndexHNSWPQ(vector_dim, pq_m, 16) # M = 16; n_prob = 8
index_hnsw.metric_type = faiss.METRIC_INNER_PRODUCT

print("-----------------Set Parameters-----------------")
index_hnsw.hnsw.efConstruction = 200 # 40 # default
index_hnsw.hnsw.efSearch = 256

print("-------------------Train---------------------")
index_hnsw.train(embeds)

print("----------------Map Index------------------")
index = faiss.IndexIDMap(index_hnsw)

print("---------------Add-------------------")
index.verbose = True # to see progress
index.add_with_ids(embeds[:vector_num], meta[:vector_num])# dataset 

print("Build ", vector_num, "index with dimension ", vector_dim)

print("-----------------Testing HNSW PQ----------------------")
print("metrics: ", index.metric_type)
# index.hnsw.search_bounded_queue = False # full scan

queries = embeds[:1]
D, I = index.search(queries, k=10)

print("id: ", meta[0])

print("retrieved: ", I)

print("----------------Writing Index-----------------")
# write_index(index, "/home/jingyuah/benchmarks/faiss_hnsw/trial.index")
write_index(index, '/data/group_data/cx_group/ann_index/hnsw/faiss_hnsw_sq_marco.bin')