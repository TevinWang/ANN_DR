
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
# vector_num = 100                      # dataset size
# nq = 1                     # nb of queries
# np.random.seed(1234)             # make reproducible

# randomData = np.random.random((vector_num, vector_dim)).astype('float32')
# ids = np.arange(vector_num)


print("----------------Building HNSW Flat-----------")
index_hnsw  = faiss.IndexHNSWFlat(vector_dim, 32, faiss.METRIC_INNER_PRODUCT) # no training; d: query dim
index_hnsw.hnsw.efConstruction = 200 # 40 # default
index_hnsw.hnsw.efSearch = 256

index = faiss.IndexIDMap(index_hnsw)

index.verbose = True # to see progress
index.add_with_ids(embeds[:vector_num], meta[:vector_num])# dataset 

print("Build ", vector_num, "index with dimension ", vector_dim)

print("-----------------Testing HNSW Flat----------------------")
# index.hnsw.search_bounded_queue = False # full scan

queries = embeds[:1]
D, I = index.search(queries, k=10)

print("id: ", meta[0])

print("retrieved: ", I)

print("----------------Writing Index-----------------")
# write_index(index, "/home/jingyuah/benchmarks/faiss_hnsw/trial.index")
write_index(index, '/data/group_data/cx_group/ann_index/hnsw/faiss_hnsw_marco_ef200.bin')
# efC = 40: approximately 56 min for load + build + save + recall@10
# efC = 200: approximately 4.5 hrs for load + build + save + recall@K
