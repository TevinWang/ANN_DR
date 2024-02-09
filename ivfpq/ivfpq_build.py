
import faiss
from faiss import write_index, read_index
import time 
import pickle
import numpy as np
import fileinput
import sys
import argparse
import os

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




parser = argparse.ArgumentParser()
parser.add_argument("--index_name")
parser.add_argument("--metrics", default='dot')
parser.add_argument("--efConstruction", default=300)
args = parser.parse_args()


# ########### MARCO Embedding
# print("-------------load embeddings------------")

# fname = '/data/user_data/jingyuah/sparse_disk/embeddings/t5-ance-ms.pickle'
# with open(fname, "rb") as f:
#     encs = pickle.load(f)

# embeds = encs[0] #(8841823, 768)
# meta = encs[1]
# print(embeds.shape) # embeddings
# print(len(meta)) # list of IDs

# vector_num  = encs[0].shape[0] # 8 841 823
# vector_dim = encs[0].shape[1] # 768


# ################### RANDOM Embedding

# vector_dim = 768                          # dimension
# vector_num = 10000                      # dataset size # > centroids=256; >=9984
# nq = 1                     # nb of queries
# np.random.seed(1234)             # make reproducible

# embeds = np.random.random((vector_num, vector_dim)).astype('float32')
# meta = np.arange(vector_num)


# ###################### MSTuring-1M
print("------------------- Load Embeddings -----------------")
name = '/home/jingyuah/big-ann-benchmarks/data/msturing-1M/base1b.fbin.crop_nb_1000000'
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

print("-----------------TRANSFORMATION-------------------")


print("----------------Building Quantizer and IVFPQ Index-----------")
if args.metrics == "dot":
    metrics = faiss.METRIC_INNER_PRODUCT
else: 
    metrics = faiss.METRIC_L2 
    
nlist = 4800 # SPANN -> 32 * 10 # base: 3200
q_nlist = 512
m = 25 # base: 128


# # OLD 
# index_name = f'ivfpq_m{m}_nlist_{nlist}'
# quantizer = faiss.IndexFlatL2(vector_dim)  # this remains the same
# index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, 8) # 8: each sub-vector is encoded as 8 bits

# # OLD
# index_name = f'IVF{nlist},PQ{m}x4fsr,RFlat'
# index_name = f'IVF{nlist}(IVF{q_nlist},PQ{m}x4fs,RFlat)'
# index_name = f'IVF{nlist},PQ{m}x4fsr,Refine(SQfp16)'

index_name = args.index_name
print("building: ", index_name)
index = faiss.index_factory(vector_dim, index_name, metrics)

print("-----------------Set Parameters-----------------")
# index.efConstruction = 200 # 500 # 40 # default
index.efConstruction = args.efConstruction

index.efSearch = 200

print("-------------------Train---------------------")
index.train(embeds[:vector_num])
assert index.is_trained

print("---------------Add-------------------")
index.verbose = True # to see progress
index.add(embeds[:vector_num]) # dataset 

print("Build ", vector_num, "index with dimension ", vector_dim)

print("-----------------Testing IVFPQ----------------------")
if index.metric_type == 0:
    print("metrics: IP")
elif index.metric_type == 1: 
    print("metrics: L2")
# index.hnsw.search_bounded_queue = False # full scan

queries = embeds[:1]
t1 = time.time()
D, I = index.search(queries, k=10)
t2 = time.time()

print("retrieved: ", I)
print("took: ", str(t2-t1))

index.nprobe = 320  # slower -> postinglist = 10 # 10% centroid

queries = embeds[:1]
t1 = time.time()
D, I = index.search(queries, k=10)
t2 = time.time()

print("retrieved: ", I)
print("took: ", str(t2-t1))

print("----------------Writing Index-----------------")
dest_dir = '/data/user_data/jingyuah/faiss/ivfpq'
output_name = f'{index_name}.bin'
write_index(index, os.path.join(dest_dir, output_name))