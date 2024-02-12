
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
parser.add_argument("--dest_dir")
parser.add_argument("--embed_path")
parser.add_argument("--metrics", default='dot')
parser.add_argument("--efConstruction", default=300)
args = parser.parse_args()

# print configs
print(f'--- --- \n building {args.index_name} \n \
    from vector {args.embed_path} \n \
    to destination at {args.dest_dir} \n --- ---')


# ########## LOADING VECTORS 
print("------------------- Load Embeddings -----------------")
name = args.embed_path
embeds = read_fbin(name)
print("=---------- shard shape --------", embeds.shape)
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
dest_dir = args.dest_dir
# output_name = index_name.replace(',', '') + 'bin'
output_name = f'{index_name}.bin'
write_index(index, os.path.join(dest_dir, output_name))