import numpy as np
import h5py
import os
import requests
import tempfile
import time

import scann


def read_ibin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        print("nvecs: ", nvecs)
        print("dim: ", dim)
        f.seek(4+4)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.uint8,
                          offset=start_idx * 4 * dim)
        
    # print(arr)
    return arr.reshape(nvecs, dim)


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
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D


def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size



# take the data and query
data_path  = '/home/jingyuah/big-ann-benchmarks/data/msturing-1M/base1b.fbin.crop_nb_1000000'
dataset = read_fbin(data_path)
query_path = '/home/jingyuah/big-ann-benchmarks/data/msturing-1M/query100K.fbin'
queries = read_fbin(query_path)
gt_path = '/home/jingyuah/big-ann-benchmarks/data/msturing-1M/msturing-gt-1M'
gt, D = knn_result_read(gt_path) # the first 100K quries -> gt ids are int

print("Gt example: ", gt[0, :10])

# ###############
# the golve dataset
# #################
# with tempfile.TemporaryDirectory() as tmp:
#     response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
#     loc = os.path.join(tmp, "glove.hdf5")
#     with open(loc, 'wb') as f:
#         f.write(response.content)
    
#     glove_h5py = h5py.File(loc, "r")
# list(glove_h5py.keys())
# ['distances', 'neighbors', 'test', 'train']
# dataset = glove_h5py['train']
# queries = glove_h5py['test']
# print(dataset.shape)
# print(queries.shape)
# gt = glove_h5py['neighbors']


# normalize
normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README

# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
# k=10 here should be smaller than the number of leaves 
searcher = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product").tree(
    num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()
    
    
# this will search the top 100 of the 2000 leaves, and compute
# the exact dot products of the top 100 candidates from asymmetric
# hashing to get the final top 10 candidates.
start = time.time()
neighbors, distances = searcher.search_batched(queries)
end = time.time()

# we are given top 100 neighbors in the ground truth, so select top 10
print("Recall:", compute_recall(neighbors, gt[:, :10]))
print("Time:", end - start)

# #########################
# increasing the leaves to search increases recall at the cost of speed
start = time.time()
neighbors, distances = searcher.search_batched(queries, leaves_to_search=150)
end = time.time()

print("Recall:", compute_recall(neighbors, gt[:, :10]))
print("Time:", end - start)

# #########################
# increasing reordering (the exact scoring of top AH candidates) has a similar effect.
start = time.time()
neighbors, distances = searcher.search_batched(queries, leaves_to_search=150, pre_reorder_num_neighbors=250)
end = time.time()

print("Recall:", compute_recall(neighbors, gt[:, :10]))
print("Time:", end - start)

# #########################
# we can also dynamically configure the number of neighbors returned
# currently returns 10 as configued in ScannBuilder()
neighbors, distances = searcher.search_batched(queries)
print(neighbors.shape, distances.shape)

# now returns 20
neighbors, distances = searcher.search_batched(queries, final_num_neighbors=20)
print(neighbors.shape, distances.shape)

# #########################
# we have been exclusively calling batch search so far; the single-query call has the same API
start = time.time()
neighbors, distances = searcher.search(queries[0], final_num_neighbors=5)
end = time.time()

print(neighbors)
print(distances)
print("Latency (ms):", 1000*(end - start))


# create serialize target dir
os.makedirs('./trial_searcher_ms-1M', exist_ok=True)

# serialize the searcher
searcher.serialize('./trial_searcher_ms-1M')

# later... restore the searcher
searcher = scann.scann_ops_pybind.load_searcher('./trial_searcher_ms-1M')

# #########################
# we have been exclusively calling batch search so far; the single-query call has the same API
start = time.time()
neighbors, distances = searcher.search(queries[0], final_num_neighbors=5)
end = time.time()

print(neighbors)
print(distances)
print("Latency (ms):", 1000*(end - start))