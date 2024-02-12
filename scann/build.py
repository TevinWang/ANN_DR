import sys
import numpy as np
import h5py
import os
import requests
import tempfile
import time
import argparse

import scann


def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        print("number of rows: ", nvecs)
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


def build_index(dataset, args):    
    
    # normalize
    normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]\

    # use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
    searcher = scann.scann_ops_pybind.builder(
            normalized_dataset, args.K, "dot_product"
        ).tree(
            num_leaves=args.num_leaves, 
            num_leaves_to_search=args.num_leaves_to_search, 
            training_sample_size=args.train_size
        ).score_ah(
            2, 
            anisotropic_quantization_threshold=0.2
        ).reorder(
            100
        ).build()
        
    # serialize the searcher and save 
    searcher.serialize(args.index_path)
    return searcher


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # the logistics 
    parser.add_argument("--data", type=str, help='path to the data file')
    parser.add_argument("--query", type=str, help='path to the query file')
    parser.add_argument("--gt", type=str,help='path to the ground truth fie')
    parser.add_argument("--index_path", type=str,help='path to the index file, should be a directory')
    # build
    parser.add_argument("--num_leaves", type=int, help='number of leaves to build')
    parser.add_argument("--train_size", type=int, help='training sample size')
    # search
    parser.add_argument("--num_leaves_to_search", type=int, help='number of leaves to search')
    parser.add_argument("--K", type=int,help='number of neighbors to return')
    parser.add_argument("--pre_reorder_num_neighbors", type=int,help='the exact scoring of top AH candidates')
    parser.add_argument("--search_only", type=str,help='just perform search on existing index')
    args = parser.parse_args()
    # print(args)
    
    
    # take the data and query
    dataset = read_fbin(args.data)
    queries = read_fbin(args.query)
    gt, D = knn_result_read(args.gt) 
    
    # warning strings
    index_not_found = "The Index Directory Is Empty or Non-Exist"
    index_dir_not_empty = "The Index Directory Is Not Empty"
     
    # check indexing directory, if not exist, build
    if args.search_only == "True":
        assert os.path.isdir(args.index_path) and len(os.listdir(args.index_path)) != 0, index_not_found
        searcher = scann.scann_ops_pybind.load_searcher(args.index_path)
    else:
        if not os.path.isdir(args.index_path):
            os.mkdir(args.index_path)
        assert len(os.listdir(args.index_path)) == 0, index_dir_not_empty
        # build index
        searcher = build_index(dataset, args)
    
    
    # search 
    start = time.time()
    
    neighbors, distances = searcher.search_batched(
        queries, 
        leaves_to_search=args.num_leaves_to_search, 
        final_num_neighbors=args.K, 
        pre_reorder_num_neighbors=args.pre_reorder_num_neighbors
    )
    
    end = time.time()
    
    
    recall = compute_recall(neighbors, gt[:, :args.K])
    latency = 1000*(end-start)/queries.shape[0]
    qps = queries.shape[0]/(end-start)
    
    print("Recall:", str(recall))
    print("Time in second:", str(end - start))
    print("Average Latency:", str(latency))
    print("Average QPS:", str(qps))
