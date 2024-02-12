import sys
import pickle
import numpy as np
import time
import os
import fileinput
import struct
import random 
 


# # old q_ids
# print("-------- parsing queries---------------")
# sys.stdout.flush()
# # parse queries
# name = '/data/user_data/jingyuah/sparse_disk/embeddings/embeddings.query.rank.'
# for i in range(8): # 0-7
#     fname = name + str(i)
#     with open(fname, "rb") as fTruth:
#         encs = pickle.load(fTruth)
#     if i == 0:
#         old_q_ids = np.array(encs[1]).astype(int)
#     else:
#         old_q_ids = np.concatenate((old_q_ids, np.array(encs[1]).astype(int)))
            

# new q_ids 
print("----------new queries-----------")
sys.stdout.flush()
query_path = "/data/user_data/yuweia/new-query"
q_doc_prefix = "query_rank_"


with open(os.path.join(query_path, q_doc_prefix + '0'), 'rb') as f:
    encs = pickle.load(f)
    q_ids = np.array(encs[1]).astype(int)
    queries_embed = encs[0] # only embeddings

for i in range(1, 11):
    with open(os.path.join(query_path, q_doc_prefix + str(i)), 'rb') as f:
        encs = pickle.load(f)
        q_ids = np.concatenate((q_ids, np.array(encs[1]).astype(int)))
        queries_embed = np.concatenate((queries_embed, encs[0]))
        
        

############# RERANK Q_IDS ####################

# # rerank from start 0 for old_q_ids
# t5_indices = np.argsort(old_q_ids) # sort the q_ids
# print("first 10 queries: ", old_q_ids[t5_indices][:10])

# rerank from start 0 for q_ids
llama_indices = np.argsort(q_ids)

# # # rerank the query embeddings based on the sorted q_ids
# print("sum of match: ", str(sum(old_q_ids[t5_indices] == q_ids[llama_indices])))
print("reranking embedding based on ascending q id ")
rerank_embed = queries_embed[llama_indices]
rerank_ids = q_ids[llama_indices]    
print("first 10 queries: ", rerank_ids[:10])
print("last 10 queries: ", rerank_ids[-10:])


##################### writing and checking



# print("sparse 0.0 has shape: ", queries_embed[:, 0, :].shape)
# print("sparse 0.2 has shape: ", queries_embed[:, 1, :].shape)

# print("the first", queries_embed[0, 0, :10])



# queries -> sparsity 0, 0.2, 0.4

# only top 1w queries
sparsities = ["00"] # ["00", "02", "04"]

for i, sparsity in enumerate(sparsities):
    
    curr_embed = rerank_embed[:, i, :]
    
    q_num = curr_embed.shape[0]
    q_dim = curr_embed.shape[1]
    
    print("i: ", i, "with sparsity", sparsity)
    print(curr_embed.shape)
    
    with open("/data/user_data/jingyuah/sparse_disk/llama/queries_full.bin", "wb") as f:
        # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
        f.write(q_num.to_bytes(4, 'little')) # number of points
        f.write(q_dim.to_bytes(4, 'little')) # dimension
        f.write(curr_embed.tobytes()) # data array itself
    


    for num in [10000, 30000]:
        indices = random.sample(range(0, q_num), num)
        sample_queries = curr_embed[indices]
        sample_ids = rerank_ids[indices]
        print("sample shape: ", sample_queries.shape)
        print("sample id shape: ", sample_ids.shape)
        
        output_file = "/data/user_data/jingyuah/sparse_disk/llama/queries_" + str(num) + ".bin"
        print("outputing to: ", output_file)
        with open(output_file, "wb") as f:
            # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
            f.write(num.to_bytes(4, 'little')) # number of points
            f.write(q_dim.to_bytes(4, 'little')) # dimension
            f.write(sample_queries.tobytes()) # data array itself
            
        output_ids = "/data/user_data/jingyuah/sparse_disk/llama/q_ids_" + str(num) + ".pkl"
        print("outputing ids to: ", output_ids)
        with open(output_ids, 'wb') as f:
            pickle.dump(sample_ids, f)
        


# ############################
# # check sparsity 0.6 document 
# name = "/home/jingyuah/embeddings/efficient-dr/sparse/doc_embed/q_embed/q_embed_06.bin"

# # read the meta data
# with open(name, "rb") as f:
#     # read the meta data
    
#     print("nrows: ", int.from_bytes(f.read(4), "little"))
#     print("ncols: ", int.from_bytes(f.read(4), "little"))
    
#     dists = []
#     for i in range(queries_embed.shape[0]):
#         curr_r = np.empty(shape=(768)) # init space for the retrieved docs
#         for j in range(768):
#             byte = f.read(4)
#             curr_r[j] = struct.unpack('f', byte)[0]
#         dists.append(curr_r)    
        

# dists = np.array(dists)
# print("dists shape: ", dists.shape)
# print("the first five dists for the first query: ", dists[0, :10])