import sys
import pickle
import numpy as np
import time
import os
import fileinput


print("-----------------DOCUMENTS LOADED-------------------")
sys.stdout.flush()    
t1 = time.time()
# load input data
fname = '/home/jingyuah/embeddings/efficient-dr/embeddings/t5-ance-ms.pickle'
with open(fname, "rb") as f:
    encs = pickle.load(f)

vector_num  = encs[0].shape[0] # 1000000 # encs[0].shape[0] # 8 841 823
vector_dim = encs[0].shape[1] # 768

embeds = encs[0] #(8841823, 768)
meta = encs[1]

t2 = time.time()
print("Embedding Shape: ", embeds.shape) # embeddings
print("Length of Metadata: ", len(meta)) # list of id's = index of embeds
print("--------load docs: ", str(t2-t1), "seconds -----------")
sys.stdout.flush()    


# transfer to bin file
curr_path = os.getcwd()
with open(os.path.join(curr_path, "/home/jingyuah/embeddings/efficient-dr/doc_embed/vectors.bin"), "wb") as f:
    # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
    f.write(vector_num.to_bytes(4, 'little')) # number of points
    f.write(vector_dim.to_bytes(4, 'little')) # dimension
    f.write(embeds.tobytes()) # data array itself


# print("-------- parsing queries---------------")
# sys.stdout.flush()
# # parse queries
# name = '/home/jingyuah/embeddings/efficient-dr/embeddings/embeddings.query.rank.'
# for i in range(8): # 0-7
#     fname = name + str(i)
#     with open(fname, "rb") as fTruth:
#         encs = pickle.load(fTruth)
#     if i == 0:
#         queries = encs[0]
#         q_ids = encs[1]
#     else:
#         queries = np.concatenate((queries, encs[0]))
#         q_ids = np.concatenate((q_ids, encs[1]))

# q_num = queries.shape[0]
# q_dim = queries.shape[1]

# print("shape: ", queries.shape)
# with open(os.path.join(curr_path, "/home/jingyuah/embeddings/efficient-dr/doc_embed/queries.bin"), "wb") as f:
#     # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
#     f.write(q_num.to_bytes(4, 'little')) # number of points
#     f.write(q_dim.to_bytes(4, 'little')) # dimension
#     f.write(queries.tobytes()) # data array itself
    
    
# print("-------- retrieved ground truth documents ---------------")
# sys.stdout.flush()
# # retrieved documents of T5 ANCE
# rname = '/home/jingyuah/efficient-dr/retrieved.train.trec'
# retrieved = {}
# for line in fileinput.input([rname]):
#     ids = line.split(' ')
#     if ids[0] not in retrieved.keys():
#         retrieved[ids[0]] = [[int(ids[2]), int(ids[3])]] # q_id: [d_id, rank]
#     else:
#         retrieved[ids[0]].append([int(ids[2]), int(ids[3])])
# # need to exclude those with less than 100 documents when doing recall
# queries_id = retrieved.keys() 

# gt = []
# for q_id in q_ids:
#     gt.append(retrieved[q_id][0]) # align 

# with open(os.path.join(curr_path, "/home/jingyuah/embeddings/efficient-dr/doc_embed/gt.bin"), "wb") as f:
#     # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
#     f.write(q_dim.to_bytes(4, 'little')) # number of points
#     f.write(vector_dim.to_bytes(4, 'little')) # dimension
#     f.write(gt.tobytes()) # data array itself
