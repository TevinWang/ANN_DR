import sys
import pickle
import numpy as np
import time
import os
import fileinput


print("-------- parsing docs---------------")
sys.stdout.flush()
s = 0
# parse queries
name = '/data/user_data/jingyuah/sparse_disk/embeddings/embeddings.query.rank.'
for i in range(8): # 0-8
    print("i: ", i)
    fname = name + str(i)
    with open(fname, "rb") as fTruth:
        encs = pickle.load(fTruth)
        num = encs[0].shape[0]
        # reshape_encs = np.reshape(encs[1], (-1, 768))
        # s += reshape_encs.shape[0]
        
        print(encs[0].shape)
        
    if i == 0:
        embed = encs[0]
    else:
        embed = np.concatenate((embed, encs[0]))
        
print(s)
        
print("shape: ", embed.shape)
q_num = embed.shape[0]
q_dim = embed.shape[1]

with open("/data/user_data/jingyuah/sparse_disk/doc_embed/marco_queries.bin", "wb") as f:
    # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
    f.write(q_num.to_bytes(4, 'little')) # number of points
    f.write(q_dim.to_bytes(4, 'little')) # dimension
    f.write(embed.tobytes()) # data array itself
    
    
print("-------------load embeddings------------")

fname = '/data/user_data/jingyuah/sparse_disk/embeddings/t5-ance-ms.pickle'
with open(fname, "rb") as f:
    encs = pickle.load(f)

embeds = encs[0] #(8841823, 768)
meta = encs[1]

q_num = embeds.shape[0]
q_dim = embeds.shape[1]

with open("/data/user_data/jingyuah/sparse_disk/doc_embed/marco_vectors.bin", "wb") as f:
    # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
    f.write(q_num.to_bytes(4, 'little')) # number of points
    f.write(q_dim.to_bytes(4, 'little')) # dimension
    f.write(embeds.tobytes()) # data array itself
    
