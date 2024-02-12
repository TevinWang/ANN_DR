import sys
import pickle
import numpy as np
import time
import os
import fileinput

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



print("-----------------DOCUMENTS LOADED-------------------")
sys.stdout.flush()    
t1 = time.time()
# load input data
fname = '/data/group_data/cx_group/efficient-dr/embeddings/t5-ance-ms.pickle'
with open(fname, "rb") as f:
    encs = pickle.load(f)

vector_num  = encs[0].shape[0] # 1000000 # encs[0].shape[0] # 8 841 823
vector_dim = encs[0].shape[1] # 768

# embeds = encs[0] #(8841823, 768)
q_ids = encs[1]

# map to int
q_ids = list(map(int, q_ids))     
# q_ids = np.array(q_ids)
# print("original id shape ", q_ids.shape)
print("ids sample: ", q_ids[:10])
print("ids shape: ", len(q_ids))

sys.stdout.flush()

# #####################################
# parse new embeddings
dir_name = '/data/user_data/yuweia/doc_llama_parallel'
print("processing: ", dir_name)
name_arr = os.listdir(dir_name)
# reshape myself version
# rerank


print("new q")
sys.stdout.flush()
first = True
for fname in name_arr:
    with open(os.path.join(dir_name, fname), "rb") as fTruth:
        encs = pickle.load(fTruth)
    if first:
        first = False
        new_ids = encs[0].astype(int)
        embed = encs[1]
    else:
        embed = np.concatenate((embed, encs[1]))
        new_ids = np.concatenate((new_ids, encs[0].astype(int)))
        
q_num = embed.shape[0]
q_dim = embed.shape[1]
# print("shape embed: ", embed.shape)
print("new ids sample: ", new_ids[:10])
print("new ids shape: ", new_ids.shape) # stay as np array -> for rerank
sys.stdout.flush()


# #####################################
# rerank
indices = np.argsort(new_ids) # original id is ranked from 0
rerank_embed = embed[indices]
rerank_q = new_ids[indices]

print("rerank q sample: ", rerank_q[:10])

#check
# print("shape of rerank: ", rerank_embed.shape)
print("all rerank q_ids matches? ", np.sum(rerank_q == q_ids)/len(q_ids))
sys.stdout.flush()


# #####################################
# parse new embeddings
old_name = '/data/user_data/jingyuah/sparse_disk/llama/vectors_0115.bin'
old_embed = read_fbin(old_name)
print("old embed shape: ", old_embed.shape)

# ##### check nan -> replaced with unpruned 
print("------------------------------------------")
nan_id = []
for i in range(q_num):
    nan_c = np.count_nonzero(np.isnan(rerank_embed[i, :]))
    if nan_c != 0: #
        nan_id.append(rerank_q[i])
        rerank_embed[i, :] = old_embed[i,:] # both are argsorted -> update to the gold original embed
        print("i: ", i, "has: ", nan_c, "nan entry") #
        print("now has this many nan entry: ", str(np.count_nonzero(np.isnan(rerank_embed[i, :]))))
print("how many are nan? ", len(nan_id))


# write bin file 
with open("/data/user_data/jingyuah/sparse_disk/llama/vectors_para_0123.bin", "wb") as f:
    # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
    f.write(q_num.to_bytes(4, 'little')) # number of points
    f.write(q_dim.to_bytes(4, 'little')) # dimension
    f.write(rerank_embed.tobytes()) # data array itself
    
# dump the nan ids 
with open("/data/user_data/jingyuah/sparse_disk/llama/nan_ids_para_0123.bin", "wb") as f:
    pickle.dump(nan_id, f)




# # ############
# # got sample 10% passage embeddings
# with open('/home/jingyuah/sparsity_index/llama/psg_ids_new.pkl', 'rb') as f:
#     sample_indices = pickle.load(f)
    
# sampled_embed = rerank_embed[sample_indices]
# print("sample embed shape: ", sampled_embed.shape)
# sample_q_num = sampled_embed.shape[0]
# sample_q_dim = sampled_embed.shape[1]

# # ##### check nan
# map = {}
# for i in range(sample_q_num):
#     nan_c = np.count_nonzero(np.isnan(sampled_embed[i, :]))
#     if nan_c != 0: #
#         print("i: ", nan_c) #
#         map[i] = nan_c
# print("how many are nan? ", len(map))



# print("written first...")
# full_output_name = '/data/user_data/jingyuah/sparse_disk/llama/vector_64_0122_0.1.bin' 
# with open(full_output_name, "wb") as f:
#     # bytes(500) is not encoding 500 but creates a bytestring w/ len == 500 -> use bytes([500])
#     f.write(sample_q_num.to_bytes(4, 'little')) # number of points
#     f.write(sample_q_dim.to_bytes(4, 'little')) # dimension
#     f.write(sampled_embed.tobytes()) # data array itself