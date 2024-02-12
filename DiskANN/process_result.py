import sys
import pickle
import numpy as np
import time
import os
import struct
        
        
    
# print("-----------------q02 retrieved-------------------")
# sys.stdout.flush()
# # load gt IDs
# name = '/home/jingyuah/marco/disk_marco/q_02/marco_result__200_idx_uint32.bin'
# # '/home/jingyuah/marco/disk_marco/q_00/marco_result__200_idx_uint32.bin'

# # read the meta data
# with open(name, "rb") as f:
#     # read the meta data
#     nrows = int.from_bytes(f.read(4), "little")
#     ncols = int.from_bytes(f.read(4), "little")
    
#     print("nrows: ", nrows)
#     print("ncols: ", ncols)
    
#     retrieved = []
#     for i in range(nrows):
#         curr_r = np.full(100, -1) # init space for the retrieved docs
#         for j in range(100):
#             curr_r[j] = int.from_bytes(f.read(4), "little")
#         retrieved.append(curr_r)    

# retrieved = np.array(retrieved)
# print("retreived shape: ", retrieved.shape)
# print("the first five id for the first query: ", retrieved[0][0:10])


# print("-----------------q00 retrieved-------------------")
# sys.stdout.flush()
# # load gt IDs
# name = '/home/jingyuah/marco/disk_marco/q_00/marco_result__200_idx_uint32.bin'

# # read the meta data
# with open(name, "rb") as f:
#     # read the meta data
#     nrows = int.from_bytes(f.read(4), "little")
#     ncols = int.from_bytes(f.read(4), "little")
    
#     print("nrows: ", nrows)
#     print("ncols: ", ncols)
    
#     retrieved = []
#     for i in range(nrows):
#         curr_r = np.full(100, -1) # init space for the retrieved docs
#         for j in range(100):
#             curr_r[j] = int.from_bytes(f.read(4), "little")
#         retrieved.append(curr_r)    

# retrieved = np.array(retrieved)
# print("retreived shape: ", retrieved.shape)
# print("the first five id for the first query: ", retrieved[0][0:10])


print("-----------------custom gt from q00-------------------")
sys.stdout.flush()
# load gt IDs
name = '/home/jingyuah/embeddings/efficient-dr/sparse/doc_embed/gt_by_00.bin'

# read the meta data
with open(name, "rb") as f:
    # read the meta data
    nrows = int.from_bytes(f.read(4), "little")
    ncols = int.from_bytes(f.read(4), "little")
    
    print("nrows: ", nrows)
    print("ncols: ", ncols)
    
    retrieved = []
    for i in range(nrows):
        curr_r = np.full(100, -1) # init space for the retrieved docs
        for j in range(100):
            curr_r[j] = int.from_bytes(f.read(4), "little")
        retrieved.append(curr_r)    



# # load distances
# name = '/home/jingyuah/marco/disk_marco/q_00/marco_result__200_dists_float.bin'

# # read the meta data
# with open(name, "rb") as f:
#     # read the meta data
#     nrows = int.from_bytes(f.read(4), "little")
#     ncols = int.from_bytes(f.read(4), "little")
    
#     print("nrows: ", nrows)
#     print("ncols: ", ncols)
    
#     dists = []
#     for i in range(nrows):
#         curr_r = np.empty(shape=(100)) # init space for the retrieved docs
#         for j in range(100):
#             byte = f.read(4)
#             curr_r[j] = struct.unpack('f', byte)[0]
#         dists.append(curr_r)    
        

# dists = np.array(dists)
# print("dists shape: ", dists.shape)
# print("the first five dists for the first query: ", dists[0][0:5])


# # dump the gt IDs to array
# print("------ready to dump------------")

# offset = 0
# print("offset: ", offset)


# # do not write dists!
# with open("/home/jingyuah/embeddings/efficient-dr/sparse/doc_embed/gt_by_00.bin", "wb") as f:
    
#     # header
#     f.write(nrows.to_bytes(4, 'little')) # number of points
#     f.write(ncols.to_bytes(4, 'little')) # dimension
#     offset += 8
    
#     # ids
#     f.seek(offset)
#     f.write(retrieved.tobytes()) # data array itself
#     offset += retrieved.nbytes

#     # # ids
#     # f.seek(offset)
#     # for i in range(nrows):
#     #     for j in range(ncols): 
#     #         f.write(struct.pack('<f', dists[i][j])) # little-endian)
    
    
# with open("/home/jingyuah/embeddings/efficient-dr/sparse/doc_embed/gt_by_00.bin", "rb") as f:
    
#     print(int.from_bytes(f.read(4), "little"))
#     print(int.from_bytes(f.read(4), "little"))
    
#     f.seek(8)
#     print('first id: ', int.from_bytes(f.read(4), "little"))
    
#     # f.seek(offset)
#     # b = f.read(4)
#     # print('first dist: ', struct.unpack('f', b)[0])

    