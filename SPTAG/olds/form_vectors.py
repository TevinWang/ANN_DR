import numpy as np
import fileinput
import pickle
import os


def augment_q(queries):
    extracol = np.zeros(len(queries), dtype='float32') # zero queries
    return np.hstack((queries, extracol.reshape(-1, 1)))   

def augment_train(data): # tranform the data to 769 dimensional
    norms = (data ** 2).sum(axis=1) # point-wise sq; sum by row
    phi = norms.max() 
    extracol = np.sqrt(phi - norms)
    return np.hstack((data, extracol.reshape(-1, 1))) # vertically stack at the end


# print("-------- retrieved ground truth documents ---------------")
# # retrieved documents of T5 ANCE
# rname = '/data/group_data/cx_group/efficient-dr/retrieved.train.trec'
# retrieved = {}
# for line in fileinput.input([rname]):
#     ids = line.split(' ')
#     if ids[0] not in retrieved.keys():
#         retrieved[ids[0]] = [[int(ids[2]), int(ids[3])]] # q_id: [d_id, rank]
#     else:
#         retrieved[ids[0]].append([int(ids[2]), int(ids[3])])
# # need to exclude those with less than 100 documents when doing recall
# queries_id = retrieved.keys() 


# print("-------- parsing queries---------------")
# # parse queries
# name = '/data/group_data/cx_group/efficient-dr/embeddings/embeddings.query.rank.'
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


# print("-----------------transform queries------------------")
# queries = augment_q(queries)
# print("Updated Queries Shape: ", queries.shape)

# dim = queries.shape[1]
# q_num = queries.shape[0]


print("--------------set path-------------------")
sys_path = '/data/user_data/jingyuah/data_txt'
query_path = os.path.join(sys_path, 'queries.tsv')
truth_path = os.path.join(sys_path, 'truths.tsv')
vector_path = os.path.join(sys_path, 'vectors.tsv')


# print("---------------WRITE QUERIES--------------------")
# with open(query_path, 'w') as f: # <metadata1>\t<v11>|<v12>|<v13>|
#     for i in range(queries.shape[0]):
#         s = str(i) + '\t'
#         for d in range(queries.shape[1]):
#             s += str(queries[i][d]) + '|'
#         s += '\n'
#         f.write(s)


# print("---------------WRITE TRUTH--------------------")
# truth = np.array(retrieved)
# k = 100
# with open(truth_path, 'w') as f: # <t11> <t12>
#     for q_id in q_ids: # same order as queries
#         truth = np.array(retrieved[q_id])[:k, 0] # gt document ids
#         for d in range(truth.shape[0]): 
#             s = str(truth[d]) + ' '
#         if (truth.shape[0] < k):
#             for d in range(truth.shape[0], k):
#                 s = str(-1) + ' '
#         s += '\n'
#         f.write(s)        


print("-----------------DOCUMENTS LOADED-------------------")

# load input data
fname = '/data/group_data/cx_group/efficient-dr/embeddings/t5-ance-ms.pickle'
with open(fname, "rb") as f:
    encs = pickle.load(f)

vector_num  = encs[0].shape[0] # 1000000 # encs[0].shape[0] # 8 841 823
vector_dim = encs[0].shape[1] # 768

embeds = encs[0] #(8841823, 768)
meta = encs[1]
print("Embedding Shape: ", embeds.shape) # embeddings
print("Length of Metadata: ", len(meta)) # list of id's = index of embeds

embeds = augment_train(embeds)


print("---------------WRITE DOCUMENTS--------------------")
with open(vector_path, 'w') as f: # <metadata1>\t<v11>|<v12>|<v13>|
    for i in range(vector_num):
        s = str(meta[i])+ '\t'
        for d in range(vector_dim):
            s += str(embeds[i][d]) + '|'
        s += '\n'
        f.write(s)

print("------------ends-----------------")