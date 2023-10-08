import subprocess
import numpy as np


vector_dim = 100                         # dimension
vector_num = 1000                      # dataset size
nq = 1                     # nb of queries
np.random.seed(1234)             # make reproducible

randomData = np.random.random((vector_num, vector_dim)).astype('float32')
randomData.shape

meta = [i for i in range(randomData.shape[0])]
print(meta[:5])

# np.save('embed.npy', randomData)
# embed = np.load('embed.npy')
# print(np.shape(embed))

# save as desired txt format

with open("randomVector.bin", "wb") as binary_file:
    # Write bytes to file
    binary_file.write(vector_num.to_bytes(4, 'big'))
    binary_file.write(vector_dim.to_bytes(4, 'big'))
    binary_file.write(randomData.tobytes())


# command to build SPANN index offline from file 
# -i raw input data
# args_cmd = ['/home/jingyuah/SPTAG/Release/indexbuilder', '-c', '/home/jingyuah/SPTAG/Release/buildconfig.ini', 
#             '-d', '128', '-v', 'Float', '-f', 'DEFAULT', '-i', "randomVector.bin", '-o', '/home/jingyuah/ANCE_T5_10', '-a', 'SPANN']
# result = subprocess.run(args_cmd, stdout = subprocess.DEVNULL)
# result = subprocess.Popen(args_cmd)

# -qd: vector_dim/2
# /home/jingyuah/SPTAG/Release/quantizer -d 100 -v Float -f DEFAULT -i randomVector.bin -o quan_doc_vectors.bin -oq q_randomVector.bin -qt PQQuantizer -qd 50 -ts 1000 -norm false
# /home/jingyuah/SPTAG/Release/indexbuilder -c /home/jingyuah/experiment/buildconfig.ini -d 100 -v Float -f DEFAULT -i randomVector.bin -o ANCE_T5_10 -a SPANN 
# /home/jingyuah/SPTAG/Release/ssdserving  -c /home/jingyuah/experiment/buildconfig.ini -pq quan_doc_vectors.bin

# print (result) 