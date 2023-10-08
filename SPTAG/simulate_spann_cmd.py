import time
import subprocess
import numpy as np
import os



f = open("simulated_spann.txt", "w")

f.write("-------------Prepare input vectors and meta data---------------\n")
vector_number = 100
vector_dimension = 10

# Randomly generate the database vectors. Currently SPTAG only support int8, int16 and float32 data type.
embed = np.random.rand(vector_number, vector_dimension).astype(np.float32) 
np.save("random_vectors.npy", embed)

# Prepare metadata for each vectors, separate them by '\n'. Currently SPTAG python wrapper only support '\n' as the separator
m = ''
for i in range(vector_number):
    m += str(i) + '\n'

f.write("### Vector Number: " + str(vector_number) + "; Vector Dimension: " + str(vector_dimension) + " ###\n")
f.write("Confirm Shape: " + str(np.shape(embed)) + '\n')


f.write("-------------Build Embedding SPANN---------------\n")

start = time.time()
# command to build SPANN index offline from file 
# -i raw input data
args_cmd = ['/home/jingyuah/SPTAG/Release/indexbuilder', '-c', '/home/jingyuah/SPTAG/Release/buildconfig.ini', 
            '-d', '128', '-v', 'Float', '-f', 'DEFAULT', '-i', "random_vectors.npy", '-o', 'ANCE_T5_10', '-a', 'SPANN']
result = subprocess.run(args_cmd, stdout = subprocess.DEVNULL)
# result = subprocess.Popen(args_cmd)

end = time.time()
print (result) 
f.write("Time Taken: " + str(end-start) + "\n")

f.close()


