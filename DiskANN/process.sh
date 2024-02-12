#!/bin/bash
#SBATCH --job-name=process_para
#SBATCH --output=outputs/llama/%x-%j.out
#SBATCH --error=outputs/llama/%x-%j.err
#SBATCH --partition=general
#SBATCH --nodes=1

#SBATCH --mem=600G

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

# Your job commands go here
echo "Jingyuah Job Starts"

eval "$(conda shell.bash hook)"
conda activate diskAnn

echo "activated"


# python process_rerank_doc_d_2.py

python process_sparse_doc_rerank.py

# python llama/sparse_doc_rerank.py /data/user_data/yuweia/doc_llama_seperate

# python llama/process_old.py

# python process_sparse_q.py

# # compute gt
# /home/jingyuah/DiskANN/build/apps/utils/compute_groundtruth  --data_type float --dist_fn mips \
#     --base_file /data/user_data/jingyuah/sparse_disk/llama/vectors_0115.bin \
#     --query_file /data/user_data/jingyuah/sparse_disk/llama/queries_full.bin \
#     --gt_file /data/user_data/jingyuah/sparse_disk/llama/gt_full.bin --K 100


echo "Jingyuah Job Ends"
