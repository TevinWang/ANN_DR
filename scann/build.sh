#!/bin/bash
#SBATCH --job-name=marco_scann
#SBATCH --output=outputs/marco_scann-%j.out
#SBATCH --error=outputs/marco_scann-%j.err
#SBATCH --partition=long
#SBATCH --nodelist=babel-3-32
#SBATCH --nodes=1

#SBATCH --mem=300G
#SBATCH --gres=gpu:0

#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:01:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

# Your job commands go here
echo "Jingyuah Job Starts"

eval "$(conda shell.bash hook)"
conda activate scann

echo "activated"

# make the folder
tgt_dir='/ANN/ann_index/marco_scann'
# rm -rf $tgt_dir

wait
mkdir $tgt_dir

python build.py --data /data/user_data/jingyuah/sparse_disk/doc_embed/marco_vectors.bin \
    --query /data/user_data/jingyuah/sparse_disk/doc_embed/marco_queries.bin \
    --gt /data/user_data/jingyuah/sparse_disk/doc_embed/gt.bin \
    --index_path $tgt_dir \
    --num_leaves 30000 --train_size 500000 \
    --num_leaves_to_search 20000 --K 100 --pre_reorder_num_neighbors 50000 \
    --search_only False

echo "Jingyuah Job Ends"
