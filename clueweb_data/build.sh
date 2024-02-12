#!/bin/bash
#SBATCH --job-name=new_2
#SBATCH --output=disk_logs/%x-%j.out
#SBATCH --error=disk_logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --nodes=1

#SBATCH --nodelist=babel-3-32

#SBATCH --mem=200G
#SBATCH --gres=gpu:0

#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

# Your job commands go here
echo "Jingyuah Job Starts"

eval "$(conda shell.bash hook)"
conda activate diskAnn

echo "activated"

mkdir /ANN/ann_index/clueweb

mkdir /ANN/ann_index/clueweb/cw22-R360-L480-2

max=2
min=2

for i in `seq $min $max`
do 
    echo "------------------- $i ------------------------"
      /home/jingyuah/trial_disk/DiskANN/build/apps/build_disk_index --data_type float --dist_fn mips \
      --data_path /data/group_data/cx_group/ann_index/clueweb/doc_embed/vectors.bin \
      --index_path_prefix /ANN/ann_index/clueweb/cw22-R360-L480/clue_ \
            -R 360 -L 480 -B 150 -M 128 --shard_num $i
done 

echo "ls what's here"

echo "Jingyuah Job Ends"



# ############################## SPLIT ##########
# mkdir /ssd/ann_index/diskann_clueweb/cw22-R360-L480
# /home/jingyuah/diskANN/DiskANN/build/apps/build_disk_index --data_type float --dist_fn mips \
#         --data_path /home/jingyuah/embeddings/clueweb22_embed/doc_embed/vectors.bin \
#         --index_path_prefix /ssd/ann_index/diskann_clueweb/cw22-R360-L480/clue_ \
#         -R 360 -L 480 -B 150 -M 128


# ##################### single shard ###########
# max=0
# min=0

# for i in `seq $min $max`
# do 
#     echo "------------------- $i ------------------------"
#     /home/jingyuah/diskANN/DiskANN/build/apps/build_disk_index --data_type float --dist_fn mips \
#         --data_path /home/jingyuah/embeddings/clueweb22_embed/doc_embed/vectors.bin \
#         --index_path_prefix /home/jingyuah/clueweb/cw22-R360-L480/clue_ \
#         -R 360 -L 480 -B 150 -M 128 --shard_num $i
# done      
# #################################################


################# merge ########
# /home/jingyuah/diskANN/DiskANN/build/apps/build_disk_index --data_type float --dist_fn mips \
#         --data_path /home/jingyuah/embeddings/clueweb22_embed/doc_embed/vectors.bin \
#         --index_path_prefix /ssd/ann_index/diskann_clueweb/cw22-R360-L480/clue_ \
#         -R 360 -L 480 -B 150 -M 128 --num_parts 9

# #######################################

########################################
# # anchor queries 
# mkdir /ssd/ann_index/diskann_clueweb/cw22-R360-L480/K100

# echo "--------700 - 1000--------"
# /home/jingyuah/diskANN/DiskANN/build/apps/search_disk_index --data_type float --dist_fn mips \
#     --index_path_prefix /ssd/ann_index/diskann_clueweb/cw22-R360-L480/clue_ \
#     --query_file /home/jingyuah/embeddings/clueweb22_embed/doc_embed/queries_anchor_text.bin \
#     --gt_file /home/jingyuah/embeddings/clueweb22_embed/doc_embed/gt_anchor_text.bin -K 100 -L 700 800 900 1000 \
#     --result_path /ssd/ann_index/diskann_clueweb/cw22-R360-L480/K100/res_ --num_nodes_to_cache 10000