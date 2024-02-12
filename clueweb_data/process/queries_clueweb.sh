#!/bin/bash

#SBATCH --job-name=anchordr_q

# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/anchordr_q_%j.out
#SBATCH --error=logs/anchordr_q_%j.err
#SBATCH --partition=babel-shared

#SBATCH --nodes 1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 12 # number cpus (threads) per task


# 327680
#SBATCH --mem=48G # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=24:00:00 # No time limit

# You can also change the number of requested GPUs
# replace the XXX with nvidia_a100-pcie-40gb or nvidia_a100-sxm4-40gb
# replace the YYY with the number of GPUs that you need, 1 to 8 PCIe or 1 to 4 SXM4

#SBATCH --gres=gpu:A6000:1

# run like this:
# for X in {1..8}; do sbatch anchor_clueweb.sh $X yiqingx/AnchorDR; done
# or
# for X in {1..8}; do sbatch index_clueweb.sh $X Luyu/co-condenser-marco; done
# or
# for X in {1..8}; do sbatch index_clueweb.sh $X facebook/contriever; done
# or
# for X in {1..8}; do sbatch index_clueweb.sh $X /home/jcoelho/models/AnchorDR/anchordr-subsample-viz-features-v2-hn-epoch2; done
# or

echo "started"

# activate your environment
eval "$(conda shell.bash hook)"
conda activate clueweb

cd /home/jingyuah/clue_web
export PYTHONPATH=.

#number of gpu
export WORLD_SIZE=1
#gpu node level id
export LOCAL_RANK=0
#gpu cluster level id
export RANK=0

export MASTER_ADDR=localhost

model="yiqingx/AnchorDR"
opts='--use_t5_decoder --use_converted'
viz_features=False

outdir="/data/user_data/jingyuah/embed"
# "/data/group_data/cx_group/clueweb22b-corpus/anchordr/queries" 
path_to_corpus="/data/datasets/clueweb22/AnchorDR_data_release/corpus.tsv" # anchor_id, anchor_text 


# queries_small.rank.0 -> encs[0]: (25967, 768); encs[1]:anchor_id 
# eval_corpus_small.rank.0 -> encs[0]: (18398, 768); encs[1]: doc_id

# mkdir $outdir

echo "#######"
echo $model
echo $opts
echo $viz_features
echo "#######"
# /home/jingyuah/OpenMatch/src/openmatch/driver/build_index.py
python -m torch.distributed.launch --nproc_per_node=1 --use_env \
    /home/jingyuah/AnchorDR/lib/openmatch/driver/build_index.py \
    --output_dir $outdir \
    --model_name_or_path $model $opts \
    --per_device_eval_batch_size 600 \
    --corpus_path $path_to_corpus \
    --doc_template "Title: <title> Text: <text>" \
    --doc_column_names id,title,text \
    --q_max_len 32 \
    --p_max_len 128 \
    --fp16 \
    --dataloader_num_workers 1 \
    # --use_viz_features $viz_features \
    2>&1 &



# # cd /data/jcoelho/clueweb22/full_en_B/index/faiss/$save_folder
# cd /data/group_data/cx_group/clueweb22b-corpus/$save_folder
# mkdir full
# for i in "${!vals[@]}"; do cd en00${vals[i]}/ && mv embeddings.corpus.rank.0 embeddings.corpus.rank.${vals[i]} && cd ..; done
# for i in "${!vals[@]}"; do mv en00${vals[i]}/embeddings.corpus.rank.${vals[i]} full/; done