#!/bin/bash

#SBATCH --job-name=index_anchordr

# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
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

# cd /home/jcoelho/structured-retrieval/AnchorDR/anchordr
cd /home/jingyuah/clue_web
export PYTHONPATH=.

#number of gpu
export WORLD_SIZE=1
#gpu node level id
export LOCAL_RANK=0
#gpu cluster level id
export RANK=0

export MASTER_ADDR=localhost


if [[ $1 == "1" ]]; then
    vals=("00" "01" "02" "03" "04" "05")
    ports=(24900 24901 24902 24903 24904 24905)
elif [[ $1 == "2" ]]; then
    vals=("06" "07" "08" "09" "10" "11")
    ports=(24906 24907 24908 24909 24910 24911)
elif [[ $1 == "3" ]]; then
    vals=("12" "13" "14" "15" "16" "17")
    ports=(24912 24913 24914 24915 24916 24917)
elif [[ $1 == "4" ]]; then
    vals=("18" "19" "20" "21" "22" "23")
    ports=(24918 24919 24920 24921 24922 24923)
elif [[ $1 == "5" ]]; then
    vals=("24" "25" "26" "27" "28" "29")
    ports=(24924 24925 24926 24927 24928 24929)
elif [[ $1 == "6" ]]; then
    vals=("30" "31" "32" "33" "34" "35")
    ports=(24930 24931 24932 24933 24934 24935)
elif [[ $1 == "7" ]]; then
    vals=("36" "37" "38" "39" "40" "41")
    ports=(24936 24937 24938 24939 24940 24941)
elif [[ $1 == "8" ]]; then
    vals=("42" "43" "44" "45" "46")
    ports=(24942 24943 24944 24945 24946)

else
    echo "Invalid parameter. Please provide a value from 1 to 8."
    exit 1
fi

model=$2



if [ "$model" == facebook/contriever ]; then 
    opts='--use_mean_pooler'
    save_folder=contriever
    viz_features=False
elif [ "$model" == Luyu/co-condenser-marco ]; then 
    opts=""
    save_folder=cocondenser
    viz_features=False
elif [ "$model" == yiqingx/AnchorDR ]; then 
    opts='--use_t5_decoder --use_converted'
    save_folder=anchordr
    viz_features=False
elif [ "$model" == /home/jcoelho/models/AnchorDR/anchordr-subsample-viz-features-v2-hn-epoch1 ]; then 
    opts='--use_t5_decoder --use_converted'
    save_folder=anchordr-subsample-viz-features-v2-hn-epoch1
    viz_features=True

fi


for i in "${!vals[@]}"; do
    val=en00${vals[i]}
    port=${ports[i]}

    # outdir="/data/jcoelho/clueweb22/full_en_B/index/faiss/$save_folder/$val"
    outdir="/data/group_data/cx_group/clueweb22b-corpus/$save_folder/$val"
    # path_to_corpus="/data/jcoelho/clueweb22/full_en_B/index/anchordr/full_corpus_$val.tsv"
    path_to_corpus="/data/group_data/cx_group/clueweb22b-corpus/corpus/full_corpus_$val.tsv"

    mkdir $outdir
    echo "$val"

    echo "#######"
    echo $model
    echo $opts
    echo $viz_features
    echo "#######"
    # /home/jingyuah/OpenMatch/src/openmatch/driver/build_index.py
    python -m torch.distributed.launch --nproc_per_node=1 --master_port $port --use_env \
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
done

wait

# cd /data/jcoelho/clueweb22/full_en_B/index/faiss/$save_folder
cd /data/group_data/cx_group/clueweb22b-corpus/$save_folder
mkdir full
for i in "${!vals[@]}"; do cd en00${vals[i]}/ && mv embeddings.corpus.rank.0 embeddings.corpus.rank.${vals[i]} && cd ..; done
for i in "${!vals[@]}"; do mv en00${vals[i]}/embeddings.corpus.rank.${vals[i]} full/; done