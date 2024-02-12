#!/bin/bash
#SBATCH --job-name=clue_1m_s
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1

#SBATCH --mem=64G
#SBATCH --gres=gpu:0

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"



echo "Jingyuah Job Starts"

eval "$(conda shell.bash hook)"
conda activate faiss

echo "activated"


metrics='l2'
nlist=6400 # SPANN -> 32 * 10 # base: 3200
q_nlist=1200
m=256 # base: 128ik9
refine_frac=10

index_name="IVF${nlist},PQ${m}"

dest_dir="/home/jingyuah/clueweb/ivfpq"
embed_path="/home/jingyuah/embeddings/clueweb22_embed/doc_embed/vectors_1m.bin"
queries_path="/home/jingyuah/embeddings/clueweb22_embed/doc_embed/queries_anchor_text.bin"
gt_path="/home/jingyuah/embeddings/clueweb22_embed/doc_embed/gt_anchor_text_1m.bin"
stats_path="/home/jingyuah/clue_web/ivfpq/outputs/$index_name.txt"

echo $index_name
echo $dest_dir

python ivfpq_build_small.py --index_name $index_name \
    --dest_dir $dest_dir --embed_path $embed_path \
    --metrics $metrics --efConstruction 4800


python refine_search.py --index_name $index_name \
    --index_dir $dest_dir --embed_path $embed_path \
    --query_path $queries_path --gt_path $gt_path --stats_path $stats_path \
    --efSearch 1000 --nprobes 4800 --refine_frac $refine_frac


echo "Jingyuah Job Ends"