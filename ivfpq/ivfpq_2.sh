#!/bin/bash
#SBATCH --job-name=ivfpq_2
#SBATCH --output=/home/jingyuah/benchmarks/faiss_ivfpq/%x-%j.out
#SBATCH --error=/home/jingyuah/benchmarks/faiss_ivfpq/%x-%j.err
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
conda activate jy

echo "activated"


metrics='l2'
nlist=6400 # SPANN -> 32 * 10 # base: 3200
q_nlist=3200
m=25 # base: 128

index_name="IVF${nlist},PQ${m},RFlat"

# index_name="IVF${nlist}(IVF${q_nlist},PQ${m}x4fs,RFlat)"
# index_name="IVF${nlist},PQ${m}x4fsr,Refine(SQfp16)"

echo $index_name

# python ivfpq_build.py --index_name $index_name --metrics $metrics --efConstruction 1200

# python ivfpq_inference.py --index_name $index_name --efSearch 600 --nprobes 1200

python refine_search.py --index_name $index_name --efSearch 400 --nprobes 1200



echo "Jingyuah Job Ends"