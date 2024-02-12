#!/bin/bash
#SBATCH --job-name=clueweb22corpus
#SBATCH --output=log_corpus/%x-%j.out
#SBATCH --error=log_corpus/%x-%j.err
#SBATCH --partition=babel-shared
#SBATCH --nodes=1

#SBATCH --mem=8G
#SBATCH --gres=gpu:0

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

# Your job commands go here
echo "Jingyuah Job Starts"

# for X in {13..20}; do python /home/jingyuah/clue_web/build_clueweb_tsv_corpus.py $X; done

# for X in {41..46}; do sbatch get_corpus.sh $X; done
python /home/jingyuah/clue_web/build_clueweb_tsv_corpus.py $1

echo "Jingyuah Job Ends"
