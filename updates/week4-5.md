### Sep 26, 2023 Updates




### #To-dos / things 

- [x] [Sep 23] Simulate SPANN 1,000,000 random vector
- [x] [Sep 26] Shared data directory for embedding sharing
- [ ] Build index for T5-ANCE (ongoing, just got the embeddings)


---


### Paper Reading 

#### Distill-VQ:
- Jointly learn IVF and PQ
- Trainable parameters: IVF centroids, PQ codebooks, query encoder
- Fixed document embeddings
- Doesn’t rely on labeled data
- Method: 
    - Well-trained document encoder: fixed document embedding
    - Quantization by IVFPQ
    - Knowledge distillation: let reconstruct embedding imitate dense embedding
        - Teachers: document and query embeddings -> predict query's relevance to sample documents
        - Students: reconstructed embeddings -> reproduce the predicted relevance
- Similarity Function: 
    - ListNet: preserve ranking order
        - Variant of KL-divergence
        -  Entropy of normalized predicted scores towards the candidate documents by teacher and student
- Candidate Sampling:
    - Top-K relevant
    - In-Batch: other queries' related documents


    ##### My thoughts: 
    - Improve retrieval quality but that advantage would get trivial when codebook is very large
    - Dependent on a good pre-trained document encoder 
    - In-batch negatives might be trivial due to batch size and uninformative samples as argued in ANCE paper 
        - ANCE uses asychronuous index update 


#### MoMA: (suggested by Yuwei) 
- Trains the augmentation component with latent labels derived from the end retrieval task, paired with hard negatives from the memory mixture
- Option to include target tasks as in-domain knowledge
- Jointly learn the main model and the augmentation retriever component
- Method: 
    - Construct hard negatives of source task with main model weights of timestep (t-1)
    - Retrieve the augmented documents with augmentation retriever weights of timestep (t-1)
    - Train main model (ANCE) with source task documents
    - Use updated attention score to form positive augmented documents, and retrieve hard negatives augmented documents with augmentation retriever weights of timestep (t-1)
    - Train the augmentation retriever (ANCE)

    ##### My thoughts: 
    - Intuitively this should improve model performance as it includes task domain when introducing the augmented documents, but I think this is not the current focus of us. It might be better if we could directly tackle the two problem discussed last week considering mixtures of memories?
        - Embeddings are very important. 
        - When we introduce the large corpus for augmentation, we have a larger dataset for the ANN index to hold and might result in the case where we have to make more duplicates for boundary vectors.  

