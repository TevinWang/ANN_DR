### Sep 26, 2023 Updates

---

### List of Things Done / To-do
- [ ] SPANN recall and MRR, compare with HNSW, FAISS, scaNN
	- [x] search algorithm is completed and tested to work correctly on smaller subset
    - [ ] search on the whole MSMARCO was unsuccessful as I discovered that the build index was partially deprecated -> fixing ongoing
- [ ] Compare with HNSW, FAISS, scaNN
	- [ ] experimenting with HNSW, building index ongoing 

### Issue about SPANN
I was building the indexes by batches last week but I realize that some of the batches was not successfully built as I try to perform search this week. 

In this case, I am trying to build the indices all at once but this is very slow. I also update some of the parameters for index building to suit the larger dataset. 

Here's some of my experiments: 

- building the whole set didn't output anything in 6 hrs (128GB mem, 2 cpu, 2 node)
- building the first 1000000 embeddings (ongoing trial) create output folder and some index files in 1 hour (32GB mem, 1cpu, 1 node), ongoing
- building the first 100000 embeddings took 4-6 minutes, successful (32GB mem, 1cpu, 1 node)
- building the first 10000 embeddings took less than 1 minute, successful (32GB mem, 1cpu, 1 node)

According to the SPANN paper, SIFT1M only took 2 minutes to build using 2 cpus and 128 GB memory. I figure there's a problem but I am not sure what's the exact issue here since smaller subsets could be built correctly. Should I ask Luoqi or Yuwei to look at my code to check? (this might take a lot of time to do so I want to check with you first)

---


### Paper Reading and Discussion

#### NCI (Neural Corpus Indexer):
- End-to-end differentiable document retrieval model that outputs the most relevant document identifier given query
- Method:
    - Document Encoding: hierarchical k-mean
	    - Cluster limit: c
		- Concatenated document semantic identifier by the node indices along the path to the leaf node 
	- Query Generation making identifiers: 
		- DocT5Query
		- Document As Query
	- Prefix-Aware Weight Adaptive decoder for document identifiers
		- Parameters is not shared across tree levels
		- Prefix-aware adaptive weights for each token classifier
	- Consistency-based regularization loss + seq2seq cross-entropy loss with teacher-forcing
- Seq2swq NN that could to optimized end-to-end
- Encoder and decoder doesn't share the same token spaces
- Token at different tree level/have same location but different prefix encode different semantics



