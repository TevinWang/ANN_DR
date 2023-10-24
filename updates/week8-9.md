### Oct 25, 2023 Updates

---

### List of Things Done / To-do
- [ ] SPANN on MSMARCO
	- [x] build index (augment vectors)
    - [x] configure different parameters to improve recall
        - refractor such that everything could be run with custom config files
        - [x] TPTNumber
        - [x] SearchPostingPageLimit and MaxCheck
        - [x] NeighborhoodSize
- [x] HNSW
- [ ] IVFPQFastScan 
- [ ] Clueweb22 with AnchorDR 
    - [ ] process Clueweb and build embeddings
    - [ ] build SPANN index
- [ ] SSD
    - [ ] Babel has SSD, but still need to check if the shared storage is mounted on SSD
- [ ] StreamingMSTuring-10M
    - [x] read <i>Generic Intent Representation in Web Search</i>


### Result
Index | Recall@10    | Recall@90 | Recall@100 | Latency per Query (ms) on Recall@100 | build_time
| -------- | ------- | ------- | ------- | ------- | ------- |
| HNSW_flat (efConstruction=40)  |  0.9696515879664146   | 0.9315167445746816 | 0.9266628142125561 | 0.4480085316626545 | ~ 1 hr
| HNSW_flat (efConstruction=200) |  0.9969153316805072    |  0.9889119091316342   | 0.9876996788651724 | 1.0177786289823003 | ~ 4.5 hrs
| SPANN 1 |  0.9734112884469726 |  0.9346213843908318 | 0.9306612626378824 | 9.764952108750552 | ~ 3 hrs
| SPANN 2 |  0.9749070960896582 | 0.9358161835834032 | 0.931822523487816 | 15.994731887360444 | ~ 8 hrs
| SPANN 3 |  0.9737669975886327 | 0.9349076130499028 | 0.9309300416062652 | 11.502816055096338 | ~ 6 hrs


### Index Specification for Final Version
- faiss_hnsw: 
    - Location: '/data/group_data/cx_group/ann_index/hnsw/faiss_hnsw_marco_ef200.bin'
    - Parameters: 
        - efConstruction: 200
        - efSearch: 256
- SPANN 1 :
    - Location: will move to group folder later
        - TPTNumber = 64
	    - MaxCheckForRefineGraph: 8192
        - NeighborhoodSize: 32
        - SearchPostingPageLimit: 96
	    - MaxCheck: 8192
- SPANN 2 : 
    - Location: will move to group folder later
        - TPTNumber = 64
	    - MaxCheckForRefineGraph: 16384
        - NeighborhoodSize: 32
        - SearchPostingPageLimit: 192
	    - MaxCheck: 16384
- SPANN 3: 
    - Location: will move to group folder later
        - TPTNumber = 64
	    - MaxCheckForRefineGraph: 8192
        - NeighborhoodSize: 64
        - SearchPostingPageLimit: 96
	    - MaxCheck: 8192


