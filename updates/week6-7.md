### Oct 11, 2023 Updates

---

### List of Things Done / To-do
- [ ] SPANN 
	- [x] search algorithm is completed and tested to work correctly on smaller subset
    - [x] debug SPANN build time (use more CPU cores) thanks to Yuwei for spending a lot of time looking at the codes with me
    - [ ] configure different parameters to improve recall
        - [x] new set of parameter -> trivial improvement
        - [x] new set of searching parameters -> not working
    - [ ] addIndex() doesn't work and since we can build now, this is considered low priority and we will continue on this when we are not as busy
- [x] HNSW
- [ ] IVFPQFastScan (working on it)

### Issues
- I was experimenting with several of the parameters listed on SPTAG that's claimed to influence index quality and recall, but changing those doesn't improve on recall, depsite a double in index building time
- I test two other search parameters (maxCheck and searchpostinglistLimit) on small sets of queries (100 or 10000), the recall rate is not changing with respect to changes in these parameters. This behavior was also listed as an open issue on the repo. 
    - maxCheck: this is listed as what would influence lantency and recall
    - searchpostinglistLimit: this is not listed as a parameter that influence recall, but I figure searching in more posting list might improve recall rate since we are looking at a larger set of "relevant" postinglists
- I will try to seek for a more effective mean to tune the parameters for a better recall during fall break


### Result
Index | Recall@10    | Recall@90 | Recall@100 | Latency per Query (ms) on Recall@100 | build_time
| -------- | ------- | ------- | ------- | ------- | ------- |
| HNSW_flat (efConstruction=40)  |  0.9696515879664146   | 0.9315167445746816 | 0.9266628142125561 | 0.4480085316626545 | ~ 1 hr
| HNSW_flat (efConstruction=200) |  0.9969153316805072    |  0.9889119091316342   | 0.9876996788651724 | 1.0177786289823003 | ~ 4.5 hrs
| SPANN 1 (basic) |  0.6195512775902304 |  0.6424368672059064 | 0.6404532490683902 | 2.179295823958327 | ~ 3 hrs
| SPANN 2 | 0.6228918417540799 | 0.6445975887071487 | 0.6425851834080363 | 5.88858663674215 | ~ 5.5 hrs

### Index Specification for Final Version
- faiss_hnsw: 
    - Location: '/data/group_data/cx_group/ann_index/hnsw/faiss_hnsw_marco_ef200.bin'
    - Parameters: 
        - efConstruction: 200
        - efSearch: 256
- SPANN 1 (basic):
    - Location: '/data/group_data/cx_group/ann_index/SPANN/spann_index'
    - Parameters that should have impact on index quality and recall: 
        - TPTNumber = 32
        - TPTLeafSize: 2000
        - GraphNeighborhoodScale: 2.000000
        - CEF: 1000
	    - MaxCheckForRefineGraph: 8192; NeighborhoodSize: 32
	    - MaxCheck: 4096
- SPANN 2 (trial with other set of parameters): 
    - Location: '/data/group_data/cx_group/ann_index/SPANN//spann_2'
    - Parameters that should have impact on index quality and recall: 
        - TPTNumber = 128
        - TPTLeafSize: 4000
        - GraphNeighborhoodScale: 2.000000
        - CEF: 1000
	    - MaxCheckForRefineGraph: 8192; NeighborhoodSize: 64
	    - MaxCheck: 1632
            - also check MaxCheck = 200, searchPostingPageLimit = 40: exactly same result


