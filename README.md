<p align="center"><img src="./adriANN.png" width="440"/></p>
<h1 align="center"> adriANN </h1>

**adriANN** is an **A**pproximate **N**earest **N**eighbors library in Rust, based on SPANN, a highly-efficient billion scale aproximate nearest neighbor search.💥

## SPANN
SPANN is a state-of-the-art hybrid vector indexing and search system designed to achieve high recall and low latency for large-scale datasets, while maintaining efficient memory and disk usage. Unlike prior hybrid ANNS (Approximate Nearest Neighbor Search) approaches such as DiskANN and HM-ANN, which rely on graph-based solutions, SPANN adopts a simpler inverted index methodology. 

### Key Features:
- **Memory-Efficient Design**: SPANN stores only the centroid points of posting lists in memory, significantly reducing memory requirements. This is important as most of the algorithms mainly focus on how to do low latency and high recall search all in memory with offline pre-built indexes. When targeting to the super large scale vector search scenarios, such as web search, the memory cost becomes extremely expensive.
- **Optimized Disk Access**: Large posting lists are stored on disk, but the system minimizes disk accesses by balancing and expanding these lists using a hierarchical balanced clustering strategy.
- **High Recall and Low Latency**: By enhancing the quality of posting lists and leveraging efficient indexing techniques, SPANN ensures state-of-the-art performance comparable to or better than existing solutions.

### How does it work?
1. Index Structure
The dataset is divided into groups called "posting lists" based on clusters of data points. Each group has a centroid, a kind of representative summary point for that group. These centroids are stored in memory for fast access, while the actual data points in the groups are stored on disk.

2. Search Process
When a query (like a question or search request) comes in, SPANN looks at the centroids in memory to quickly find the groups that are most relevant to the query. It then loads only those relevant groups from the disk into memory for a more detailed search to find the best matches.

3. Optimizations
SPANN keeps the size of each group (posting list) manageable so it can load them quickly from disk. It also improves the groups by including additional nearby points to cover "boundary" cases where the best matches might be split across different groups. During the search, SPANN dynamically decides how many groups to check based on the query's complexity, ensuring a good balance between speed and accuracy.



## References
- [SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search](https://arxiv.org/abs/2111.08566)
- [SPFresh: Incremental In-Place Update for Billion-Scale Vector Search](https://arxiv.org/abs/2410.14452)
- [Paper's repo](https://github.com/microsoft/SPTAG)