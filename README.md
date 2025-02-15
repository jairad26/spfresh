<h1 align="center"> SPFresh </h1>

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](README.md)

**SPFresh** is an **A**pproximate **N**earest **N**eighbors library in Rust, based on [SPANN: Highly-efficient Billion scale Approximate Nearest Neighbor Search]((https://arxiv.org/abs/2111.08566)). It aims to be:
- **Memory-Efficient Design**: SPANN stores only the centroid points of posting lists in memory, significantly reducing memory requirements. This is important as most of the algorithms mainly focus on how to do low latency and high recall search all in memory with offline pre-built indexes. When targeting to the super large scale vector search scenarios, such as web search, the memory cost becomes extremely expensive.
- **Optimized Disk Access**: Large posting lists are stored on disk, but the system minimizes disk accesses by balancing and expanding these lists using a hierarchical balanced clustering strategy.  
- **High Recall, Low Latency**: Leverages hierarchical balanced clustering strategies to achieve lightning-fast lookups with exceptional accuracy. The in-memory index is based on [Kiddo](https://github.com/sdd/kiddo/tree/master), a high-performance, k-d tree library.  

## Usage Examples

Building a Spann index:
```rust
use ndarray::{Array1, Array2};
use spann::spann::config::Config;
use spann::spann::spann_builder::SpannIndexBuilder;

fn main() {
    // 1. Load configuration from file
    // example_config.yaml
    // 
    // clustering_params:
    //   distance_metric: "Euclidean"
    //   initialization_method: "Random"
    //   initial_k: 4
    // output_path: "data"

    let config: Config =
        Config::from_file("examples/example_config.yaml").expect("Failed to load configuration");

    // 2. Prepare your dataset
    let data = Array2::from_shape_vec(
        (6, 2), 
        vec![
            1.0, 2.0,
            1.5, 2.5,
            8.0, 8.0,
            8.5, 8.5,
            4.0, 4.0,
            4.5, 4.5,
        ]
    ).unwrap();

    // 3. Build the index using a chosen dimension (here: 2)
    let spann_index = SpannIndexBuilder::<f32>::new(config)
        .with_data(data.view())
        .build::<2>()
        .expect("Failed to build SPANN index");

    println!("SPANN index built successfully!");
}
```

Querying the Spann index:
```rust
use ndarray::Array1;
use spann::spann::config::Config;
use spann::spann::spann_builder::SpannIndexBuilder;

fn main() {
    // 1. Load configuration
    let config: Config =
        Config::from_file("examples/example_config.yaml").expect("Failed to load configuration");

    // 2. Load the index from previously built data
    let spann_index = SpannIndexBuilder::<f32>::new(config)
        .load::<2>()
        .expect("Failed to build SPANN index");

    // 3. Query your vector
    let k = 1; // number of nearest neighbors
    let query_vector = Array1::from(vec![1.0, 2.0]);
    let result = spann_index
        .find_k_nearest_neighbor_spann(&query_vector.view(), k)
        .unwrap();

    // 4. Print result
    println!("Nearest neighbour: point_id: {:?} and vector: {:?}",
             result[0].point_id,
             result[0].vector);
}
```
## Configuration
A typical `config.yaml` might look like this:

```yaml
clustering_params:
# Supported values for distance_metric: "Euclidean", "Manhattan", "Chebyshev"
distance_metric: "Euclidean"
# Supported values for initialization_method: "Random", "KMeans++"
initialization_method: "Random"
# 'initial_k' is the initial number of clusters
initial_k: 4

# Path where index files are stored
output_path: "data"
```

- `distance_metric`: Controls how vectors are compared. Choose what's best for your data type. Currently supported: `Euclidean`, `Manhattan`, `Chebyshev`.
- `initial_k`: The number of cluster centroids to start with. Tweak this for controlling the granularity of your clusters.
- `initialization_method`: How centroids are initialized (`Random` or `KMeans++`).
- `output_path`: Where the index and posting lists get stored.

### How does it work?
**- Index Structure**

The dataset is divided into groups called "posting lists" based on clusters of data points. Each group has a centroid, a kind of representative summary point for that group. These centroids are stored inside an in-memory index for fast access, while the actual data points in the groups are stored on disk.

**- Search Process**

When a query (like a question or search request) comes in, SPANN looks at the centroids in memory to quickly find the groups that are most relevant to the query. It then loads only those relevant groups from the disk into memory for a more detailed search to find the best matches.

**- Optimizations**

SPANN keeps the size of each group (posting list) manageable so it can load them quickly from disk. It also improves the groups by including additional nearby points to cover "boundary" cases where the best matches might be split across different groups. During the search, SPANN dynamically decides how many groups to check, ensuring a good balance between speed and accuracy.


## References
- [SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search](https://arxiv.org/abs/2111.08566)
- [SPFresh: Incremental In-Place Update for Billion-Scale Vector Search](https://arxiv.org/abs/2410.14452)
- [Paper's repo](https://github.com/microsoft/SPTAG)

<hr /> <div align="center"> <em> Made with ❤ and posting lists. <br/> We <strong>welcome contributions</strong>: issues, PRs, discussions. </em> </div>
