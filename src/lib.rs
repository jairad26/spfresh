/// adriANN: An Approximate Nearest Neighbors library in Rust
///
/// This library is based on SPANN, a highly-efficient billion-scale approximate nearest neighbor search.
///
/// # Modules
/// - `clustering`: Contains the hierarchical clustering implementation.
/// - `visualization`: Provides tools for visualizing the clustering results.

pub mod clustering;
pub mod visualization;

// Re-export key components for easier access
pub use clustering::{Cluster, HierarchicalClustering};