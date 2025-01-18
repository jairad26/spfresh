/// adriANN: An Approximate Nearest Neighbors library in Rust
///
/// This library is based on SPANN, a highly-efficient billion-scale approximate nearest neighbor search.
///
/// # Modules
/// - `clustering`: Contains the implementation of the hierarchical clustering algorithm used in SPANN.
/// - `core`: Contains the core float data type.
/// - `distances`: Contains the implementation of the distance metrics used everywhere in the project.
/// - `spann`: Contains the implementation of the SPANN index.
pub mod clustering;
pub mod core;
pub mod distances;
pub mod spann;
