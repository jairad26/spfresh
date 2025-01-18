use crate::core::float::AdriannFloat;
use crate::distances::DistanceMetric;
use std::sync::Arc;

pub enum InitializationMethod {
    Random,
    KMeansPlusPlus,
}

pub struct ClusteringParams<F: AdriannFloat> {
    pub distance_metric: Arc<dyn DistanceMetric<F>>,
    pub initialization_method: InitializationMethod,
    pub desired_cluster_size: Option<usize>,
    pub initial_k: usize,
    pub rng_seed: Option<u64>,
}

pub struct Cluster {
    pub centroid_idx: Option<usize>, // Store index of centroid in this cluster. SPANN uses real vectors as centroids.
    pub points: Vec<usize>,          // Store indices of points in this cluster
    pub depth: usize,                // Track hierarchy depth
}

impl Cluster {
    // Create a new instance of Cluster
    pub fn new(centroid_idx: usize, points: Vec<usize>, depth: usize) -> Self {
        Self {
            centroid_idx: Some(centroid_idx),
            points,
            depth,
        }
    }
}
