use crate::clustering::config::Config;
use crate::clustering::hierarchical::HierarchicalClustering;
use crate::clustering::DistanceMetric;
use log::info;
//use crate::visualization;
use ndarray::ArrayView2;
use num_traits::float::FloatCore;
use num_traits::{Float, FromPrimitive, Signed};
use rand_distr::uniform::SampleUniform;
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;
use std::{fmt::Debug, sync::Arc};

use super::spann_index::SpannIndex;

#[derive(Debug, Clone, Copy)]
pub enum InitializationMethod {
    Random,
    KMeansPlusPlus,
}

pub struct SpannIndexBuilder<'a, F: Float> {
    config: Config,
    data: Option<ArrayView2<'a, F>>,
}

impl<'a, F: Float> SpannIndexBuilder<'a, F>
where
    F: Float
        + Debug
        + Default
        + AddAssign
        + AddAssign<&'a F>
        + std::iter::Sum
        + std::iter::Sum<&'a F>
        + ndarray::ScalarOperand
        + SampleUniform
        + Serialize
        + for<'de> Deserialize<'de>
        + FloatCore
        + Default
        + Signed
        + Debug
        + Copy
        + Sync
        + Send
        + std::ops::AddAssign
        + FromPrimitive,
{
    /// Create a new builder from a config.
    pub fn new(config: Config) -> Self {
        Self { config, data: None }
    }

    /// Optionally specify in-memory data (instead of reading from file).
    pub fn with_data(mut self, data: ArrayView2<'a, F>) -> Self {
        self.data = Some(data);
        self
    }

    pub fn build<const N: usize>(self) -> Result<SpannIndex<N, F>, Box<dyn std::error::Error>> {
        self.config.setup_logging();

        // Get data from either source
        let data = if let Some(data) = self.data {
            data
        } else {
            return Err("No data provided (in-memory or file)".into());
        };

        // Validate data dimensions
        if data.ncols() != N {
            return Err(format!(
                "Data dimension mismatch: expected {}, got {}",
                N,
                data.ncols()
            )
            .into());
        }

        let clustering_params = self.config.to_clustering_params();
        let mut clustering: HierarchicalClustering<N, F> =
            HierarchicalClustering::new(clustering_params, data);
        clustering.fit()?;
        //visualization::print_cluster_analysis(&clustering.clusters, &data.to_owned());

        let mut spann_index = SpannIndex::<N, F>::new();
        spann_index.create_posting_lists(&clustering.data, &clustering.clusters);
        spann_index.build_kdtree(&clustering.data, &clustering.clusters);

        if let Some(output_path) = &self.config.output_path {
            let _ = spann_index.save_posting_list(&format!("{}/output.posting", output_path));
            let _ = spann_index.save_kdtree(&format!("{}/output.kdtree", output_path));
        }
        Ok(spann_index)
    }

    pub fn load<const N: usize>(&self) -> Result<SpannIndex<N, F>, Box<dyn std::error::Error>> {
        self.config.setup_logging();
        let mut spann_index = SpannIndex::<N, F>::new();
        if let Some(output_path) = &self.config.output_path {
            let _ = spann_index.load_posting_list(&format!("{}/output.posting", output_path));
            let _ = spann_index.load_kdtree(&format!("{}/output.kdtree", output_path));
        } else {
            return Err("Output path is not specified".into());
        }
        Ok(spann_index)
    }
}

#[derive(Debug, Clone)]
pub struct ClusteringParams<F: Float> {
    pub distance_metric: Arc<dyn DistanceMetric<F>>,
    pub initialization_method: InitializationMethod,
    pub desired_cluster_size: usize,
    pub initial_k: usize,
    pub rng_seed: Option<u64>,
}

#[derive(Debug, Clone)]
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
