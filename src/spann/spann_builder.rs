use log::info;
use ndarray::ArrayView2;
use crate::clustering::HierarchicalClustering;
use crate::core::float::AdriannFloat;
use crate::spann::config::Config;
use crate::spann::SpannIndex;

pub struct SpannIndexBuilder<'a, F: AdriannFloat> {
    config: Config,
    data: Option<ArrayView2<'a, F>>,
}

impl<'a, F: AdriannFloat> SpannIndexBuilder<'a, F> {
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
        info!(
            "Building SPANN index with configuration: {}",
            self.config.to_string()
        );
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

        let mut clustering_params = self.config.to_clustering_params();
        clustering_params.desired_cluster_size =
            Some((data.nrows() as f64 * 0.18).round() as usize);
        let mut clustering: HierarchicalClustering<N, F> =
            HierarchicalClustering::new(clustering_params, data);
        clustering.fit()?;

        if let Some(output_path) = &self.config.output_path {
            let mut spann_index = SpannIndex::<N, F>::new(output_path).unwrap();
            spann_index.create_posting_lists(&clustering.data, &clustering.clusters);
            spann_index.build_kdtree(&clustering.data, &clustering.clusters);
            let _ = spann_index.save_posting_list();
            let _ = spann_index.save_kdtree(&format!("{}/output.kdtree", output_path));
            Ok(spann_index)
        } else {
            Err("Output path is not specified".into())
        }
    }

    pub fn load<const N: usize>(&self) -> Result<SpannIndex<N, F>, Box<dyn std::error::Error>> {
        if let Some(output_path) = &self.config.output_path {
            let mut spann_index = SpannIndex::<N, F>::new(output_path).unwrap();
            let _ = spann_index.load_posting_list(output_path);
            let _ = spann_index.load_kdtree(&format!("{}/output.kdtree", output_path));
            Ok(spann_index)
        } else {
            Err("Output path is not specified".into())
        }
    }
}