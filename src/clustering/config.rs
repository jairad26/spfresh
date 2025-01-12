use super::distance::{ChebyshevDistance, ManhattanDistance, SquaredEuclideanDistance};
use crate::clustering::float::AdriannFloat;
use crate::clustering::{ClusteringParams, InitializationMethod};
use log::{error, LevelFilter};
use ndarray::Array2;
use serde::Deserialize;
use std::{fmt, sync::Arc};

#[derive(Debug, Deserialize)]
pub struct ClusteringParamsConfig {
    pub distance_metric: String,       // E.g., "Euclidean"
    pub initialization_method: String, // E.g., "KMeansPlusPlus"
    pub desired_cluster_size: usize,
    pub initial_k: usize,
}

#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    pub level: String, // Log level, e.g., "info", "debug", "warn", "error"
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub clustering_params: ClusteringParamsConfig,
    pub data_file: Option<String>,   // Path to the dataset file
    pub logging: LoggingConfig,      // Logging settings
    pub output_path: Option<String>, // Path to store the SPANN index
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Configuration:")?;
        writeln!(f, "  Clustering Parameters:")?;
        writeln!(
            f,
            "    Desired Cluster Size: {}",
            self.clustering_params.desired_cluster_size
        )?;
        writeln!(f, "    Initial K: {}", self.clustering_params.initial_k)?;
        writeln!(
            f,
            "    Distance Metric: {}",
            self.clustering_params.distance_metric
        )?;
        writeln!(
            f,
            "    Initialization Method: {}",
            self.clustering_params.initialization_method
        )?;
        if let Some(data_file) = &self.data_file {
            writeln!(f, "  Data File: {}", data_file)?;
        } else {
            writeln!(f, "  Data File: None")?;
        }
        writeln!(f, "  Logging:")?;
        writeln!(f, "    Level: {}", self.logging.level)?;
        if let Some(output_path) = &self.output_path {
            writeln!(f, "  Output Path: {}", output_path)?;
        } else {
            writeln!(f, "  Output Path: None")?;
        }
        Ok(())
    }
}

impl Config {
    /// Reads the YAML configuration file and returns a `Config` instance.
    pub fn from_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file_content = std::fs::read_to_string(file_path)?;
        let config: Config = serde_yaml::from_str(&file_content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        // Validate distance metric
        match self.clustering_params.distance_metric.as_str() {
            "Euclidean" | "Manhattan" => (),
            _ => {
                return Err(format!(
                    "Unsupported distance metric: {}",
                    self.clustering_params.distance_metric
                ))
            }
        }

        // Validate initialization method
        match self.clustering_params.initialization_method.as_str() {
            "Random" | "KMeansPlusPlus" => (),
            _ => {
                return Err(format!(
                    "Unsupported initialization method: {}",
                    self.clustering_params.initialization_method
                ))
            }
        }

        // Validate numeric parameters
        if self.clustering_params.desired_cluster_size == 0 {
            return Err("desired_cluster_size must be greater than 0".to_string());
        }
        if self.clustering_params.initial_k == 0 {
            return Err("initial_k must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Converts `ClusteringParamsConfig` into `ClusteringParams`.
    pub fn to_clustering_params<F: AdriannFloat>(&self) -> ClusteringParams<F> {
        ClusteringParams {
            distance_metric: match self.clustering_params.distance_metric.as_str() {
                "Euclidean" => Arc::new(SquaredEuclideanDistance),
                "Manhattan" => Arc::new(ManhattanDistance),
                "Chebyshev" => Arc::new(ChebyshevDistance),
                _ => panic!(
                    "Unsupported distance metric: {}",
                    self.clustering_params.distance_metric
                ),
            },
            initialization_method: match self.clustering_params.initialization_method.as_str() {
                "Random" => InitializationMethod::Random,
                "KMeansPlusPlus" => InitializationMethod::KMeansPlusPlus,
                _ => panic!(
                    "Unsupported initialization method: {}",
                    self.clustering_params.initialization_method
                ),
            },
            desired_cluster_size: self.clustering_params.desired_cluster_size,
            initial_k: self.clustering_params.initial_k,
            rng_seed: None,
        }
    }

    /// Sets up logging based on the logging level in the configuration.
    pub fn setup_logging(&self) {
        let level_filter = match self.logging.level.to_lowercase().as_str() {
            "debug" => LevelFilter::Debug,
            "info" => LevelFilter::Info,
            "warn" => LevelFilter::Warn,
            "error" => LevelFilter::Error,
            _ => panic!("Unsupported log level: {}", self.logging.level),
        };

        if let Err(e) = env_logger::Builder::new()
            .filter_level(level_filter)
            .try_init()
        {
            error!("Failed to initialize logger: {}", e);
        }
    }

    /// Reads the dataset file.
    pub fn load_data(&self) -> Array2<f64> {
        // TODO: Replace with actual data loading logic.
        Array2::<f64>::zeros((100, 2)) // Mocked data
    }
}
