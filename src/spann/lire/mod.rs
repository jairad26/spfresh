//! LIRE (Lightweight Incremental RE-balancing) implementation for SPFresh
//! Provides efficient in-place updates to vector indices

mod protocol;
mod operations;
mod pipeline;
mod storage;

pub use protocol::{LireProtocol, UpdateResult};
pub use operations::{Split, Merge, Reassign};
pub use pipeline::TwoStagePipeline;
pub use storage::LireStorage;

use std::error::Error;
use std::fmt;

/// Error types specific to LIRE operations
#[derive(Debug)]
pub enum LireError {
    /// Error during partition split
    SplitError(String),
    /// Error during partition merge
    MergeError(String),
    /// Error during vector reassignment
    ReassignError(String),
    /// Error in storage operations
    StorageError(String),
    /// Error in pipeline processing
    PipelineError(String),
}

impl fmt::Display for LireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LireError::SplitError(msg) => write!(f, "Split error: {}", msg),
            LireError::MergeError(msg) => write!(f, "Merge error: {}", msg),
            LireError::ReassignError(msg) => write!(f, "Reassign error: {}", msg),
            LireError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            LireError::PipelineError(msg) => write!(f, "Pipeline error: {}", msg),
        }
    }
}

impl Error for LireError {}

pub type LireResult<T> = Result<T, LireError>;
