use super::{LireError, LireResult};
use crate::core::float::SpannFloat;
use std::path::PathBuf;

/// Represents a versioned vector in storage
#[derive(Debug)]
pub struct VersionedVector<F> {
    pub vector: Vec<F>,
    pub version: u64,
    pub is_deleted: bool,
}

/// SSD-backed storage engine for LIRE
pub struct LireStorage {
    base_path: PathBuf,
    current_version: u64,
}

impl LireStorage {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            base_path: path.into(),
            current_version: 0,
        }
    }

    pub fn store_vector<F: SpannFloat>(
        &mut self,
        posting_id: usize,
        vector_id: usize,
        vector: Vec<F>,
    ) -> LireResult<u64> {
        self.current_version += 1;
        // TODO: Implement actual storage logic
        Ok(self.current_version)
    }

    pub fn mark_deleted(
        &mut self,
        posting_id: usize,
        vector_id: usize,
    ) -> LireResult<()> {
        // TODO: Implement deletion logic
        Ok(())
    }
}
