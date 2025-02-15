use super::{LireError, LireResult};
use crate::core::float::SpannFloat;
use ndarray::ArrayView1;

/// Result of an update operation
#[derive(Debug)]
pub struct UpdateResult {
    pub vectors_reassigned: usize,
    pub partitions_affected: Vec<usize>,
}

/// Core LIRE protocol trait
pub trait LireProtocol<F: SpannFloat> {
    /// Insert a new vector into the index
    fn insert(&mut self, vector: ArrayView1<F>, id: usize) -> LireResult<UpdateResult>;
    
    /// Delete a vector from the index
    fn delete(&mut self, id: usize) -> LireResult<UpdateResult>;
    
    /// Check if any partitions need maintenance
    fn check_maintenance_needed(&self) -> bool;
    
    /// Perform maintenance operations (splits, merges, reassignments)
    fn perform_maintenance(&mut self) -> LireResult<UpdateResult>;
}
