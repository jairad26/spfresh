use super::{LireError, LireResult};
use crate::core::float::SpannFloat;
use crate::distances::DistanceMetric;
use crate::spann::posting_lists::PointData;
use ndarray::{Array1, ArrayView1};
use std::collections::HashSet;
use std::sync::Arc;

/// Configuration parameters for LIRE protocol
#[derive(Debug, Clone)]
pub struct LireConfig {
    /// Maximum size of a partition before splitting
    pub max_partition_size: usize,
    /// Minimum size of a partition before considering merge
    pub min_partition_size: usize,
    /// Number of nearby postings to scan for reassignment
    pub nearby_posting_count: usize,
    /// Threshold for triggering garbage collection
    pub gc_threshold: f64,
}

impl Default for LireConfig {
    fn default() -> Self {
        Self {
            max_partition_size: 10000,
            min_partition_size: 1000,
            nearby_posting_count: 64,
            gc_threshold: 0.3,
        }
    }
}

/// Result of an update operation
#[derive(Debug)]
pub struct UpdateResult {
    /// Number of vectors that were reassigned
    pub vectors_reassigned: usize,
    /// List of partition IDs that were affected
    pub partitions_affected: HashSet<usize>,
    /// Version number of the update
    pub version: u64,
}

/// Core LIRE protocol implementation
pub struct LireProtocol<F: SpannFloat> {
    config: LireConfig,
    distance_metric: Arc<dyn DistanceMetric<F>>,
    storage: Arc<super::storage::LireStorage>,
}

impl<F: SpannFloat> LireProtocol<F> {
    pub fn new(
        config: LireConfig,
        distance_metric: Arc<dyn DistanceMetric<F>>,
        storage: Arc<super::storage::LireStorage>,
    ) -> Self {
        Self {
            config,
            distance_metric,
            storage,
        }
    }

    /// Insert a new vector into the index
    pub fn insert(&self, vector: ArrayView1<F>, id: usize, posting_id: usize) -> LireResult<UpdateResult> {
        // Store the vector
        let version = self.storage.store_vector(posting_id, id, vector.to_vec())?;
        
        let mut result = UpdateResult {
            vectors_reassigned: 0,
            partitions_affected: HashSet::from([posting_id]),
            version,
        };

        // Check if partition needs splitting
        if self.needs_split(posting_id)? {
            self.schedule_maintenance(posting_id)?;
        }

        Ok(result)
    }

    /// Delete a vector from the index
    pub fn delete(&self, id: usize, posting_id: usize) -> LireResult<UpdateResult> {
        self.storage.mark_deleted(posting_id, id)?;
        
        let mut result = UpdateResult {
            vectors_reassigned: 0,
            partitions_affected: HashSet::from([posting_id]),
            version: self.storage.get_posting_version(posting_id)?,
        };

        // Check if partition needs merging
        if self.needs_merge(posting_id)? {
            self.schedule_maintenance(posting_id)?;
        }

        Ok(result)
    }

    /// Check if a partition needs to be split
    fn needs_split(&self, posting_id: usize) -> LireResult<bool> {
        let metadata = self.storage.get_posting_metadata(posting_id)?;
        Ok(metadata.vector_count > self.config.max_partition_size)
    }

    /// Check if a partition needs to be merged
    fn needs_merge(&self, posting_id: usize) -> LireResult<bool> {
        let metadata = self.storage.get_posting_metadata(posting_id)?;
        Ok(metadata.vector_count < self.config.min_partition_size)
    }

    /// Schedule maintenance for a partition
    fn schedule_maintenance(&self, posting_id: usize) -> LireResult<()> {
        // This will be implemented when we add the pipeline component
        // For now, just mark the partition as needing maintenance
        Ok(())
    }

    /// Find the nearest partition for a vector
    pub fn find_nearest_partition(&self, vector: ArrayView1<F>, candidates: &[usize]) -> LireResult<usize> {
        let mut nearest_posting = candidates[0];
        let mut min_distance = F::max_value();

        for &posting_id in candidates {
            let centroid = self.storage.get_posting_centroid(posting_id)?;
            let distance = self.distance_metric.compute(&vector, &centroid.view());
            
            if distance < min_distance {
                min_distance = distance;
                nearest_posting = posting_id;
            }
        }

        Ok(nearest_posting)
    }

    /// Get nearby postings for a given posting
    pub fn get_nearby_postings(&self, posting_id: usize) -> LireResult<Vec<usize>> {
        // This will be implemented when we add the posting list component
        // For now, return an empty vector
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distances::SquaredEuclideanDistance;
    use tempfile::TempDir;
    use ndarray::Array1;

    fn create_test_protocol() -> (LireProtocol<f32>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(super::super::storage::LireStorage::new(temp_dir.path()).unwrap());
        let distance_metric = Arc::new(SquaredEuclideanDistance);
        let config = LireConfig::default();
        
        (LireProtocol::new(config, distance_metric, storage), temp_dir)
    }

    #[test]
    fn test_insert_vector() {
        let (protocol, _temp_dir) = create_test_protocol();
        let vector = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        
        let result = protocol.insert(vector.view(), 1, 0).unwrap();
        assert_eq!(result.vectors_reassigned, 0);
        assert!(result.partitions_affected.contains(&0));
    }

    #[test]
    fn test_delete_vector() {
        let (protocol, _temp_dir) = create_test_protocol();
        
        // First insert a vector
        let vector = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        protocol.insert(vector.view(), 1, 0).unwrap();
        
        // Then delete it
        let result = protocol.delete(1, 0).unwrap();
        assert_eq!(result.vectors_reassigned, 0);
        assert!(result.partitions_affected.contains(&0));
    }

    #[test]
    fn test_partition_maintenance_triggers() {
        let (protocol, _temp_dir) = create_test_protocol();
        
        // Insert enough vectors to trigger split
        for i in 0..protocol.config.max_partition_size + 1 {
            let vector = Array1::from_vec(vec![i as f32, 0.0, 0.0]);
            protocol.insert(vector.view(), i, 0).unwrap();
        }
        
        assert!(protocol.needs_split(0).unwrap());
        
        // Delete enough vectors to trigger merge
        for i in 0..protocol.config.max_partition_size {
            protocol.delete(i, 0).unwrap();
        }
        
        assert!(protocol.needs_merge(0).unwrap());
    }
}
