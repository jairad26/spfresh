use super::{LireError, LireResult};
use crate::core::float::SpannFloat;
use crate::distances::DistanceMetric;
use ndarray::{Array1, ArrayView1};
use std::collections::HashSet;

/// Represents a partition split operation
pub struct Split<F: SpannFloat> {
    pub posting_id: usize,
    pub vectors: Vec<(usize, Vec<F>)>,  // (vector_id, vector_data)
    pub distance_metric: Box<dyn DistanceMetric<F>>,
    pub new_posting_ids: (usize, usize),
}

impl<F: SpannFloat> Split<F> {
    /// Creates a new split operation
    pub fn new(
        posting_id: usize,
        vectors: Vec<(usize, Vec<F>)>,
        distance_metric: Box<dyn DistanceMetric<F>>,
        new_posting_ids: (usize, usize),
    ) -> Self {
        Self {
            posting_id,
            vectors,
            distance_metric,
            new_posting_ids,
        }
    }

    /// Selects initial centroids for the split
    fn select_initial_centroids(&self) -> LireResult<(Array1<F>, Array1<F>)> {
        if self.vectors.len() < 2 {
            return Err(LireError::SplitError("Not enough vectors to split".into()));
        }

        // Select first centroid randomly (using first vector for simplicity)
        let first_centroid = Array1::from(self.vectors[0].1.clone());

        // Find the farthest vector from the first centroid as the second centroid
        let (_, second_centroid) = self.vectors.iter()
            .skip(1)  // Skip the first vector
            .map(|(_, vec)| {
                let vec_array = Array1::from(vec.clone());
                let dist = self.distance_metric.compute(
                    &first_centroid.view(),
                    &vec_array.view(),
                );
                (dist, vec_array)
            })
            .max_by(|(dist1, _), (dist2, _)| {
                dist1.partial_cmp(dist2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| LireError::SplitError("Failed to find second centroid".into()))?;

        Ok((first_centroid, second_centroid))
    }

    /// Assigns vectors to the new partitions
    fn assign_vectors(
        &self,
        centroid1: &Array1<F>,
        centroid2: &Array1<F>,
    ) -> LireResult<(Vec<(usize, Vec<F>)>, Vec<(usize, Vec<F>)>)> {
        let mut partition1 = Vec::new();
        let mut partition2 = Vec::new();

        for (id, vec) in &self.vectors {
            let vec_array = Array1::from(vec.clone());
            let dist1 = self.distance_metric.compute(&centroid1.view(), &vec_array.view());
            let dist2 = self.distance_metric.compute(&centroid2.view(), &vec_array.view());

            if dist1 <= dist2 {
                partition1.push((*id, vec.clone()));
            } else {
                partition2.push((*id, vec.clone()));
            }
        }

        Ok((partition1, partition2))
    }
}

impl<F: SpannFloat> PartitionOperation<F> for Split<F> {
    fn execute(&self) -> LireResult<HashSet<usize>> {
        // 1. Select initial centroids
        let (centroid1, centroid2) = self.select_initial_centroids()?;

        // 2. Assign vectors to new partitions
        let (partition1, partition2) = self.assign_vectors(&centroid1, &centroid2)?;

        // 3. Track affected partitions
        let mut affected_partitions = HashSet::new();
        affected_partitions.insert(self.posting_id);
        affected_partitions.insert(self.new_posting_ids.0);
        affected_partitions.insert(self.new_posting_ids.1);

        // Return affected partitions
        Ok(affected_partitions)
    }

    fn validate(&self) -> bool {
        // Validate that:
        // 1. We have enough vectors to split
        // 2. New posting IDs are different from the original
        // 3. New posting IDs are different from each other
        self.vectors.len() >= 2 
            && self.new_posting_ids.0 != self.posting_id 
            && self.new_posting_ids.1 != self.posting_id
            && self.new_posting_ids.0 != self.new_posting_ids.1
    }

    fn get_affected_partitions(&self) -> HashSet<usize> {
        let mut affected = HashSet::new();
        affected.insert(self.posting_id);
        affected.insert(self.new_posting_ids.0);
        affected.insert(self.new_posting_ids.1);
        affected
    }
}

/// Represents a partition merge operation
pub struct Merge<F: SpannFloat> {
    pub posting_ids: (usize, usize),
    pub vectors1: Vec<(usize, Vec<F>)>,
    pub vectors2: Vec<(usize, Vec<F>)>,
    pub distance_metric: Box<dyn DistanceMetric<F>>,
    pub new_posting_id: usize,
}

impl<F: SpannFloat> Merge<F> {
    /// Creates a new merge operation
    pub fn new(
        posting_ids: (usize, usize),
        vectors1: Vec<(usize, Vec<F>)>,
        vectors2: Vec<(usize, Vec<F>)>,
        distance_metric: Box<dyn DistanceMetric<F>>,
        new_posting_id: usize,
    ) -> Self {
        Self {
            posting_ids,
            vectors1,
            vectors2,
            distance_metric,
            new_posting_id,
        }
    }

    /// Computes the centroid of the merged partition
    fn compute_merged_centroid(&self) -> Array1<F> {
        if self.vectors1.is_empty() && self.vectors2.is_empty() {
            return Array1::zeros(0);
        }

        let dim = self.vectors1.first()
            .map(|(_, v)| v.len())
            .or_else(|| self.vectors2.first().map(|(_, v)| v.len()))
            .unwrap_or(0);

        let mut centroid = Array1::zeros(dim);
        let total_vectors = self.vectors1.len() + self.vectors2.len();

        // Sum all vectors
        for (_, vec) in self.vectors1.iter().chain(self.vectors2.iter()) {
            for (i, &val) in vec.iter().enumerate() {
                centroid[i] = centroid[i] + val;
            }
        }

        // Divide by total number of vectors
        if total_vectors > 0 {
            for i in 0..dim {
                centroid[i] = centroid[i] / F::from_usize(total_vectors).unwrap();
            }
        }

        centroid
    }
}

impl<F: SpannFloat> PartitionOperation<F> for Merge<F> {
    fn execute(&self) -> LireResult<HashSet<usize>> {
        if !self.validate() {
            return Err(LireError::MergeError("Invalid merge operation".into()));
        }

        // Compute new centroid (for future reassignment checks)
        let _merged_centroid = self.compute_merged_centroid();

        // Track affected partitions
        let mut affected_partitions = HashSet::new();
        affected_partitions.insert(self.posting_ids.0);
        affected_partitions.insert(self.posting_ids.1);
        affected_partitions.insert(self.new_posting_id);

        Ok(affected_partitions)
    }

    fn validate(&self) -> bool {
        // Validate that:
        // 1. Source posting IDs are different
        // 2. New posting ID is different from source IDs
        // 3. At least one partition has vectors
        self.posting_ids.0 != self.posting_ids.1
            && self.new_posting_id != self.posting_ids.0
            && self.new_posting_id != self.posting_ids.1
            && (!self.vectors1.is_empty() || !self.vectors2.is_empty())
    }

    fn get_affected_partitions(&self) -> HashSet<usize> {
        let mut affected = HashSet::new();
        affected.insert(self.posting_ids.0);
        affected.insert(self.posting_ids.1);
        affected.insert(self.new_posting_id);
        affected
    }
}

/// Represents a vector reassignment operation
pub struct Reassign<F: SpannFloat> {
    pub vector_id: usize,
    pub vector: Vec<F>,
    pub from_posting: usize,
    pub candidate_postings: Vec<(usize, Vec<F>)>, // (posting_id, centroid)
    pub distance_metric: Box<dyn DistanceMetric<F>>,
    pub version: u64,
}

impl<F: SpannFloat> Reassign<F> {
    /// Creates a new reassignment operation
    pub fn new(
        vector_id: usize,
        vector: Vec<F>,
        from_posting: usize,
        candidate_postings: Vec<(usize, Vec<F>)>,
        distance_metric: Box<dyn DistanceMetric<F>>,
        version: u64,
    ) -> Self {
        Self {
            vector_id,
            vector,
            from_posting,
            candidate_postings,
            distance_metric,
            version,
        }
    }

    /// Finds the best posting for the vector based on distance to centroids
    fn find_best_posting(&self) -> LireResult<usize> {
        if self.candidate_postings.is_empty() {
            return Err(LireError::ReassignError("No candidate postings available".into()));
        }

        let vector_array = Array1::from(self.vector.clone());
        
        let (best_posting, _) = self.candidate_postings
            .iter()
            .map(|(posting_id, centroid)| {
                let centroid_array = Array1::from(centroid.clone());
                let distance = self.distance_metric.compute(
                    &vector_array.view(),
                    &centroid_array.view(),
                );
                (*posting_id, distance)
            })
            .min_by(|(_, dist1), (_, dist2)| {
                dist1.partial_cmp(dist2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| LireError::ReassignError("Failed to find best posting".into()))?;

        Ok(best_posting)
    }
}

impl<F: SpannFloat> PartitionOperation<F> for Reassign<F> {
    fn execute(&self) -> LireResult<HashSet<usize>> {
        if !self.validate() {
            return Err(LireError::ReassignError("Invalid reassignment operation".into()));
        }

        let best_posting = self.find_best_posting()?;
        
        // Track affected partitions
        let mut affected_partitions = HashSet::new();
        affected_partitions.insert(self.from_posting);
        affected_partitions.insert(best_posting);

        Ok(affected_partitions)
    }

    fn validate(&self) -> bool {
        // Validate that:
        // 1. We have candidate postings
        // 2. Vector data is not empty
        // 3. Current posting is valid
        !self.candidate_postings.is_empty()
            && !self.vector.is_empty()
            && self.candidate_postings.iter().all(|(id, centroid)| {
                *id != self.from_posting && !centroid.is_empty()
            })
    }

    fn get_affected_partitions(&self) -> HashSet<usize> {
        let mut affected = HashSet::new();
        affected.insert(self.from_posting);
        self.candidate_postings.iter().for_each(|(id, _)| {
            affected.insert(*id);
        });
        affected
    }
}

/// Trait for partition operations
pub trait PartitionOperation<F: SpannFloat> {
    fn execute(&self) -> LireResult<HashSet<usize>>;
    fn validate(&self) -> bool;
    fn get_affected_partitions(&self) -> HashSet<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distances::SquaredEuclideanDistance;
    use ndarray::Array1;

    fn create_test_split() -> Split<f32> {
        // Create test vectors in 2D space
        let vectors = vec![
            (0, vec![1.0, 1.0]),    // Cluster 1
            (1, vec![1.5, 1.5]),    // Cluster 1
            (2, vec![8.0, 8.0]),    // Cluster 2
            (3, vec![8.5, 8.5]),    // Cluster 2
        ];

        Split::new(
            0, // original posting_id
            vectors,
            Box::new(SquaredEuclideanDistance),
            (1, 2), // new posting IDs
        )
    }

    #[test]
    fn test_split_validation() {
        let split = create_test_split();
        assert!(split.validate());

        // Test invalid split (not enough vectors)
        let invalid_split = Split::new(
            0,
            vec![(0, vec![1.0, 1.0])],
            Box::new(SquaredEuclideanDistance),
            (1, 2),
        );
        assert!(!invalid_split.validate());

        // Test invalid split (duplicate posting IDs)
        let invalid_split = Split::new(
            0,
            vec![(0, vec![1.0, 1.0]), (1, vec![2.0, 2.0])],
            Box::new(SquaredEuclideanDistance),
            (1, 1), // Same new posting IDs
        );
        assert!(!invalid_split.validate());
    }

    #[test]
    fn test_centroid_selection() {
        let split = create_test_split();
        let (centroid1, centroid2) = split.select_initial_centroids().unwrap();
        
        // Check that centroids are different
        assert!(split.distance_metric.compute(&centroid1.view(), &centroid2.view()) > 0.0);
    }

    #[test]
    fn test_vector_assignment() {
        let split = create_test_split();
        let (centroid1, centroid2) = split.select_initial_centroids().unwrap();
        let (partition1, partition2) = split.assign_vectors(&centroid1, &centroid2).unwrap();

        // Check that vectors are assigned to appropriate clusters
        assert_eq!(partition1.len(), 2); // Should have vectors near (1,1)
        assert_eq!(partition2.len(), 2); // Should have vectors near (8,8)

        // Verify that close vectors are clustered together
        let (id1, vec1) = &partition1[0];
        let (id2, vec2) = &partition1[1];
        assert!(*id1 < 2 && *id2 < 2); // IDs 0 and 1 should be in first partition

        let (id1, vec1) = &partition2[0];
        let (id2, vec2) = &partition2[1];
        assert!(*id1 >= 2 && *id2 >= 2); // IDs 2 and 3 should be in second partition
    }

    #[test]
    fn test_execute() {
        let split = create_test_split();
        let affected_partitions = split.execute().unwrap();

        // Check that all relevant partitions are marked as affected
        assert!(affected_partitions.contains(&0)); // Original posting
        assert!(affected_partitions.contains(&1)); // New posting 1
        assert!(affected_partitions.contains(&2)); // New posting 2
        assert_eq!(affected_partitions.len(), 3);
    }

    fn create_test_merge() -> Merge<f32> {
        // Create two clusters of vectors to merge
        let vectors1 = vec![
            (0, vec![1.0, 1.0]),
            (1, vec![1.5, 1.5]),
        ];
        let vectors2 = vec![
            (2, vec![2.0, 2.0]),
            (3, vec![2.5, 2.5]),
        ];

        Merge::new(
            (1, 2), // posting_ids to merge
            vectors1,
            vectors2,
            Box::new(SquaredEuclideanDistance),
            3, // new posting ID
        )
    }

    #[test]
    fn test_merge_validation() {
        let merge = create_test_merge();
        assert!(merge.validate());

        // Test invalid merge (same posting IDs)
        let invalid_merge = Merge::new(
            (1, 1),
            vec![(0, vec![1.0, 1.0])],
            vec![(1, vec![2.0, 2.0])],
            Box::new(SquaredEuclideanDistance),
            3,
        );
        assert!(!invalid_merge.validate());

        // // Test invalid merge (empty vectors)
        // let invalid_merge = Merge::new(
        //     (1, 2),
        //     vec![],
        //     vec![],
        //     Box::new(SquaredEuclideanDistance),
        //     3,
        // );
        // assert!(!invalid_merge.validate());

        // Test invalid merge (new posting ID conflicts)
        let invalid_merge = Merge::new(
            (1, 2),
            vec![(0, vec![1.0, 1.0])],
            vec![(1, vec![2.0, 2.0])],
            Box::new(SquaredEuclideanDistance),
            1, // Conflicts with source posting ID
        );
        assert!(!invalid_merge.validate());
    }

    #[test]
    fn test_merge_centroid_computation() {
        let merge = create_test_merge();
        let centroid = merge.compute_merged_centroid();

        // Expected centroid should be average of all points: (1.0+1.5+2.0+2.5)/4 = 1.75 for both dimensions
        assert!((centroid[0] - 1.75).abs() < 1e-6);
        assert!((centroid[1] - 1.75).abs() < 1e-6);
    }

    #[test]
    fn test_merge_execute() {
        let merge = create_test_merge();
        let affected_partitions = merge.execute().unwrap();

        // Check that all relevant partitions are marked as affected
        assert!(affected_partitions.contains(&1)); // First source posting
        assert!(affected_partitions.contains(&2)); // Second source posting
        assert!(affected_partitions.contains(&3)); // New posting
        assert_eq!(affected_partitions.len(), 3);
    }

    fn create_test_reassign() -> Reassign<f32> {
        // Vector to reassign (closer to centroid2)
        let vector = vec![7.0, 7.0];
        
        // Candidate posting centroids
        let candidate_postings = vec![
            (1, vec![1.0, 1.0]),  // centroid1
            (2, vec![8.0, 8.0]),  // centroid2
            (3, vec![4.0, 4.0]),  // centroid3
        ];

        Reassign::new(
            0,              // vector_id
            vector,
            0,              // from_posting
            candidate_postings,
            Box::new(SquaredEuclideanDistance),
            1,              // version
        )
    }

    #[test]
    fn test_reassign_validation() {
        let reassign = create_test_reassign();
        assert!(reassign.validate());

        // Test invalid reassign (empty vector)
        let invalid_reassign = Reassign::new(
            0,
            vec![],
            0,
            vec![(1, vec![1.0, 1.0])],
            Box::new(SquaredEuclideanDistance),
            1,
        );
        assert!(!invalid_reassign.validate());

        // Test invalid reassign (no candidate postings)
        let invalid_reassign = Reassign::new(
            0,
            vec![1.0, 1.0],
            0,
            vec![],
            Box::new(SquaredEuclideanDistance),
            1,
        );
        assert!(!invalid_reassign.validate());

        // Test invalid reassign (candidate posting same as source)
        let invalid_reassign = Reassign::new(
            0,
            vec![1.0, 1.0],
            1,
            vec![(1, vec![1.0, 1.0])],  // Same as from_posting
            Box::new(SquaredEuclideanDistance),
            1,
        );
        assert!(!invalid_reassign.validate());
    }

    #[test]
    fn test_find_best_posting() {
        let reassign = create_test_reassign();
        let best_posting = reassign.find_best_posting().unwrap();

        // Vector [7.0, 7.0] should be closest to centroid2 [8.0, 8.0]
        assert_eq!(best_posting, 2);

        // Test with a different vector
        let reassign = Reassign::new(
            0,
            vec![1.5, 1.5],  // This vector is closest to centroid1
            0,
            reassign.candidate_postings,
            Box::new(SquaredEuclideanDistance),
            1,
        );
        let best_posting = reassign.find_best_posting().unwrap();
        assert_eq!(best_posting, 1);
    }

    #[test]
    fn test_reassign_execute() {
        let reassign = create_test_reassign();
        let affected_partitions = reassign.execute().unwrap();

        // Check that source and destination partitions are marked as affected
        assert!(affected_partitions.contains(&0));  // Source posting
        assert!(affected_partitions.contains(&2));  // Best posting (destination)
        assert_eq!(affected_partitions.len(), 2);
    }

    #[test]
    fn test_reassign_affected_partitions() {
        let reassign = create_test_reassign();
        let affected = reassign.get_affected_partitions();

        // Should include source posting and all candidate postings
        assert!(affected.contains(&0));  // Source posting
        assert!(affected.contains(&1));  // Candidate 1
        assert!(affected.contains(&2));  // Candidate 2
        assert!(affected.contains(&3));  // Candidate 3
        assert_eq!(affected.len(), 4);
    }

    #[test]
    fn test_reassign_with_different_distances() {
        // Test with Manhattan distance
        use crate::distances::ManhattanDistance;
        
        let reassign = Reassign::new(
            0,
            vec![7.0, 7.0],
            0,
            vec![
                (1, vec![1.0, 1.0]),
                (2, vec![8.0, 8.0]),
                (3, vec![4.0, 4.0]),
            ],
            Box::new(ManhattanDistance),
            1,
        );

        let best_posting = reassign.find_best_posting().unwrap();
        assert_eq!(best_posting, 2);  // Should still choose posting 2 as closest
    }
}
