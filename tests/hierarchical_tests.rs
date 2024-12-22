#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use adriann::clustering::{
        ClusteringParams, HierarchicalClustering, InitializationMethod, SquaredEuclideanDistance,
    };
    use ndarray::Array2;

    #[test]
    fn test_clustering() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 2.5, 8.0, 8.0, 8.5, 8.5, 4.0, 4.0, 4.5, 4.5],
        )
        .unwrap();

        let params = ClusteringParams {
            distance_metric: Arc::new(SquaredEuclideanDistance),
            initialization_method: InitializationMethod::KMeansPlusPlus,
            desired_cluster_size: 2,
            initial_k: 3,
            rng_seed: Some(42),
        };

        let mut clustering = HierarchicalClustering::<2, f64>::new(params, data.view());
        clustering.fit().expect("Clustering failed");

        assert_eq!(clustering.clusters.len(), 3);
        for cluster in &clustering.clusters {
            assert!(cluster.points.len() <= 2);
        }
    }
}
