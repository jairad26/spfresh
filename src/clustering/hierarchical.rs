use crate::core::float::AdriannFloat;
use log::{debug, error};
use ndarray::ArrayView2;
use rand::seq::{IndexedRandom, IteratorRandom};
use rand::{rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::error::Error;
use std::sync::Arc;
use crate::clustering::utils::compute_mean;
use crate::distances::DistanceMetric;

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

pub struct HierarchicalClustering<'a, const N: usize, F: AdriannFloat> {
    pub clusters: Vec<Cluster>,
    pub data: Arc<ArrayView2<'a, F>>,
    pub params: ClusteringParams<F>,
}

impl<'a, const N: usize, F: AdriannFloat> HierarchicalClustering<'a, N, F>
where
    F: AdriannFloat,
{
    /// A constant factor for deciding whether a point is a "boundary" point
    /// (i.e., it's also close enough to other clusters).
    const BOUNDARY_THRESHOLD: f64 = 1.1;

    pub fn new(params: ClusteringParams<F>, data: ArrayView2<'a, F>) -> Self {
        Self {
            clusters: Vec::new(),
            data: Arc::new(data),
            params,
        }
    }

    pub fn fit(&mut self) -> Result<(), Box<dyn Error>> {
        self.initialize_clusters(self.params.initial_k);
        self.assign_points();
        self.update_centroids();
        self.subdivide_clusters();
        Ok(())
    }

    /// Subdivides clusters
    fn subdivide_clusters(&mut self) {
        let mut i = 0;
        while i < self.clusters.len() {
            let cluster_size = self.clusters[i].points.len();

            if cluster_size > self.params.desired_cluster_size.unwrap() {
                debug!(
                    "Subdividing cluster {} with size {} until we get to {}",
                    i,
                    cluster_size,
                    self.params.desired_cluster_size.unwrap()
                );

                // Get the points from the current cluster
                let points = self.clusters[i].points.clone();
                let depth = self.clusters[i].depth;

                // Split into two clusters - store results temporarily
                let (subcluster1, subcluster2) = self.create_subclusters(&points, depth + 1);

                // Replace the original cluster with subcluster1
                self.clusters[i] = subcluster1;

                // Add subcluster2 at the end
                self.clusters.push(subcluster2);

                // Don't increment i since we need to check the new cluster at position i again
            } else {
                i += 1;
            }
        }
    }

    fn create_subclusters(&self, points: &[usize], new_depth: usize) -> (Cluster, Cluster) {
        let mut rng = self.get_rng();

        // Select initial centroids
        let centroid1_idx = *points.choose(&mut rng).unwrap();
        let centroid2_idx = points
            .iter()
            .filter(|&&idx| idx != centroid1_idx)
            .fold((0, F::zero()), |(max_idx, max_dist), &idx| {
                let dist = self
                    .params
                    .distance_metric
                    .compute(&self.data.row(centroid1_idx), &self.data.row(idx));
                if dist > max_dist {
                    (idx, dist)
                } else {
                    (max_idx, max_dist)
                }
            })
            .0;

        let centroids = vec![(centroid1_idx, new_depth), (centroid2_idx, new_depth)];
        let assignments = self.assign_points_to_clusters(points, &centroids);

        (
            Cluster::new(centroid1_idx, assignments[0].clone(), new_depth),
            Cluster::new(centroid2_idx, assignments[1].clone(), new_depth),
        )
    }

    // with SPANN we need to update the centroids
    fn update_centroids(&mut self) {
        let distance_metric = &self.params.distance_metric;

        // We'll collect new centroids in a separate vector
        let new_centroids: Vec<Option<usize>> = self
            .clusters
            .par_iter()
            .map(|cluster| {
                if cluster.points.is_empty() {
                    // If no points, centroid_idx remains the same
                    return cluster.centroid_idx;
                }

                // Compute the mean for cluster's points
                let mean = compute_mean(&self.data, &cluster.points);

                // Find the row that is closest to this mean
                let (best_idx, _best_dist) = cluster
                    .points
                    .par_iter()
                    .map(|&pt| {
                        let d = distance_metric.compute(&self.data.row(pt), &mean.view());
                        (pt, d)
                    })
                    .reduce(
                        || (0, F::infinity()),
                        |(min_idx, min_dist), (pt, dist)| {
                            if dist < min_dist {
                                (pt, dist)
                            } else {
                                (min_idx, min_dist)
                            }
                        },
                    );

                Some(best_idx)
            })
            .collect();

        // Now update the clusters on a single thread
        for (cluster, new_idx) in self.clusters.iter_mut().zip(new_centroids) {
            cluster.centroid_idx = new_idx;
        }
    }

    /// Returns a random number generator based on the seed (or entropy).
    fn get_rng(&self) -> SmallRng {
        match self.params.rng_seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_rng(&mut rng())
        }
    }

    /// Initializes clusters using the specified method.
    fn initialize_clusters(&mut self, k: usize) {
        match self.params.initialization_method {
            InitializationMethod::Random => self.initialize_clusters_randomly(k),
            InitializationMethod::KMeansPlusPlus => self.initialize_clusters_kmeans_plus_plus(k),
        }
    }

    /// Initializes clusters randomly by selecting `k` points from the dataset.
    fn initialize_clusters_randomly(&mut self, k: usize) {
        let n_points = self.data.nrows();
        let mut rng = self.get_rng();

        let centroid_indices: Vec<usize> = (0..n_points).choose_multiple(&mut rng, k);

        self.clusters = centroid_indices
            .into_iter()
            .map(|idx| Cluster::new(idx, Vec::new(), 0))
            .collect();
    }

    /// Returns a single "best cluster" label for each point in 0..nrows.
    /// If a point belongs to multiple clusters (boundary), we pick the cluster
    /// whose centroid is truly nearest.
    pub fn labels(&self) -> Vec<usize> {
        let mut labels = vec![0usize; self.data.nrows()];

        for c_idx in 0..self.clusters.len() {
            for &point_idx in &self.clusters[c_idx].points {
                // If this point belongs to multiple clusters, pick the closest
                let point = self.data.row(point_idx);
                let this_dist = &self.params.distance_metric.compute(
                    &point,
                    &self
                        .data
                        .row(self.clusters[c_idx].centroid_idx.unwrap())
                        .view(),
                );

                // Compare to distance from the cluster currently in labels
                let old_label = labels[point_idx];
                let old_dist = &self.params.distance_metric.compute(
                    &point,
                    &self
                        .data
                        .row(self.clusters[old_label].centroid_idx.unwrap())
                        .view(),
                );
                if this_dist < old_dist {
                    labels[point_idx] = c_idx;
                }
            }
        }

        labels
    }

    /// Initializes clusters using the KMeans++ method.
    fn initialize_clusters_kmeans_plus_plus(&mut self, k: usize) {
        let n_points = self.data.nrows();
        let mut rng = self.get_rng();
        // Randomly select the first centroid
        let first_idx = (0..n_points)
            .choose(&mut rng)
            .expect("Failed to choose first centroid");
        self.clusters.push(Cluster::new(first_idx, Vec::new(), 0));

        // Select the remaining k-1 centroids
        for _ in 1..k {
            let distances: Vec<F> = (0..n_points)
                .into_par_iter()
                .map(|i| {
                    let point = self.data.row(i);
                    // Distance to the *closest* existing centroid
                    self.clusters
                        .par_iter()
                        .map(|cluster| {
                            self.params.distance_metric.compute(
                                &point,
                                &self.data.row(cluster.centroid_idx.unwrap()).view(),
                            )
                        })
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .collect();

            let sum = distances.iter().fold(F::zero(), |acc, &x| acc + x);
            let weights: Vec<F> = distances
                .par_iter()
                .map(|&d| (d * d) / F::max(sum, F::from(1e-10).unwrap()))
                .collect();
            let indices: Vec<_> = (0..n_points).collect();
            // Weighted selection
            let chosen_idx = indices
                .choose_weighted(&mut rng, |i| weights[*i].to_f64().unwrap())
                .unwrap_or_else(|e| {
                    error!("Weighted selection failed: {:?}", e);
                    indices.choose(&mut rng).unwrap()
                });
            self.clusters.push(Cluster::new(*chosen_idx, Vec::new(), 0));
        }
    }

    fn assign_points_to_clusters(
        &self,
        point_indices: &[usize],
        centroids: &[(usize, usize)],
    ) -> Vec<Vec<usize>> {
        let num_centroids = centroids.len();

        let assignments = point_indices
            .par_iter()
            .map(|&point_idx| {
                let point = self.data.row(point_idx);
                let mut distances = Vec::with_capacity(num_centroids);

                // Pre-compute all distances
                for (idx, &(centroid_idx, _)) in centroids.iter().enumerate() {
                    let distance = self
                        .params
                        .distance_metric
                        .compute(&point, &self.data.row(centroid_idx).view());
                    distances.push((idx, distance));
                }

                let (best_cluster, min_distance) = distances.iter().fold(
                    (0, F::infinity()),
                    |(min_idx, min_dist), &(idx, dist)| {
                        if dist < min_dist {
                            (idx, dist)
                        } else {
                            (min_idx, min_dist)
                        }
                    },
                );

                let threshold = min_distance * F::from(Self::BOUNDARY_THRESHOLD).unwrap();
                let c1_idx = centroids[best_cluster].0;

                let mut point_assignments = vec![false; num_centroids];
                point_assignments[best_cluster] = true;

                for &(idx, dist) in &distances {
                    if idx != best_cluster && dist < threshold {
                        let c2_idx = centroids[idx].0;
                        let centroid_dist = self
                            .params
                            .distance_metric
                            .compute(&self.data.row(c1_idx).view(), &self.data.row(c2_idx).view());

                        if centroid_dist >= dist {
                            point_assignments[idx] = true;
                        }
                    }
                }

                // Return assignments for this point
                (point_idx, point_assignments)
            })
            .collect::<Vec<_>>();

        // Merge results into the final assignments vector
        let mut final_assignments = vec![Vec::new(); num_centroids];
        for (point_idx, point_assignments) in assignments {
            for (cluster_idx, &assigned) in point_assignments.iter().enumerate() {
                if assigned {
                    final_assignments[cluster_idx].push(point_idx);
                }
            }
        }

        final_assignments
    }

    /// Assigns each data point to the closest cluster centroid.
    /// Also handles "boundary" points.
    fn assign_points(&mut self) {
        let centroids: Vec<(usize, usize)> = self
            .clusters
            .iter()
            .map(|c| (c.centroid_idx.unwrap(), c.depth))
            .collect();

        let point_indices: Vec<usize> = (0..self.data.nrows()).collect();
        let assignments = self.assign_points_to_clusters(&point_indices, &centroids);

        // Update clusters with new assignments
        for (idx, points) in assignments.into_iter().enumerate() {
            if self.clusters[idx].points != points {
                debug!(
                    "Cluster {} size changed from {} to {} points",
                    idx,
                    self.clusters[idx].points.len(),
                    points.len()
                );
                self.clusters[idx].points = points;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use ndarray::array;
    use crate::clustering::{ClusteringParams, HierarchicalClustering, InitializationMethod};
    use crate::distances::SquaredEuclideanDistance;

    fn create_test_data() -> ndarray::Array2<f64> {
        array![[1.0, 2.0], [1.5, 2.5], [8.0, 8.0], [8.5, 8.5], [4.0, 4.0], [4.5, 4.5]]
    }

    #[test]
    fn test_initialize_clusters_randomly() {
        let data = create_test_data();
        let params = ClusteringParams {
            distance_metric: Arc::new(SquaredEuclideanDistance),
            initialization_method: InitializationMethod::Random,
            desired_cluster_size: Some(3),
            initial_k: 2,
            rng_seed: Some(42),
        };
        let mut clustering = HierarchicalClustering::<2, f64>::new(params, data.view());
        clustering.initialize_clusters(2);

        assert_eq!(clustering.clusters.len(), 2);
        for cluster in &clustering.clusters {
            assert!(cluster.centroid_idx.is_some());
        }

    }

    #[test]
    fn test_initialize_clusters_kmeans_plus_plus() {
        let data = create_test_data();
        let params = ClusteringParams {
            distance_metric: Arc::new(SquaredEuclideanDistance),
            initialization_method: InitializationMethod::KMeansPlusPlus,
            desired_cluster_size: Some(3),
            initial_k: 2,
            rng_seed: Some(42),
        };
        let mut clustering = HierarchicalClustering::<2, f64>::new(params, data.view());
        clustering.initialize_clusters(2);

        assert_eq!(clustering.clusters.len(), 2);
        for cluster in &clustering.clusters {
            assert!(cluster.centroid_idx.is_some());
        }
    }

    #[test]
    fn test_subdivide_clusters() {
        let data = create_test_data();
        let params = ClusteringParams {
            distance_metric: Arc::new(SquaredEuclideanDistance),
            initialization_method: InitializationMethod::Random,
            desired_cluster_size: Some(2),
            initial_k: 1,
            rng_seed: Some(42),
        };
        let mut clustering = HierarchicalClustering::<2, f64>::new(params, data.view());
        clustering.initialize_clusters(1);
        clustering.assign_points();
        clustering.update_centroids();
        clustering.subdivide_clusters();

        assert!(clustering.clusters.len() > 1);
        for cluster in &clustering.clusters {
            assert!(cluster.points.len() <= 2);
        }
    }

    #[test]
    fn test_assign_points() {
        let data = create_test_data();
        let params = ClusteringParams {
            distance_metric: Arc::new(SquaredEuclideanDistance),
            initialization_method: InitializationMethod::Random,
            desired_cluster_size: Some(3),
            initial_k: 2,
            rng_seed: Some(42),
        };
        let mut clustering = HierarchicalClustering::<2, f64>::new(params, data.view());
        clustering.initialize_clusters(2);
        clustering.assign_points();
        assert_eq!(clustering.clusters.iter().map(|c| c.points.len()).sum::<usize>(), data.nrows());
        for (i, cluster) in clustering.clusters.iter().enumerate() {
            assert!(
                !cluster.points.is_empty(),
                "Cluster {} is unexpectedly empty after point assignment.",
                i
            );
        }
    }

    #[test]
    fn test_fit() {
        let data = create_test_data();

        let params = ClusteringParams {
            distance_metric: Arc::new(SquaredEuclideanDistance),
            initialization_method: InitializationMethod::KMeansPlusPlus,
            desired_cluster_size: Some(2),
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

