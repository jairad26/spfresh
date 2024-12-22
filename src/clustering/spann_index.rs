use std::io;
use std::iter::Sum;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use kiddo::KdTree;
use log::error;
use num_traits::float::FloatCore;
use num_traits::{Float, FromPrimitive, Signed};
use rand_distr::uniform::SampleUniform;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs::File;
use std::ops::AddAssign;

use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::clustering::distance::{DistanceMetric, SquaredEuclideanDistance};
use crate::clustering::posting_lists::{InMemoryPostingListStore, PostingListStore};
use crate::clustering::Cluster;

pub struct SpannIndex<const N: usize, F: Float>
where
    F: Float
        + Debug
        + Default
        + Sum
        + AddAssign
        + SampleUniform
        + Serialize
        + FloatCore
        + Sync
        + Signed
        + Send
        + for<'de> Deserialize<'de>
        + FromPrimitive,
{
    pub kdtree: Option<KdTree<F, N>>,
    pub posting_list_store: Option<InMemoryPostingListStore<F>>,
}

impl<const N: usize, F: Float> Default for SpannIndex<N, F>
where
    F: Float
        + Sum
        + SampleUniform
        + Serialize
        + FloatCore
        + Default
        + Debug
        + Sync
        + Send
        + AddAssign
        + Signed
        + for<'de> Deserialize<'de>
        + ndarray::ScalarOperand
        + FromPrimitive,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, F: Float> SpannIndex<N, F>
where
    F: Float
        + Debug
        + Default
        + AddAssign
        + Sum
        + SampleUniform
        + Serialize
        + for<'de> Deserialize<'de>
        + FromPrimitive
        + FloatCore
        + Copy
        + Sync
        + Signed
        + Send,
{
    pub fn new() -> Self {
        Self {
            kdtree: None,
            posting_list_store: None,
        }
    }

    pub fn load_posting_list(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        match PostingListStore::load_from_file(path) {
            Ok(posting_list_store) => {
                self.posting_list_store = Some(posting_list_store);
                Ok(())
            }
            Err(e) => {
                eprintln!("Failed to load posting list from {}: {}", path, e);
                Err(Box::new(e))
            }
        }
    }

    pub fn save_posting_list(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(posting_list_store) = &self.posting_list_store {
            posting_list_store.save_to_file(path)?;
            Ok(())
        } else {
            eprintln!("Posting list is not available");
            Err("Posting list is not available".into())
        }
    }

    /// Create a memory-based posting list from the given clusters.
    pub fn create_posting_lists(
        &mut self,
        data: &ArrayView2<F>,
        clusters: &[Cluster],
    ) -> Option<&InMemoryPostingListStore<F>> {
        let mut posting_list_store = InMemoryPostingListStore::<F>::new();

        for (cluster_id, cluster) in clusters.iter().enumerate() {
            let num_points = cluster.points.len();
            let num_features = data.ncols();
            let mut points_array = Array2::zeros((num_points, num_features));
            let point_ids: Vec<usize> = cluster.points.clone();
            for (i, &point_idx) in cluster.points.iter().enumerate() {
                points_array.row_mut(i).assign(&data.row(point_idx));
            }
            // change to view
            posting_list_store.insert_posting_list(cluster_id, points_array, point_ids);
        }
        self.posting_list_store = Some(posting_list_store);
        self.posting_list_store.as_ref()
    }

    /// Builds the KD-tree from the centroids of the given `clusters`.
    /// Returns the resulting KdTree, also stored in self.
    pub fn build_kdtree(
        &mut self,
        data: &ArrayView2<F>,
        clusters: &[Cluster],
    ) -> Option<&KdTree<F, N>> {
        if data.ncols() != N {
            error!("Data column count does not match N={}", N);
            return None;
        }

        let mut tree: KdTree<F, N> = KdTree::new();
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            if let Some(centroid_idx) = cluster.centroid_idx {
                let row = data.row(centroid_idx);
                let array_slice = row.as_slice().expect("Data row mismatch");
                let array: [F; N] = array_slice.try_into().expect("Length mismatch");
                tree.add(&array, cluster_id as u64);
            }
        }
        self.kdtree = Some(tree);
        self.kdtree.as_ref()
    }

    pub fn save_kdtree(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let encoder = GzEncoder::new(file, Compression::default());
        if let Some(tree) = &self.kdtree {
            match bincode::serialize_into(encoder, tree) {
                Ok(_) => Ok(()),
                Err(e) => {
                    eprintln!("Failed to serialize KD-Tree to {}: {}", path, e);
                    Err(Box::new(e))
                }
            }
        } else {
            eprintln!("KD-Tree is not available");
            Err("KD-Tree is not available".into())
        }
    }

    pub fn load_kdtree(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let decompressor = GzDecoder::new(file);
        match bincode::deserialize_from(decompressor) {
            Ok(tree) => {
                self.kdtree = Some(tree);
                Ok(())
            }
            Err(e) => {
                eprintln!("Failed to deserialize KD-Tree from {}: {}", path, e);
                Err(Box::new(e))
            }
        }
    }

    /// Saves the KD-tree and posting lists to disk.
    pub fn store_index(
        &mut self,
        kdtree_path: &str,
        posting_list_path: &str,
        data: &ArrayView2<F>,
        clusters: &[Cluster],
    ) -> io::Result<()> {
        self.build_kdtree(data, clusters);
        self.create_posting_lists(data, clusters)
            .expect("Error creating posting lists");
        self.save_kdtree(kdtree_path).expect("Error saving KD-Tree");
        self.save_posting_list(posting_list_path)
            .expect("Error saving posting list");
        Ok(())
    }

    pub fn find_k_nearest_neighbor_spann(
        &self,
        query: &ArrayView1<F>,
        k: usize,
    ) -> Option<Vec<usize>> {
        let tree = self.kdtree.as_ref().expect("KD-Tree is not available");
        let posting_list_store = self
            .posting_list_store
            .as_ref()
            .expect("Posting list is not available");

        let query_array: [F; N] = query
            .as_slice()
            .expect("Query length mismatch")
            .try_into()
            .expect("Query length mismatch");
        let nearest_centroids = tree.nearest_n::<kiddo::SquaredEuclidean>(&query_array, k);

        let mut all_candidates: Vec<(F, usize)> = Vec::new();
        for nn in nearest_centroids {
            if let Some(points) = posting_list_store.get_posting_list(nn.item as usize) {
                for point_data in points {
                    let point_vector = ArrayView1::from(&point_data.vector);
                    let dist = SquaredEuclideanDistance.compute(&query.view(), &point_vector);
                    all_candidates.push((dist, point_data.point_id));
                }
            }
        }

        if all_candidates.is_empty() {
            error!("No candidate points found for nearest centroids.");
            return None;
        }

        all_candidates
            .sort_by(|(d1, _), (d2, _)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal));

        if all_candidates.len() > k {
            all_candidates.truncate(k);
        }
        let nearest_points = all_candidates.into_iter().map(|(_, point)| point).collect();

        Some(nearest_points)
    }
}
