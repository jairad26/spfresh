use ndarray::Array2;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::{fs, io};
use fxhash::FxHashMap;

#[derive(Serialize, Deserialize)]
pub struct PointData<F: Float> {
    pub point_id: usize,
    pub vector: Vec<F>,
}

pub trait PostingListStore<F: Float>: Sized {
    /// Insert or update the posting list for a given `cluster_id`.
    fn insert_posting_list(&mut self, cluster_id: usize, vectors: Array2<F>, point_ids: Vec<usize>);

    /// Retrieve a reference to the posting list for `cluster_id`.
    fn get_posting_list(&self, cluster_id: usize) -> Option<&[PointData<F>]>;

    /// Save the entire posting-list data structure to a binary file.
    fn save_to_file(&self, file_path: &str) -> io::Result<()>;

    /// Load a new `Self` from a binary file.
    fn load_from_file(file_path: &str) -> io::Result<Self>
    where
        Self: Sized;
}

#[derive(Serialize, Deserialize)]
pub struct InMemoryPostingListStore<F: Float> {
    data: FxHashMap<usize, Vec<PointData<F>>>,
}

impl<F: Float> Default for InMemoryPostingListStore<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> InMemoryPostingListStore<F> {
    /// Create a new, empty in-memory store.
    pub fn new() -> Self {
        Self {
            data: FxHashMap::default(),
        }
    }
}

impl<F: Float + for<'de> Deserialize<'de> + Serialize> PostingListStore<F>
    for InMemoryPostingListStore<F>
{
    fn insert_posting_list(
        &mut self,
        cluster_id: usize,
        vectors: Array2<F>,
        point_ids: Vec<usize>,
    ) {
        assert_eq!(
            vectors.nrows(),
            point_ids.len(),
            "Number of vectors must match number of IDs"
        );

        let points: Vec<PointData<F>> = vectors
            .rows()
            .into_iter()
            .zip(point_ids)
            .map(|(row, id)| PointData {
                point_id: id,
                vector: row.to_vec(),
            })
            .collect();

        self.data.insert(cluster_id, points);
    }

    fn get_posting_list(&self, cluster_id: usize) -> Option<&[PointData<F>]> {
        self.data.get(&cluster_id).map(|v| v.as_slice())
    }

    fn save_to_file(&self, file_path: &str) -> io::Result<()> {
        let encoded =
            bincode::serialize(self).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(file_path, encoded)
    }

    fn load_from_file(file_path: &str) -> io::Result<Self> {
        let bytes = fs::read(file_path)?;
        let store: Self =
            bincode::deserialize(&bytes).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(store)
    }
}
