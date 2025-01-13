use fxhash::FxHashMap;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::{fs, io, path::PathBuf};

#[derive(Serialize, Deserialize)]
pub struct PointData<F> {
    pub point_id: usize,
    pub vector: Vec<F>,
}

pub trait PostingListStore<F>: Sized {
    /// Insert or update the posting list for a given `cluster_id`.
    fn insert_posting_list(
        &mut self,
        cluster_id: usize,
        vectors: Array2<F>,
        point_ids: Vec<usize>,
    ) -> io::Result<()>;
    fn get_posting_list(&self, cluster_id: usize) -> io::Result<Option<Vec<PointData<F>>>>;
    fn save_to_directory(&self) -> io::Result<()>;
    fn load_from_directory(dir_path: &str) -> io::Result<Self>;
}

pub struct FileBasedPostingListStore {
    base_directory: PathBuf,
    cluster_ids: FxHashMap<usize, ()>, // Track which cluster IDs exist
}

impl FileBasedPostingListStore {
    pub fn new(directory: &str) -> io::Result<Self> {
        let base_directory = PathBuf::from(directory);
        fs::create_dir_all(&base_directory)?;

        Ok(Self {
            base_directory,
            cluster_ids: FxHashMap::default(),
        })
    }

    fn get_posting_list_path(&self, cluster_id: usize) -> PathBuf {
        self.base_directory
            .join(format!("posting_list_{}.bin", cluster_id))
    }

    fn load_cluster_ids(directory: &PathBuf) -> io::Result<FxHashMap<usize, ()>> {
        let path = directory.join("cluster_ids.bin");
        if !path.exists() {
            return Ok(FxHashMap::default());
        }

        let bytes = fs::read(path)?;
        let cluster_ids: Vec<usize> =
            bincode::deserialize(&bytes).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(cluster_ids.into_iter().map(|id| (id, ())).collect())
    }
}

impl<F: Serialize + for<'de> Deserialize<'de> + Clone> PostingListStore<F>
    for FileBasedPostingListStore
{
    fn insert_posting_list(
        &mut self,
        cluster_id: usize,
        vectors: Array2<F>,
        point_ids: Vec<usize>,
    ) -> io::Result<()> {
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

        let file_path = self.get_posting_list_path(cluster_id);
        let encoded = bincode::serialize(&points)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        fs::write(&file_path, encoded)?;

        self.cluster_ids.insert(cluster_id, ());
        PostingListStore::<F>::save_to_directory(self)?;

        Ok(())
    }

    fn get_posting_list(&self, cluster_id: usize) -> io::Result<Option<Vec<PointData<F>>>> {
        if !self.cluster_ids.contains_key(&cluster_id) {
            return Ok(None);
        }
        let file_path = self.get_posting_list_path(cluster_id);
        let points = bincode::deserialize(&fs::read(&file_path)?)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        Ok(Some(points))
    }

    fn save_to_directory(&self) -> io::Result<()> {
        // Save cluster IDs in the new directory
        let encoded = bincode::serialize(&self.cluster_ids.keys().collect::<Vec<_>>())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(fs::write(
            self.base_directory.join("cluster_ids.bin"),
            encoded,
        )?)
    }

    fn load_from_directory(dir_path: &str) -> io::Result<Self> {
        let base_directory = PathBuf::from(dir_path);
        if !base_directory.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Directory does not exist",
            ));
        }

        let cluster_ids = Self::load_cluster_ids(&base_directory)?;
        Ok(Self {
            base_directory,
            cluster_ids,
        })
    }
}
