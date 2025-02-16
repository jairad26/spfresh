use super::{LireError, LireResult};
use crate::core::float::SpannFloat;
use crate::spann::posting_lists::PointData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::collections::HashMap;
use std::fs;
use std::io;
use serde::{Deserialize, Serialize};
use std::thread;
use std::time::Duration;
use std::sync::Arc;
use ndarray::Array1;

/// Represents a versioned vector in storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedVector<F> {
    pub vector: Vec<F>,
    pub version: u64,
    pub is_deleted: bool,
}

/// Represents a posting's metadata
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PostingMetadata {
    pub version: u64,
    pub vector_count: usize,
    pub centroid: Vec<f32>,
}

/// SSD-backed storage engine for LIRE
pub struct LireStorage {
    base_path: PathBuf,
    current_version: AtomicU64,
    postings_metadata: RwLock<HashMap<usize, PostingMetadata>>,
}

impl LireStorage {
    pub fn new<P: Into<PathBuf>>(path: P) -> io::Result<Self> {
        let base_path = path.into();
        fs::create_dir_all(&base_path)?;
        fs::create_dir_all(base_path.join("postings"))?;
        fs::create_dir_all(base_path.join("metadata"))?;

        // Load existing metadata
        let mut metadata = HashMap::new();
        let metadata_dir = base_path.join("metadata");
        if metadata_dir.exists() {
            for entry in fs::read_dir(metadata_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    if let Some(posting_id) = entry.file_name().to_str()
                        .and_then(|s| s.strip_suffix("_meta.bin"))
                        .and_then(|s| s.strip_prefix("posting_"))
                        .and_then(|s| s.parse::<usize>().ok()) 
                    {
                        let data = fs::read(entry.path())?;
                        if let Ok(posting_metadata) = bincode::deserialize(&data) {
                            metadata.insert(posting_id, posting_metadata);
                        }
                    }
                }
            }
        }

        Ok(Self {
            base_path,
            current_version: AtomicU64::new(1),
            postings_metadata: RwLock::new(metadata),
        })
    }

    /// Get the path for a specific posting's data
    fn get_posting_path(&self, posting_id: usize) -> PathBuf {
        self.base_path.join("postings").join(format!("posting_{}.bin", posting_id))
    }

    /// Get the path for a specific posting's metadata
    fn get_metadata_path(&self, posting_id: usize) -> PathBuf {
        self.base_path.join("metadata").join(format!("posting_{}_meta.bin", posting_id))
    }

    /// Store a vector in a posting with versioning
    pub fn store_vector<F: SpannFloat + Serialize>(
        &self,
        posting_id: usize,
        vector_id: usize,
        vector: Vec<F>,
    ) -> LireResult<u64> {
        let version = self.current_version.fetch_add(1, Ordering::SeqCst);
        let versioned_vector = VersionedVector {
            vector: vector.clone(),
            version,
            is_deleted: false,
        };

        // Update posting metadata
        let mut metadata = self.postings_metadata.write().map_err(|e| 
            LireError::StorageError(format!("Failed to acquire metadata lock: {}", e))
        )?;

        let posting_metadata = metadata.entry(posting_id).or_insert(PostingMetadata {
            version,
            vector_count: 0,
            centroid: vector.iter().map(|x| x.to_f32().unwrap()).collect(),
        });
        posting_metadata.vector_count += 1;
        posting_metadata.version = version;

        // Serialize and store the vector
        let path = self.get_posting_path(posting_id);
        let encoded = bincode::serialize(&versioned_vector)
            .map_err(|e| LireError::StorageError(format!("Serialization failed: {}", e)))?;

        fs::write(&path, encoded)
            .map_err(|e| LireError::StorageError(format!("Failed to write vector: {}", e)))?;

        // Save metadata
        self.save_metadata(posting_id, posting_metadata)?;

        Ok(version)
    }

    /// Mark a vector as deleted
    pub fn mark_deleted(
        &self,
        posting_id: usize,
        vector_id: usize,
    ) -> LireResult<()> {
        let path = self.get_posting_path(posting_id);
        if !path.exists() {
            return Err(LireError::StorageError(format!(
                "Posting {} does not exist",
                posting_id
            )));
        }

        // Read and update vectors
        let data = fs::read(&path)
            .map_err(|e| LireError::StorageError(format!("Failed to read posting: {}", e)))?;
        let mut vectors: HashMap<usize, VersionedVector<f32>> = bincode::deserialize(&data)
            .map_err(|e| LireError::StorageError(format!("Failed to deserialize vectors: {}", e)))?;

        // Mark vector as deleted
        if let Some(vector) = vectors.get_mut(&vector_id) {
            vector.is_deleted = true;
        } else {
            return Err(LireError::StorageError(format!(
                "Vector {} not found in posting {}",
                vector_id, posting_id
            )));
        }

        // Save updated vectors
        let encoded = bincode::serialize(&vectors)
            .map_err(|e| LireError::StorageError(format!("Failed to serialize vectors: {}", e)))?;
        fs::write(&path, encoded)
            .map_err(|e| LireError::StorageError(format!("Failed to write vectors: {}", e)))?;

        // Update metadata
        let mut metadata = self.postings_metadata.write().map_err(|e|
            LireError::StorageError(format!("Failed to acquire metadata lock: {}", e))
        )?;

        if let Some(posting_metadata) = metadata.get_mut(&posting_id) {
            posting_metadata.vector_count -= 1;
            posting_metadata.version = self.current_version.fetch_add(1, Ordering::SeqCst);
            self.save_metadata(posting_id, posting_metadata)?;
        }

        Ok(())
    }

    /// Save posting metadata to disk
    fn save_metadata(&self, posting_id: usize, metadata: &PostingMetadata) -> LireResult<()> {
        let path = self.get_metadata_path(posting_id);
        let encoded = bincode::serialize(metadata)
            .map_err(|e| LireError::StorageError(format!("Failed to serialize metadata: {}", e)))?;

        fs::write(&path, encoded)
            .map_err(|e| LireError::StorageError(format!("Failed to write metadata: {}", e)))?;

        Ok(())
    }

    /// Get the current version of a posting
    pub fn get_posting_version(&self, posting_id: usize) -> LireResult<u64> {
        let metadata = self.postings_metadata.read().map_err(|e|
            LireError::StorageError(format!("Failed to acquire metadata lock: {}", e))
        )?;

        metadata.get(&posting_id)
            .map(|m| m.version)
            .ok_or_else(|| LireError::StorageError(format!("Posting {} not found", posting_id)))
    }

    /// Check if a posting needs garbage collection
    pub fn needs_garbage_collection(&self, posting_id: usize, threshold: f64) -> LireResult<bool> {
        let metadata = self.postings_metadata.read().map_err(|e|
            LireError::StorageError(format!("Failed to acquire metadata lock: {}", e))
        )?;

        if let Some(metadata) = metadata.get(&posting_id) {
            // Count deleted vectors
            let path = self.get_posting_path(posting_id);
            let data = fs::read(&path)
                .map_err(|e| LireError::StorageError(format!("Failed to read posting: {}", e)))?;
            
            let vectors: HashMap<usize, VersionedVector<f32>> = bincode::deserialize(&data)
                .map_err(|e| LireError::StorageError(format!("Failed to deserialize vectors: {}", e)))?;
            
            let deleted_count = vectors.values().filter(|v| v.is_deleted).count();
            let total_count = vectors.len();
            
            if total_count == 0 {
                Ok(false)
            } else {
                let deletion_ratio = deleted_count as f64 / total_count as f64;
                Ok(deletion_ratio > threshold)
            }
        } else {
            Ok(false)
        }
    }

    /// Get metadata for a specific posting
    pub fn get_posting_metadata(&self, posting_id: usize) -> LireResult<PostingMetadata> {
        let metadata = self.postings_metadata.read().map_err(|e| 
            LireError::StorageError(format!("Failed to acquire metadata lock: {}", e))
        )?;

        metadata.get(&posting_id)
            .cloned()
            .ok_or_else(|| LireError::StorageError(format!("Posting {} not found", posting_id)))
    }

    /// Get the centroid for a specific posting
    pub fn get_posting_centroid<F: SpannFloat>(&self, posting_id: usize) -> LireResult<Array1<F>> {
        let metadata = self.get_posting_metadata(posting_id)?;
        Ok(Array1::from_vec(metadata.centroid.iter().map(|&x| F::from_f32(x).unwrap()).collect()))
    }

    /// Update the centroid for a specific posting
    pub fn update_posting_centroid(&self, posting_id: usize, centroid: Vec<f32>) -> LireResult<()> {
        let mut metadata = self.postings_metadata.write().map_err(|e| 
            LireError::StorageError(format!("Failed to acquire metadata lock: {}", e))
        )?;

        if let Some(posting_metadata) = metadata.get_mut(&posting_id) {
            posting_metadata.centroid = centroid;
            posting_metadata.version = self.current_version.fetch_add(1, Ordering::SeqCst);
            self.save_metadata(posting_id, posting_metadata)?;
        } else {
            return Err(LireError::StorageError(format!("Posting {} not found", posting_id)));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_storage() -> (LireStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = LireStorage::new(temp_dir.path()).unwrap();
        (storage, temp_dir)
    }

    #[test]
    fn test_basic_vector_storage() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store a vector
        let vector = vec![1.0f32, 2.0, 3.0];
        let version = storage.store_vector(0, 1, vector.clone()).unwrap();
        
        // Verify version is greater than 0
        assert!(version > 0);
        
        // Verify posting version matches
        let posting_version = storage.get_posting_version(0).unwrap();
        assert_eq!(posting_version, version);
    }

    #[test]
    fn test_version_increment() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store multiple vectors and verify versions increment
        let vector = vec![1.0f32, 2.0, 3.0];
        let version1 = storage.store_vector(0, 1, vector.clone()).unwrap();
        let version2 = storage.store_vector(0, 2, vector.clone()).unwrap();
        let version3 = storage.store_vector(0, 3, vector.clone()).unwrap();
        
        assert!(version2 > version1);
        assert!(version3 > version2);
    }

    #[test]
    fn test_deletion() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store and then delete a vector
        let vector = vec![1.0f32, 2.0, 3.0];
        storage.store_vector(0, 1, vector).unwrap();
        
        // Mark as deleted
        storage.mark_deleted(0, 1).unwrap();
        
        // Verify metadata is updated
        let metadata = storage.postings_metadata.read().unwrap();
        let posting_metadata = metadata.get(&0).unwrap();
        assert_eq!(posting_metadata.vector_count, 0);
    }

    #[test]
    fn test_garbage_collection_threshold() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store multiple vectors
        let vector = vec![1.0f32, 2.0, 3.0];
        storage.store_vector(0, 1, vector.clone()).unwrap();
        storage.store_vector(0, 2, vector.clone()).unwrap();
        storage.store_vector(0, 3, vector.clone()).unwrap();
        
        // Delete some vectors
        storage.mark_deleted(0, 1).unwrap();
        storage.mark_deleted(0, 2).unwrap();
        
        // Check garbage collection threshold
        assert!(storage.needs_garbage_collection(0, 0.5).unwrap());
        assert!(!storage.needs_garbage_collection(0, 0.9).unwrap());
    }

    #[test]
    fn test_concurrent_access() {
        let (storage, _temp_dir) = create_test_storage();
        let storage = Arc::new(storage);
        
        let mut handles = vec![];
        
        // Spawn multiple threads to store vectors
        for i in 0..10 {
            let storage_clone = Arc::clone(&storage);
            let handle = thread::spawn(move || {
                let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
                storage_clone.store_vector(0, i, vector).unwrap()
            });
            handles.push(handle);
        }
        
        // Collect all versions
        let versions: Vec<u64> = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // Verify all versions are unique
        let mut unique_versions = versions.clone();
        unique_versions.sort();
        unique_versions.dedup();
        assert_eq!(versions.len(), unique_versions.len());
    }

    #[test]
    fn test_error_handling() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Test deleting non-existent posting
        let result = storage.mark_deleted(999, 1);
        assert!(matches!(result, Err(LireError::StorageError(_))));
        
        // Test getting version of non-existent posting
        let result = storage.get_posting_version(999);
        assert!(matches!(result, Err(LireError::StorageError(_))));
    }

    #[test]
    fn test_metadata_persistence() {
        let (storage, temp_dir) = create_test_storage();
        
        // Store a vector
        let vector = vec![1.0f32, 2.0, 3.0];
        let version = storage.store_vector(0, 1, vector).unwrap();
        
        // Create new storage instance pointing to same directory
        let storage2 = LireStorage::new(temp_dir.path()).unwrap();
        
        // Verify metadata is loaded correctly
        let posting_version = storage2.get_posting_version(0).unwrap();
        assert_eq!(posting_version, version);
    }

    #[test]
    fn test_multiple_postings() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store vectors in different postings
        let vector = vec![1.0f32, 2.0, 3.0];
        storage.store_vector(0, 1, vector.clone()).unwrap();
        storage.store_vector(1, 1, vector.clone()).unwrap();
        storage.store_vector(2, 1, vector.clone()).unwrap();
        
        // Verify each posting has correct metadata
        let metadata = storage.postings_metadata.read().unwrap();
        assert_eq!(metadata.len(), 3);
        for i in 0..3 {
            assert!(metadata.contains_key(&i));
            assert_eq!(metadata.get(&i).unwrap().vector_count, 1);
        }
    }

    #[test]
    fn test_posting_metadata() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store a vector
        let vector = vec![1.0f32, 2.0, 3.0];
        storage.store_vector(0, 1, vector.clone()).unwrap();
        
        // Get metadata
        let metadata = storage.get_posting_metadata(0).unwrap();
        assert_eq!(metadata.vector_count, 1);
        assert_eq!(metadata.centroid, vector);
    }

    #[test]
    fn test_centroid_update() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store initial vector
        let vector1 = vec![1.0f32, 2.0, 3.0];
        storage.store_vector(0, 1, vector1).unwrap();
        
        // Update centroid
        let new_centroid = vec![2.0f32, 3.0, 4.0];
        storage.update_posting_centroid(0, new_centroid.clone()).unwrap();
        
        // Verify centroid update
        let centroid = storage.get_posting_centroid::<f32>(0).unwrap();
        assert_eq!(centroid.as_slice().unwrap(), new_centroid.as_slice());
    }

    #[test]
    fn test_invalid_posting_metadata() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Try to get metadata for non-existent posting
        let result = storage.get_posting_metadata(999);
        assert!(matches!(result, Err(LireError::StorageError(_))));
    }
}
