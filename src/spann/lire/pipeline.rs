use super::{LireError, LireResult, UpdateResult};
use super::operations::{PartitionOperation, Split, Merge, Reassign};
use crate::core::float::SpannFloat;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};
use std::thread::{self, JoinHandle};
use log::{debug, error, info};

/// Represents a background task in the pipeline
#[derive(Debug)]
pub enum BackgroundTask<F: SpannFloat> {
    Split(Box<Split<F>>),
    Merge(Box<Merge<F>>),
    Reassign(Box<Reassign<F>>),
    Shutdown,
}

/// Status of a partition
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PartitionStatus {
    Ready,
    Processing,
    NeedsMaintenance,
}

/// Two-stage pipeline for handling updates
pub struct TwoStagePipeline<F: SpannFloat + 'static> {
    background_sender: Sender<BackgroundTask<F>>,
    partition_statuses: Arc<Mutex<HashMap<usize, PartitionStatus>>>,
    background_thread: Option<JoinHandle<()>>,
    is_running: bool,
}

impl<F: SpannFloat> TwoStagePipeline<F> {
    pub fn new() -> Self {
        let (sender, receiver) = channel();
        let partition_statuses = Arc::new(Mutex::new(HashMap::new()));
        let mut pipeline = Self {
            background_sender: sender,
            partition_statuses: partition_statuses.clone(),
            background_thread: None,
            is_running: false,
        };
        
        pipeline.init_background_thread(receiver, partition_statuses);
        pipeline
    }

    fn init_background_thread(
        &mut self,
        receiver: Receiver<BackgroundTask<F>>,
        partition_statuses: Arc<Mutex<HashMap<usize, PartitionStatus>>>,
    ) {
        let thread = thread::spawn(move || {
            Self::background_worker(receiver, partition_statuses);
        });
        
        self.background_thread = Some(thread);
    }

    fn background_worker(
        receiver: Receiver<BackgroundTask<F>>,
        partition_statuses: Arc<Mutex<HashMap<usize, PartitionStatus>>>,
    ) {
        for task in receiver.iter() {
            match task {
                BackgroundTask::Shutdown => {
                    info!("Background worker received shutdown signal");
                    break;
                }
                BackgroundTask::Split(split) => {
                    Self::process_split(*split, &partition_statuses);
                }
                BackgroundTask::Merge(merge) => {
                    Self::process_merge(*merge, &partition_statuses);
                }
                BackgroundTask::Reassign(reassign) => {
                    Self::process_reassign(*reassign, &partition_statuses);
                }
            }
        }
    }

    fn process_split(
        split: Split<F>,
        partition_statuses: &Arc<Mutex<HashMap<usize, PartitionStatus>>>,
    ) {
        let affected_partitions = {
            let mut statuses = partition_statuses.lock().unwrap();
            // Mark source partition as processing
            statuses.insert(split.posting_id, PartitionStatus::Processing);
            split.get_affected_partitions()
        };

        match split.execute() {
            Ok(_) => {
                let mut statuses = partition_statuses.lock().unwrap();
                // Update statuses for all affected partitions
                for partition_id in affected_partitions {
                    statuses.insert(partition_id, PartitionStatus::Ready);
                }
                debug!("Split operation completed successfully");
            }
            Err(e) => {
                error!("Split operation failed: {}", e);
                let mut statuses = partition_statuses.lock().unwrap();
                // Mark affected partitions as needing maintenance
                for partition_id in affected_partitions {
                    statuses.insert(partition_id, PartitionStatus::NeedsMaintenance);
                }
            }
        }
    }

    fn process_merge(
        merge: Merge<F>,
        partition_statuses: &Arc<Mutex<HashMap<usize, PartitionStatus>>>,
    ) {
        let affected_partitions = {
            let mut statuses = partition_statuses.lock().unwrap();
            // Mark source partitions as processing
            statuses.insert(merge.posting_ids.0, PartitionStatus::Processing);
            statuses.insert(merge.posting_ids.1, PartitionStatus::Processing);
            merge.get_affected_partitions()
        };

        match merge.execute() {
            Ok(_) => {
                let mut statuses = partition_statuses.lock().unwrap();
                for partition_id in affected_partitions {
                    statuses.insert(partition_id, PartitionStatus::Ready);
                }
                debug!("Merge operation completed successfully");
            }
            Err(e) => {
                error!("Merge operation failed: {}", e);
                let mut statuses = partition_statuses.lock().unwrap();
                for partition_id in affected_partitions {
                    statuses.insert(partition_id, PartitionStatus::NeedsMaintenance);
                }
            }
        }
    }

    fn process_reassign(
        reassign: Reassign<F>,
        partition_statuses: &Arc<Mutex<HashMap<usize, PartitionStatus>>>,
    ) {
        let affected_partitions = {
            let mut statuses = partition_statuses.lock().unwrap();
            statuses.insert(reassign.from_posting, PartitionStatus::Processing);
            reassign.get_affected_partitions()
        };

        match reassign.execute() {
            Ok(_) => {
                let mut statuses = partition_statuses.lock().unwrap();
                for partition_id in affected_partitions {
                    statuses.insert(partition_id, PartitionStatus::Ready);
                }
                debug!("Reassign operation completed successfully");
            }
            Err(e) => {
                error!("Reassign operation failed: {}", e);
                let mut statuses = partition_statuses.lock().unwrap();
                for partition_id in affected_partitions {
                    statuses.insert(partition_id, PartitionStatus::NeedsMaintenance);
                }
            }
        }
    }

    pub fn submit_task(&self, task: BackgroundTask<F>) -> LireResult<()> {
        if !self.is_running {
            return Err(LireError::PipelineError("Pipeline not running".into()));
        }

        self.background_sender.send(task).map_err(|e| {
            LireError::PipelineError(format!("Failed to submit task: {}", e))
        })?;

        Ok(())
    }

    pub fn start(&mut self) -> LireResult<()> {
        if self.is_running {
            return Err(LireError::PipelineError("Pipeline already running".into()));
        }
        self.is_running = true;
        info!("Pipeline started");
        Ok(())
    }

    pub fn stop(&mut self) -> LireResult<()> {
        if !self.is_running {
            return Err(LireError::PipelineError("Pipeline not running".into()));
        }

        // Send shutdown signal
        self.background_sender
            .send(BackgroundTask::Shutdown)
            .map_err(|e| LireError::PipelineError(format!("Failed to send shutdown signal: {}", e)))?;

        // Wait for background thread to finish
        if let Some(thread) = self.background_thread.take() {
            thread.join().map_err(|_| {
                LireError::PipelineError("Failed to join background thread".into())
            })?;
        }

        self.is_running = false;
        info!("Pipeline stopped");
        Ok(())
    }

    pub fn get_partition_status(&self, partition_id: usize) -> Option<PartitionStatus> {
        self.partition_statuses
            .lock()
            .ok()
            .and_then(|statuses| statuses.get(&partition_id).copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distances::SquaredEuclideanDistance;
    use std::time::Duration;
    use std::thread::sleep;

    fn create_test_split() -> Split<f32> {
        let vectors = vec![
            (0, vec![1.0, 1.0]),
            (1, vec![1.5, 1.5]),
            (2, vec![8.0, 8.0]),
            (3, vec![8.5, 8.5]),
        ];

        Split::new(
            0,
            vectors,
            Box::new(SquaredEuclideanDistance),
            (1, 2),
        )
    }

    fn create_test_merge() -> Merge<f32> {
        let vectors1 = vec![
            (0, vec![1.0, 1.0]),
            (1, vec![1.5, 1.5]),
        ];
        let vectors2 = vec![
            (2, vec![2.0, 2.0]),
            (3, vec![2.5, 2.5]),
        ];

        Merge::new(
            (1, 2),
            vectors1,
            vectors2,
            Box::new(SquaredEuclideanDistance),
            3,
        )
    }

    fn create_test_reassign() -> Reassign<f32> {
        Reassign::new(
            0,
            vec![7.0, 7.0],
            0,
            vec![
                (1, vec![1.0, 1.0]),
                (2, vec![8.0, 8.0]),
                (3, vec![4.0, 4.0]),
            ],
            Box::new(SquaredEuclideanDistance),
            1,
        )
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = TwoStagePipeline::<f32>::new();
        assert!(!pipeline.is_running);
        assert!(pipeline.background_thread.is_some());
    }

    #[test]
    fn test_pipeline_start_stop() {
        let mut pipeline = TwoStagePipeline::<f32>::new();
        
        // Test starting
        assert!(pipeline.start().is_ok());
        assert!(pipeline.is_running);
        
        // Test double start
        assert!(pipeline.start().is_err());
        
        // Test stopping
        assert!(pipeline.stop().is_ok());
        assert!(!pipeline.is_running);
        
        // Test double stop
        assert!(pipeline.stop().is_err());
    }

    #[test]
    fn test_submit_split_task() {
        let mut pipeline = TwoStagePipeline::<f32>::new();
        pipeline.start().unwrap();

        let split = create_test_split();
        let task = BackgroundTask::Split(Box::new(split));
        
        assert!(pipeline.submit_task(task).is_ok());
        
        // Give some time for background processing
        sleep(Duration::from_millis(100));
        
        // Check partition statuses
        assert_eq!(
            pipeline.get_partition_status(1),
            Some(PartitionStatus::Ready)
        );
    }

    #[test]
    fn test_submit_merge_task() {
        let mut pipeline = TwoStagePipeline::<f32>::new();
        pipeline.start().unwrap();

        let merge = create_test_merge();
        let task = BackgroundTask::Merge(Box::new(merge));
        
        assert!(pipeline.submit_task(task).is_ok());
        
        // Give some time for background processing
        sleep(Duration::from_millis(100));
        
        // Check partition statuses
        assert_eq!(
            pipeline.get_partition_status(3),
            Some(PartitionStatus::Ready)
        );
    }

    #[test]
    fn test_submit_reassign_task() {
        let mut pipeline = TwoStagePipeline::<f32>::new();
        pipeline.start().unwrap();

        let reassign = create_test_reassign();
        let task = BackgroundTask::Reassign(Box::new(reassign));
        
        assert!(pipeline.submit_task(task).is_ok());
        
        // Give some time for background processing
        sleep(Duration::from_millis(100));
        
        // Check partition statuses
        assert_eq!(
            pipeline.get_partition_status(0),
            Some(PartitionStatus::Ready)
        );
    }

    #[test]
    fn test_partition_status_tracking() {
        let mut pipeline = TwoStagePipeline::<f32>::new();
        pipeline.start().unwrap();

        // Submit a split task
        let split = create_test_split();
        let affected_partitions = split.get_affected_partitions();
        let task = BackgroundTask::Split(Box::new(split));
        
        pipeline.submit_task(task).unwrap();
        
        // Give some time for processing
        sleep(Duration::from_millis(100));
        
        // Check all affected partitions are marked as Ready
        for partition_id in affected_partitions {
            assert_eq!(
                pipeline.get_partition_status(partition_id),
                Some(PartitionStatus::Ready)
            );
        }
    }

    #[test]
    fn test_submit_task_to_stopped_pipeline() {
        let pipeline = TwoStagePipeline::<f32>::new();
        let split = create_test_split();
        let task = BackgroundTask::Split(Box::new(split));
        
        // Should fail because pipeline isn't started
        assert!(pipeline.submit_task(task).is_err());
    }

    #[test]
    fn test_multiple_tasks() {
        let mut pipeline = TwoStagePipeline::<f32>::new();
        pipeline.start().unwrap();

        // Submit multiple tasks in sequence
        let split = create_test_split();
        let merge = create_test_merge();
        let reassign = create_test_reassign();

        pipeline.submit_task(BackgroundTask::Split(Box::new(split))).unwrap();
        pipeline.submit_task(BackgroundTask::Merge(Box::new(merge))).unwrap();
        pipeline.submit_task(BackgroundTask::Reassign(Box::new(reassign))).unwrap();

        // Give time for processing
        sleep(Duration::from_millis(300));

        // Verify pipeline is still operational
        assert!(pipeline.is_running);
        
        // Stop pipeline
        assert!(pipeline.stop().is_ok());
    }
}
