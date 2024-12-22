use std::fs::File;
use std::io::BufReader;

use adriann::clustering::config::Config;
use adriann::clustering::SpannIndexBuilder;
use log::info;
use ndarray::Array2;
use std::io::Read;

fn read_fvecs_as_array(file_path: &str) -> Array2<f32> {
    let file = File::open(file_path).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    let mut data: Vec<f32> = Vec::new();
    let mut row_count = 0;
    let mut dim = 0;

    loop {
        let mut dim_buffer = [0u8; 4];
        if reader.read_exact(&mut dim_buffer).is_err() {
            break; // End of file
        }

        // Read the dimension of the vector
        dim = i32::from_le_bytes(dim_buffer) as usize;
        let mut vector_buffer = vec![0u8; dim * 4];
        reader
            .read_exact(&mut vector_buffer)
            .expect("Failed to read vector data");

        // Convert raw bytes into f32 values
        let values: Vec<f32> = vector_buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        data.extend(values);
        row_count += 1;
    }

    Array2::from_shape_vec((row_count, dim), data).expect("Failed to create ndarray")
}

/// Reads ground truth neighbors from an ivecs file.
fn read_groundtruth(file_path: &str) -> Vec<Vec<usize>> {
    let file = File::open(file_path).expect("Failed to open ground truth file");
    let mut reader = BufReader::new(file);
    let mut groundtruth = Vec::new();

    loop {
        let mut dim_buffer = [0u8; 4];
        if reader.read_exact(&mut dim_buffer).is_err() {
            break; // End of file
        }
        let dim = i32::from_le_bytes(dim_buffer) as usize;

        let mut vector_buffer = vec![0u8; dim * 4];
        reader
            .read_exact(&mut vector_buffer)
            .expect("Failed to read ground truth data");

        let indices = vector_buffer
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as usize)
            .collect();

        groundtruth.push(indices);
    }

    groundtruth
}

/// Compares search results with ground truth for accuracy benchmarking.
fn compare_results(result: &[usize], groundtruth: &[usize]) {
    let hits = result.iter().filter(|&&r| groundtruth.contains(&r)).count();
    let precision = hits as f32 / result.len() as f32;
    info!("Precision: {:.2}%", precision * 100.0);
}

fn get_groundtruth_k(groundtruth: &[Vec<usize>]) -> usize {
    if let Some(first_row) = groundtruth.first() {
        first_row.len() // Assume all rows have the same number of neighbors
    } else {
        panic!("Ground truth is empty");
    }
}

fn main() {
    let data = read_fvecs_as_array("data/sift_small/siftsmall_base.fvecs");
    let query = read_fvecs_as_array("data/sift_small/siftsmall_query.fvecs");
    let config: Config =
        Config::from_file("examples/config.yaml").expect("Failed to load configuration");


    //let data = read_fvecs_as_array("data/sift/sift_base.fvecs");
    let spann_index = SpannIndexBuilder::<f32>::new(config)
        .with_data(data.view())
        .build::<128>()
        .expect("Failed to build SPANN index");
    /*

    let spann_index = SpannIndexBuilder::<f32>::new(config)
        .load::<128>()
        .expect("Failed to build SPANN index");
    */
    let groundtruth = read_groundtruth("data/sift_small/siftsmall_groundtruth.ivecs");
    let k = get_groundtruth_k(&groundtruth);
    info!("Searching for {} nearest neighbors...", k);
    // Benchmark nearest neighbor search
    for (i, query_vector) in query.rows().into_iter().enumerate() {
        let result = spann_index
            .find_k_nearest_neighbor_spann(&query_vector, k)
            .unwrap();
        // Compare with ground truth (if available)
        if let Some(gt) = groundtruth.get(i) {
            compare_results(&result, gt);
        }
    }
}
