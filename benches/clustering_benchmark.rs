use adriann::clustering::distance::SquaredEuclideanDistance;
use adriann::clustering::{Config, DistanceMetric, SpannIndexBuilder};
use std::fs::File;
use std::io::{BufReader, Read};

use criterion::{black_box, criterion_group, criterion_main, measurement::WallTime, Criterion};
use log::info;
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

fn generate_random_data(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);

    // Create a standard normal distribution
    let normal = StandardNormal;

    // Generate random data using array_init to avoid intermediate Vec allocation
    Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
}

fn benchmark_distance_computation(c: &mut Criterion) {
    let data = generate_random_data(1000, 10, 42);
    let point1 = data.row(0);
    let point2 = data.row(1);

    c.bench_function("distance_computation_euclidean", |b| {
        b.iter(|| {
            SquaredEuclideanDistance.compute(&point1, &point2);
        });
    });

    c.bench_function("distance_computation_manhattan", |b| {
        b.iter(|| {
            SquaredEuclideanDistance.compute(&point1, &point2);
        });
    });
}

/// Reads .fvecs file into an ndarray of shape (num_vectors, dim).
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

        // Dimension of the vector
        dim = i32::from_le_bytes(dim_buffer) as usize;

        let mut vector_buffer = vec![0u8; dim * 4];
        reader
            .read_exact(&mut vector_buffer)
            .expect("Failed to read vector data");

        let values: Vec<f32> = vector_buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        data.extend(values);
        row_count += 1;
    }

    Array2::from_shape_vec((row_count, dim), data).expect("Failed to create ndarray")
}

/// Reads ground-truth neighbors from an .ivecs file.
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

fn get_groundtruth_k(groundtruth: &[Vec<usize>]) -> usize {
    if let Some(first_row) = groundtruth.first() {
        first_row.len() // assume all rows have the same number of neighbors
    } else {
        panic!("Ground truth is empty");
    }
}

fn compare_results(result: &[usize], groundtruth: &[usize]) {
    let hits = result.iter().filter(|&&r| groundtruth.contains(&r)).count();
    let precision = hits as f32 / result.len() as f32;
    info!("Precision: {:.2}%", precision * 100.0);
}

fn bench_spann_build_index(c: &mut Criterion) {
    // Load the dataset once outside the benchmark loop
    let data = read_fvecs_as_array("data/sift_small/siftsmall_base.fvecs");

    c.bench_function("SPANN index build", |b| {
        b.iter(|| {
            // Build the index on each iteration
            let spann_index = SpannIndexBuilder::<f32>::new(
                Config::from_file("examples/config.yaml").expect("Failed to load configuration"),
            )
            .with_data(data.view())
            .build::<128>()
            .expect("Failed to build SPANN index");

            // Use black_box so the compiler doesn't optimize away creation
            black_box(spann_index);
        });
    });
}

fn setup_spann_for_search() -> (
    Array2<f32>,
    Vec<Vec<usize>>,
    usize,
    adriann::clustering::SpannIndex<128, f32>,
) {
    // Prepare query data
    let query = read_fvecs_as_array("data/sift_small/siftsmall_query.fvecs");

    // Prepare groundtruth
    let groundtruth = read_groundtruth("data/sift_small/siftsmall_groundtruth.ivecs");
    let k = get_groundtruth_k(&groundtruth);

    // Load config
    let config: Config =
        Config::from_file("examples/config.yaml").expect("Failed to load configuration");

    // Either build or load your index. Here we just do .load() for illustration:
    let spann_index = SpannIndexBuilder::<f32>::new(config)
        .load::<128>()
        .expect("Failed to load SPANN index");

    (query, groundtruth, k, spann_index)
}

/// Criterion benchmark for SPANN nearest neighbor search.
fn bench_spann_search(c: &mut Criterion) {
    // Setup phase (I/O and index loading) outside the benchmarking loop
    let (query, groundtruth, k, spann_index) = setup_spann_for_search();

    // We wrap our benchmark in a closure so that Criterion can measure it.
    c.bench_function("SPANN k-NN search (all queries)", |b| {
        b.iter(|| {
            for (i, query_vector) in query.rows().into_iter().enumerate() {
                let result = spann_index
                    .find_k_nearest_neighbor_spann(black_box(&query_vector), black_box(k))
                    .unwrap();

                // (Optional) Compare with ground truth
                if let Some(gt) = groundtruth.get(i) {
                    compare_results(&result, gt);
                }
            }
        });
    });
}

fn criterion_config() -> Criterion<WallTime> {
    Criterion::default().measurement_time(std::time::Duration::new(60, 0))
}

criterion_group!(
    name = benches;
    config = criterion_config();
    targets = benchmark_distance_computation, bench_spann_build_index, bench_spann_search
);
criterion_main!(benches);
