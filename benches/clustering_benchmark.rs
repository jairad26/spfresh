use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use adriann::clustering::{ Config, SpannIndexBuilder};

/// Generate random data matrix of specified size
fn generate_random_data(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
}

/// Build SPANN index for given data
fn build_index<const D: usize>(data: &Array2<f32>) -> adriann::clustering::SpannIndex<D, f32> {
    let config: Config =
        Config::from_file("examples/config.yaml").expect("Failed to load configuration");

    SpannIndexBuilder::<f32>::new(config)
        .with_data(data.view())
        .build::<D>()
        .expect("Failed to build SPANN index")
}

fn bench_index_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("SPANN Index Build");

    // Test different dataset sizes
    let sizes = vec![
        (1_000, 128),    // Small dataset
        (10_000, 128),   // Medium dataset
        (100_000, 128),  // Large dataset
        (1_000_000, 128),// Very Large dataset
    ];

    for (num_points, dims) in sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", num_points, dims)),
            &(num_points, dims),
            |b, &(n, d)| {
                let data = generate_random_data(n, d, 42);
                b.iter(|| {
                    black_box(build_index::<128>(&data));
                });
            },
        );
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("SPANN Search");

    // Test different dataset sizes
    let sizes = vec![
        (1_000, 128, 10),     // Small dataset, k=10
        (10_000, 128, 10),    // Medium dataset, k=10
        (100_000, 128, 10),   // Large dataset, k=10
    ];

    for (num_points, dims, k) in sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}_k{}", num_points, dims, k)),
            &(num_points, dims, k),
            |b, &(n, d, k)| {
                // Setup: Generate data and build index
                let data = generate_random_data(n, d, 42);
                let index = build_index::<128>(&data);

                // Generate query points
                let query_points = generate_random_data(100, d, 43); // 100 query points

                b.iter(|| {
                    for query in query_points.rows() {
                        black_box(index.find_k_nearest_neighbor_spann(&query, k)
                            .expect("Search failed"));
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_index_build,
    bench_search
);
criterion_main!(benches);