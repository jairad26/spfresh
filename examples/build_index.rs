use ndarray::{Array1, Array2};
use adriann::spann::config::Config;
use adriann::spann::spann_builder::SpannIndexBuilder;

fn main() {
    let config: Config =
        Config::from_file("examples/example_config.yaml").expect("Failed to load configuration");

    let data = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 1.5, 2.5, 8.0, 8.0, 8.5, 8.5, 4.0, 4.0, 4.5, 4.5],
    ).unwrap();

    let spann_index = SpannIndexBuilder::<f32>::new(config)
        .with_data(data.view())
        .build::<2>()
        .expect("Failed to build SPANN index");

    let k = 1;
    let query_vector = Array1::from(vec![1.0, 2.0]);
    let result = spann_index
            .find_k_nearest_neighbor_spann(&query_vector.view(), k)
            .unwrap();
    println!("{:?}", result);
    // [PointData { point_id: 0, vector: [1.0, 2.0] }]
}

