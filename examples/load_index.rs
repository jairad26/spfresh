use ndarray::Array1;
use adriann::spann::config::Config;
use adriann::spann::spann_builder::SpannIndexBuilder;

fn main() {
    let config: Config =
        Config::from_file("examples/example_config.yaml").expect("Failed to load configuration");
    config.setup_logging();

    let spann_index = SpannIndexBuilder::<f32>::new(config)
        .load::<2>()
        .expect("Failed to build SPANN index");

    let k = 1;
    let query_vector = Array1::from(vec![1.0, 2.0]);
    let result = spann_index
        .find_k_nearest_neighbor_spann(&query_vector.view(), k)
        .unwrap();
    println!("Nearest neighbour: point_id:{:?} and vector:{:?}", result[0].point_id, result[0].vector);
}
