use adriann::clustering::SpannIndexBuilder;
use adriann::spann::config::Config;


fn main() {
    let config: Config =
        Config::from_file("examples/example_config.yaml").expect("Failed to load configuration");
    config.setup_logging();
    

    let spann_index = SpannIndexBuilder::<f32>::new(config)
        .with_data(data.view())
        .build::<128>()
        .expect("Failed to build SPANN index");

    let k = 10;
    let query_vector =
    let result = spann_index
            .find_k_nearest_neighbor_spann(&query_vector, k)
            .unwrap();
    println!("{:?}", result);

}

