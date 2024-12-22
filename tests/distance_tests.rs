#[cfg(test)]
mod tests {
    use adriann::clustering::{
        distance::{ManhattanDistance, SquaredEuclideanDistance},
        DistanceMetric,
    };
    use ndarray::array;
    use num_traits::Float;

    #[test]
    fn test_euclidean_distance() {
        let point1 = array![1.0, 2.0, 3.0];
        let point2 = array![4.0, 5.0, 6.0];
        let distance = SquaredEuclideanDistance.compute(&point1.view(), &point2.view());
        assert!((distance - 5.19615242).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let point1 = array![1.0, 2.0, 3.0];
        let point2 = array![4.0, 5.0, 6.0];
        let distance = ManhattanDistance.compute(&point1.view(), &point2.view());
        assert_eq!(distance, 9.0);
    }
}
