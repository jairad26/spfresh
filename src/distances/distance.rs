use crate::core::float::SpannFloat;
use ndarray::ArrayView1;
use ndarray_stats::DeviationExt;
use std::fmt::Debug;

/// Trait defining the interface for distance metrics
pub trait DistanceMetric<F: SpannFloat>: Send + Sync {
    /// Computes the distance between two points. Panics if the points have different dimensions.
    fn compute(&self, point1: &ArrayView1<F>, point2: &ArrayView1<F>) -> F;
}

/// [Squared Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
#[derive(Debug, Clone, Copy)]
pub struct SquaredEuclideanDistance;

impl<F: SpannFloat> DistanceMetric<F> for SquaredEuclideanDistance {
    #[inline]
    fn compute(&self, point1: &ArrayView1<F>, point2: &ArrayView1<F>) -> F {
        point1.sq_l2_dist(point2).unwrap()
    }
}

/// [Manhattan Distance](https://en.wikipedia.org/wiki/Taxicab_geometry)
#[derive(Debug, Clone, Copy)]
pub struct ManhattanDistance;

impl<F: SpannFloat> DistanceMetric<F> for ManhattanDistance {
    #[inline]
    fn compute(&self, point1: &ArrayView1<F>, point2: &ArrayView1<F>) -> F {
        point1.l1_dist(point2).unwrap()
    }
}

/// [Chebyshev Distance](https://en.wikipedia.org/wiki/Chebyshev_distance)
#[derive(Debug, Clone, Copy)]
pub struct ChebyshevDistance;

impl<F: SpannFloat> DistanceMetric<F> for ChebyshevDistance {
    #[inline]
    fn compute(&self, point1: &ArrayView1<F>, point2: &ArrayView1<F>) -> F {
        point1.linf_dist(point2).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use num_traits::ToPrimitive;
    use crate::distances::{ChebyshevDistance, DistanceMetric, ManhattanDistance, SquaredEuclideanDistance};

    #[test]
    fn test_squared_euclidean_distance() {
        let point1 = array![1.0, 2.0, 3.0];
        let point2 = array![4.0, 5.0, 6.0];
        let metric = SquaredEuclideanDistance;

        let result = metric.compute(&point1.view(), &point2.view());
        let expected = 27.0; // (4-1)^2 + (5-2)^2 + (6-3)^2

        assert!((result - expected).to_f64().unwrap().abs() < 1e-6, "Expected {}, got {}", expected, result);
    }

    #[test]
    fn test_manhattan_distance() {
        let point1 = array![1.0, 2.0, 3.0];
        let point2 = array![4.0, 5.0, 6.0];
        let metric = ManhattanDistance;

        let result = metric.compute(&point1.view(), &point2.view());
        let expected = 9.0; // |4-1| + |5-2| + |6-3|

        assert!((result - expected).to_f64().unwrap().abs() < 1e-6, "Expected {}, got {}", expected, result);
    }

    #[test]
    fn test_chebyshev_distance() {
        let point1 = array![1.0, 2.0, 3.0];
        let point2 = array![4.0, 5.0, 6.0];
        let metric = ChebyshevDistance;

        let result = metric.compute(&point1.view(), &point2.view());
        let expected = 3.0; // max(|4-1|, |5-2|, |6-3|)

        assert!((result - expected).to_f64().unwrap().abs() < 1e-6, "Expected {}, got {}", expected, result);
    }

    #[test]
    fn test_zero_distance() {
        let point1 = array![1.0, 2.0, 3.0];
        let point2 = array![1.0, 2.0, 3.0];

        let metrics: Vec<Box<dyn DistanceMetric<f64>>> = vec![
            Box::new(SquaredEuclideanDistance),
            Box::new(ManhattanDistance),
            Box::new(ChebyshevDistance),
        ];

        for metric in metrics {
            let result = metric.compute(&point1.view(), &point2.view());
            let expected = 0.0;

            assert!((result - expected).abs() < 1e-6, "Expected {}, got {}", expected, result);
        }
    }
}
