use crate::clustering::float::AdriannFloat;
use ndarray::ArrayView1;
use ndarray_stats::DeviationExt;
use std::fmt::Debug;

/// Trait defining the interface for distance metrics
pub trait DistanceMetric<F: AdriannFloat>: Send + Sync {
    /// Computes the distance between two points. Panics if the points have different dimensions.
    fn compute(&self, point1: &ArrayView1<F>, point2: &ArrayView1<F>) -> F;
}

/// [Squared Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
#[derive(Debug, Clone, Copy)]
pub struct SquaredEuclideanDistance;

impl<F: AdriannFloat> DistanceMetric<F> for SquaredEuclideanDistance {
    #[inline]
    fn compute(&self, point1: &ArrayView1<F>, point2: &ArrayView1<F>) -> F {
        point1.sq_l2_dist(point2).unwrap()
    }
}

/// [Manhattan Distance](https://en.wikipedia.org/wiki/Taxicab_geometry)
#[derive(Debug, Clone, Copy)]
pub struct ManhattanDistance;

impl<F: AdriannFloat> DistanceMetric<F> for ManhattanDistance {
    #[inline]
    fn compute(&self, point1: &ArrayView1<F>, point2: &ArrayView1<F>) -> F {
        point1.l1_dist(point2).unwrap()
    }
}

/// [Chebyshev Distance](https://en.wikipedia.org/wiki/Chebyshev_distance)
#[derive(Debug, Clone, Copy)]
pub struct ChebyshevDistance;

impl<F: AdriannFloat> DistanceMetric<F> for ChebyshevDistance {
    #[inline]
    fn compute(&self, point1: &ArrayView1<F>, point2: &ArrayView1<F>) -> F {
        point1.linf_dist(point2).unwrap()
    }
}
