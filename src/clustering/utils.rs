use ndarray::{Array1, ArrayView2, Axis};
use num_traits::FromPrimitive;
use crate::core::float::AdriannFloat;

pub fn compute_mean<F>(data: &ArrayView2<F>, indices: &[usize]) -> Array1<F>
where
    F: AdriannFloat + std::ops::Add<Output = F>,
    F: FromPrimitive
{
    if indices.is_empty() {
        return Array1::<F>::zeros(data.ncols());
    }
    let selected_data = data.select(Axis(0), indices);
    selected_data.mean_axis(Axis(0)).unwrap()
}

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive;
    use ndarray::array;
    #[test]
    fn test_compute_mean() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let indices = vec![0, 2]; // Select the first and third rows

        let result = compute_mean(&data.view(), &indices);
        let expected = array![3.0, 4.0]; // Mean of [1.0, 2.0] and [5.0, 6.0]

        assert!((result[0] - expected[0]).to_f64().unwrap().abs() < 1e-6, "Expected {}, got {}", expected[0], result[0]);
        assert!((result[1] - expected[1]).to_f64().unwrap().abs() < 1e-6, "Expected {}, got {}", expected[1], result[1]);
    }
}