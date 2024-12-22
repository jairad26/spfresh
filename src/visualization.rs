use crate::clustering::{distance::SquaredEuclideanDistance, Cluster, DistanceMetric};
use colored::Colorize;
use ndarray::Array2;
use num_traits::{Float, Signed};
use std::fmt;
use std::ops::AddAssign;

pub fn print_cluster_analysis<F>(clusters: &[Cluster], data: &Array2<F>)
where
    F: Float + AddAssign + Signed,
{
    println!("\n{}", "=== Cluster Analysis ===".bold());

    let stats = calculate_cluster_stats(clusters, data);
    print_summary_statistics(&stats);
    //print_detailed_cluster_info(&stats);
    //print_hierarchy_visualization(clusters);
}

fn calculate_cluster_stats<F>(clusters: &[Cluster], data: &Array2<F>) -> Vec<ClusterStats>
where
    F: Float + AddAssign + Signed,
{
    clusters
        .iter()
        .enumerate()
        .map(|(idx, cluster)| {
            let (avg_distance, sum_distance) = calculate_distance_stats(cluster, data);
            ClusterStats {
                cluster_id: idx,
                size: cluster.points.len(),
                avg_distance_to_centroid: avg_distance,
                sum_distance_to_centroid: sum_distance,
                depth: cluster.depth,
            }
        })
        .collect()
}

fn calculate_distance_stats<F>(cluster: &Cluster, data: &Array2<F>) -> (f64, f64)
where
    F: Float + AddAssign + Signed,
{
    if cluster.points.is_empty() {
        return (0.0, 0.0);
    }

    let sum_distance: f64 = cluster
        .points
        .iter()
        .map(|&idx| {
            let point = data.row(idx);
            SquaredEuclideanDistance
                .compute(&point, &data.row(cluster.centroid_idx.unwrap()).view())
                .to_f64()
                .unwrap()
        })
        .sum();

    let avg_distance = sum_distance / cluster.points.len() as f64;
    (avg_distance, sum_distance)
}

#[derive(Debug)]
struct ClusterStats {
    cluster_id: usize,
    size: usize,
    avg_distance_to_centroid: f64,
    sum_distance_to_centroid: f64,
    depth: usize,
}

impl fmt::Display for ClusterStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cluster {}: {} points, Avg Distance: {:.2}, Sum Distance: {:.2}, Depth: {}",
            self.cluster_id,
            self.size,
            self.avg_distance_to_centroid,
            self.sum_distance_to_centroid,
            self.depth
        )
    }
}

fn print_summary_statistics(stats: &[ClusterStats]) {
    println!("\n{}", "Summary Statistics:".bold());
    println!("Total Clusters: {}", stats.len());

    let total_points: usize = stats.iter().map(|s| s.size).sum();
    let avg_cluster_size = total_points as f64 / stats.len() as f64;
    let max_depth = stats.iter().map(|s| s.depth).max().unwrap_or(0);
    let total_sum_distance: f64 = stats.iter().map(|s| s.sum_distance_to_centroid).sum();
    let overall_avg_distance = total_sum_distance / total_points as f64;

    println!("Total Points: {}", total_points);
    println!("Average Cluster Size: {:.2}", avg_cluster_size);
    println!("Maximum Hierarchy Depth: {}", max_depth);
    println!("Total Sum of Distances: {:.2}", total_sum_distance);
    println!("Overall Average Distance: {:.2}", overall_avg_distance);
}

fn _print_detailed_cluster_info(stats: &[ClusterStats]) {
    println!("\n{}", "Detailed Cluster Information:".bold());
    for stat in stats {
        let info = format!("{}", stat);
        match stat.size {
            size if size < 5 => println!("{}", info.red()),
            size if size > 15 => println!("{}", info.yellow()),
            _ => println!("{}", info.green()),
        }
    }
}

fn _print_hierarchy_visualization(clusters: &[Cluster]) {
    println!("\n{}", "Hierarchy Visualization:".bold());
    let max_depth = clusters.iter().map(|c| c.depth).max().unwrap_or(0);

    for depth in 0..=max_depth {
        println!("\nDepth {}:", depth);
        for (idx, cluster) in clusters.iter().enumerate() {
            if cluster.depth == depth {
                println!("  ├── Cluster {}: ({} points)", idx, cluster.points.len());
            }
        }
    }
}
