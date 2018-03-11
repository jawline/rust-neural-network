extern crate csv;
extern crate rand;
extern crate gnuplot;
extern crate rust_graph;

mod data;
mod neuron;
mod train;
mod plot;
mod steps;
mod network;

#[cfg(test)]
mod tests;

use rand::Rng;
use network::Network;
use steps::{Sigmoid, ReLU, Heaviside};
use data::{NormalRange, NormalizedSet};

pub fn dist((px, py): (f64, f64), (zx, zy): (f64, f64)) -> f64 {
	((px - zx).powi(2) + (py - zy).powi(2)).sqrt()
}

pub const RANDOM_INPUT: &'static Fn() -> Vec<f64> = &|| {
	let mut rng = rand::thread_rng();
	[rng.gen::<f64>().abs() * 100.0, rng.gen::<f64>().abs() * 100.0].to_vec()
};

pub const CLASSIFY_FUNCTION: &'static Fn(&[f64]) -> Vec<f64> = &|d| {
	if dist((d[0] * 100.0, d[1] * 100.0), (50.0, 50.0)) < 30.0 {
		[1.0, 0.0].to_vec()
	} else {
		[0.0, 1.0].to_vec()
	}
};

pub fn gen_guide_points(range: &NormalRange) -> Vec<Vec<f64>> {
	let mut points = Vec::new();

	for x in 0..100 {
		for y in 0..100 {
			let normalized = range.point(&[x as f64, y as f64]);
			if CLASSIFY_FUNCTION(&normalized)[0] == 1.0 {
				points.push(normalized);
			}
		}
	}

	points
}

fn main() {

    let mut network = Network::build(2, &[3, 2]);

    let step_fn = ReLU { scalar: 0.01 };
    let points: Vec<Vec<f64>> = (0..5000).map(|_| RANDOM_INPUT()).collect();

    let training_set = NormalizedSet::with_bounds(
    	&points,
    	&[0.0, 0.0],
    	&[100.0, 100.0]
    );

    let points: Vec<Vec<f64>> = (0..500).map(|_| RANDOM_INPUT()).collect();

    let active_set = NormalizedSet::with_bounds(
    	&points,
    	&[0.0, 0.0],
    	&[100.0, 100.0]
    );

    train::train_network(&mut network,
    	0.01,
    	&training_set.data, 
    	false,
    	|a, _| a < 100,
    	CLASSIFY_FUNCTION,
    	&step_fn);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();

    println!("Done Training");

	for input in &active_set.data {
		let expected = CLASSIFY_FUNCTION(input);
		let found = network.process(input, &step_fn);
		let error = expected.iter().zip(found.iter()).fold(0.0, |l, (&e, &f)| l + (e - f.round()).abs());
		if error == 0.0 {
			&mut good_points
		} else {
			println!("Failed {:?} vs {:?}", expected, found);
			&mut bad_points
		}.push(input.clone());
	}

	let percentage_failed = bad_points.len() as f64 / active_set.data.len() as f64;
	plot::plot2("Dual", gen_guide_points(&training_set.range), good_points, bad_points);
	assert!(percentage_failed < 0.10);
}