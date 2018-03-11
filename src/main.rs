extern crate csv;
extern crate rand;
extern crate gnuplot;
extern crate rust_graph;

mod data;
mod neuron;
mod train;
mod circle;
mod plot;
mod steps;
mod network;

#[cfg(test)]
mod tests;

use rand::Rng;
use network::Network;
use steps::{Sigmoid, ReLU, Heaviside};
use circle::{gen_even_set, RANDOM_INPUT, CLASSIFY_FUNCTION};

fn dist((px, py): (f64, f64), (zx, zy): (f64, f64)) -> f64 {
	((px - zx).powi(2) + (py - zy).powi(2)).sqrt()
}

fn gen_guide_points() -> Vec<Vec<f64>> {
	let mut points = Vec::new();

	for x in 0..100 {
		for y in 0..100 {
			if CLASSIFY_FUNCTION(x as f64, y as f64)[0] == 1.0 {
				points.push([x as f64, y as f64].to_vec());
			}
		}
	}

	points
}

fn main() {

    let mut network = Network::build(2, &[4, 2]);

    let step_fn = ReLU { scalar: 0.001 };

    train::train_network(&mut network,
    	0.05,
    	&gen_even_set(50000), 
    	false,
    	|a, _| a < 15,
    	CLASSIFY_FUNCTION,
    	&step_fn);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 50;

    println!("Done Training");

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		println!("{:?} {:?}", CLASSIFY_FUNCTION(input[0], input[1]), network.process(&input, &step_fn));
		if CLASSIFY_FUNCTION(input[0], input[1])[0] == network.process(&input, &step_fn)[0].round() {
			&mut good_points
		} else {
			&mut bad_points
		}.push(input);
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	let percentage_failed = bad_points.len() as f64 / attempts as f64;
	plot::plot2("Dual", gen_guide_points(), good_points, bad_points);
	assert!(percentage_failed < 0.10);
}