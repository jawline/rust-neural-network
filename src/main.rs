extern crate rand;
extern crate gnuplot;

mod neuron;
mod train;
mod plot;
mod steps;
mod network;

use neuron::Neuron;
use rand::Rng;
use steps::HEAVISIDE;

pub fn cf(x: f64) -> f64 {
	1.0 * x + 4.0
}

pub fn c1(x: f64, y: f64) -> f64 {
	if cf(x) > y { 1.0 } else { 0.0 }
}

fn make_guess(p: &Neuron, input: &Vec<f64>) -> bool {
	let found = p.process(input.clone(), HEAVISIDE);
	found == c1(input[0], input[1])
}

const CLASSIFY_FUNCTION: &'static Fn(f64, f64) -> f64 = &|x, y| c1(x, y);

const RANDOM_INPUT: &'static Fn() -> Vec<f64> = &|| {
	let mut rng = rand::thread_rng();
	[rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 120.0].to_vec()
};

fn main() {

	let mut rng = rand::thread_rng();

    println!("Perceptron");

    let mut perceptron = Neuron::new(RANDOM_INPUT(), rng.gen::<f64>());
    train::train_perceptron(&mut perceptron, 1000, 0.9, RANDOM_INPUT, CLASSIFY_FUNCTION, HEAVISIDE);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 500;

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		if !make_guess(&perceptron, &input) {
			bad_points.push(input);
		} else {
			good_points.push(input);
		}
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	plot::plot(100.0, cf(0.0), cf(100.0), good_points, bad_points);
}