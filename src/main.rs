extern crate rand;
extern crate gnuplot;

mod neuron;
mod train;
mod plot;

use neuron::Neuron;
use rand::Rng;

pub fn cf(x: f64) -> f64 {
	1.0 * x + 4.0
}

pub fn c1(x: f64, y: f64) -> f64 {
	if cf(x) > y { 1.0 } else { 0.0 }
}

fn make_guess(p: &Neuron, input: &Vec<f64>) -> bool {
	let found = p.process(input.clone(), STEP_FUNCTION);
	found == c1(input[0], input[1])
}

const STEP_FUNCTION: &'static Fn(&Neuron, f64) -> f64 = &|_, v| v;
const CLASSIFY_FUNCTION: &'static Fn(f64, f64) -> f64 = &|x, y| c1(x, y);

fn main() {

	let mut rng = rand::thread_rng();

    println!("Perceptron");

    let mut perceptron = Neuron::new([rng.gen::<f64>(), rng.gen::<f64>()].to_vec(), rng.gen::<f64>());
    train::train_perceptron(&mut perceptron, 1000, 0.9, CLASSIFY_FUNCTION, STEP_FUNCTION);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 500;

	for _ in 0..attempts {
		let input = [rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 120.0].to_vec();
		if !make_guess(&perceptron, &input) {
			bad_points.push(input);
		} else {
			good_points.push(input);
		}
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	plot::plot(100.0, cf(0.0), cf(100.0), good_points, bad_points);
}