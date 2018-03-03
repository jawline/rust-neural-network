use tests::env::random_neuron;

use train;
use steps::{HEAVISIDE};

use rand;
use rand::Rng;

fn cf(x: f64) -> f64 {
	1.0 * x + 4.0
}

fn c1(x: f64, y: f64) -> f64 {
	if cf(x) > y { 1.0 } else { 0.0 }
}

const RANDOM_INPUT: &'static Fn() -> Vec<f64> = &|| {
	let mut rng = rand::thread_rng();
	[rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 120.0].to_vec()
};

const CLASSIFY_FUNCTION: &'static Fn(f64, f64) -> f64 = &|x, y| c1(x, y);

#[test]
fn do_perceptron() {
    let mut perceptron = random_neuron(2);
    train::train_perceptron(&mut perceptron, 10000, 0.05, RANDOM_INPUT, CLASSIFY_FUNCTION, HEAVISIDE);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 250;

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		if perceptron.activate(&input, HEAVISIDE) == c1(input[0], input[1]) {
			&mut good_points
		} else {
			&mut bad_points
		}.push(input);
	}

	let percentage_failed = bad_points.len() as f64 / attempts as f64;

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, percentage_failed * 100.0);

	//plot::plot("Perceptron", 100.0, cf(0.0), cf(100.0), good_points, bad_points);
	//println!("{}", percentage_failed);
	assert!(percentage_failed < 0.25);
}