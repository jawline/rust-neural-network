mod env;

use network::{Layer, Network};
use neuron::Neuron;

use rand;
use rand::Rng;

use train;
use steps::{TRANSFER, HEAVISIDE};

fn cf(x: f64) -> f64 {
	1.0 * x + 4.0
}

fn c1(x: f64, y: f64) -> f64 {
	if ((x * x) + (y * y)).sqrt() - (((30.0 * 30.0) + (60.0 * 60.0)) as f64).sqrt() < 10.0 {
		1.0
	} else {
		0.0
	}
}

const RANDOM_INPUT: &'static Fn() -> Vec<f64> = &|| {
	let mut rng = rand::thread_rng();
	[rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 120.0].to_vec()
};

fn random_neuron(num_inputs: usize) -> Neuron {
	let mut rng = rand::thread_rng();
	Neuron::new((0..num_inputs).map(|_| rng.gen::<f64>() * 10000.0).collect(), rng.gen::<f64>() * 10000.0)
}

fn make_guess(p: &mut Neuron, input: &Vec<f64>) -> bool {
	let found = p.process(&input, HEAVISIDE);
	found == c1(input[0], input[1])
}

const CLASSIFY_FUNCTION: &'static Fn(f64, f64) -> f64 = &|x, y| c1(x, y);

#[test]
fn do_perceptron() {
    let mut perceptron = random_neuron(2);
    train::train_perceptron(&mut perceptron, 10000, 0.4, RANDOM_INPUT, CLASSIFY_FUNCTION, HEAVISIDE);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 500;

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		if !make_guess(&mut perceptron, &input) {
			bad_points.push(input);
		} else {
			good_points.push(input);
		}
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	//plot::plot("Perceptron", 100.0, cf(0.0), cf(100.0), good_points, bad_points);

	
}

#[test]
fn do_dual() {

    let layer1 = Layer::new(&[random_neuron(2), random_neuron(2)]);
    let layer2 = Layer::new(&[random_neuron(2)]);
    let mut network = Network::new(&[layer1, layer2]);

    train::train_network(&mut network, 1000, 0.4, RANDOM_INPUT, CLASSIFY_FUNCTION, TRANSFER);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 500;

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		if c1(input[0], input[1]) == network.process(&input, TRANSFER) {
			&mut good_points
		} else {
			&mut bad_points
		}.push(input);
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	//plot::plot("Dual", 100.0, cf(0.0), cf(100.0), good_points, bad_points);
}