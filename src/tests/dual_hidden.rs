use tests::env::random_neuron;

use network::{Layer, Network};
use neuron::Neuron;
use rand;
use rand::Rng;

use train;
use steps::{TRANSFER, TRANSFER_DERIVITIVE};
use plot;

fn distance((c_x, c_y): (f64, f64), (p_x, p_y): (f64, f64)) -> f64 {
	((c_x - p_x).powi(2) + (c_y - p_y).powi(2)).sqrt()
}

fn c1(x: f64, y: f64) -> f64 {
	let is_close = distance((30.0, 40.0), (x, y)) < 10.0;
	if is_close { 1.0 } else { 0.0 }
}

const RANDOM_INPUT: &'static Fn() -> Vec<f64> = &|| {
	let mut rng = rand::thread_rng();
	[rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 120.0].to_vec()
};

const CLASSIFY_FUNCTION: &'static Fn(f64, f64) -> f64 = &|x, y| c1(x, y);

#[test]
fn do_dual() {

    let layer1 = Layer::new(&[random_neuron(2), random_neuron(2)]);
    let layer2 = Layer::new(&[random_neuron(2)]);
    let mut network = Network::new(&[layer1, layer2]);

    train::train_network(&mut network, 1000, 0.4, RANDOM_INPUT, CLASSIFY_FUNCTION, TRANSFER, TRANSFER_DERIVITIVE);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 250;

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		println!("{} {}", network.process(&input, TRANSFER), c1(input[0], input[1]));
		if c1(input[0], input[1]) == network.process(&input, TRANSFER) {
			&mut good_points
		} else {
			&mut bad_points
		}.push(input);
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	let percentage_failed = bad_points.len() as f64 / attempts as f64;
	plot::plot("Dual", 100.0, 0.0, 100.0, good_points, bad_points);
	assert!(percentage_failed < 0.25);
}