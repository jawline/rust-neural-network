use network::Network;
use rand;
use rand::Rng;

use train;
use steps::{TRANSFER, TRANSFER_DERIVITIVE};

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

const CLASSIFY_FUNCTION: &'static Fn(f64, f64) -> Vec<f64> = &|x, y| [c1(x, y)].to_vec();

#[test]
fn network_line() {

    let mut network = Network::build(2, &[4, 1]);

    let training_sample: Vec<Vec<f64>> = (0..10000).map(|_| RANDOM_INPUT()).collect();

    train::train_network(&mut network,
    	10,
    	0.1,
    	&training_sample,
    	CLASSIFY_FUNCTION,
    	TRANSFER,
    	TRANSFER_DERIVITIVE);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 250;

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		if c1(input[0], input[1]) == network.process(&input, TRANSFER)[0].round() {
			&mut good_points
		} else {
			&mut bad_points
		}.push(input);
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	let percentage_failed = bad_points.len() as f64 / attempts as f64;
	//plot::plot("Network - Line", 100.0, cf(0.0), cf(100.0), good_points, bad_points);
	assert!(percentage_failed < 0.25);
}