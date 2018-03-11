use network::Network;
use rand;
use rand::Rng;

use train;
use steps::Heaviside;
use plot;

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

const CLASSIFY_FUNCTION: &'static Fn(&[f64]) -> Vec<f64> = &|d| [c1(d[0], d[1])].to_vec();

#[test]
fn network_line() {

    let mut network = Network::build(2, &[1]);

    let training_sample: Vec<Vec<f64>> = (0..10000).map(|_| RANDOM_INPUT()).collect();

    let step_fn = Heaviside{};

    train::train_network(&mut network,
    	0.1,
    	|rate, cur, prev| if cur >= prev { rate * 0.95 } else { rate },
    	&training_sample,
    	|a, _| a < 30,
    	CLASSIFY_FUNCTION,
    	&step_fn);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 250;

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		println!("Pt {:?} vs {:?}", c1(input[0], input[1]), network.process(&input, &step_fn)[0].round());
		if c1(input[0], input[1]) == network.process(&input, &step_fn)[0].round() {
			&mut good_points
		} else {
			&mut bad_points
		}.push(input);
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	let passed = (bad_points.len() as f64 / attempts as f64) < 0.25;
	
	if !passed {
		plot::plot("Network - Line", 100.0, cf(0.0), cf(100.0), good_points, bad_points);
	}

	assert!(passed);
}