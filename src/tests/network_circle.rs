use rand;
use rand::Rng;

use network::Network;

use train;
use steps::{TRANSFER, TRANSFER_DERIVITIVE};
use plot;

fn dist((px, py): (f64, f64), (zx, zy): (f64, f64)) -> f64 {
	((px - zx).powi(2) + (py - zy).powi(2)).sqrt()
}

const RANDOM_INPUT: &'static Fn() -> Vec<f64> = &|| {
	let mut rng = rand::thread_rng();
	[rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 100.0].to_vec()
};

const CLASSIFY_FUNCTION: &'static Fn(f64, f64) -> Vec<f64> = &|x, y| {

	if dist((x, y), (50.0, 40.0)) < 35.0 {
		[1.0, 0.0].to_vec()
	} else {
		[0.0, 1.0].to_vec()
	}
};

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

#[test]
fn network_circle() {

    let mut network = Network::build(2, &[4, 2]);

    train::train_network(&mut network, 100, 5000, 0.09, RANDOM_INPUT, CLASSIFY_FUNCTION, TRANSFER, TRANSFER_DERIVITIVE);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 250;

	for _ in 0..attempts {
		let input = RANDOM_INPUT();
		println!("{:?} {:?}", CLASSIFY_FUNCTION(input[0], input[1]), network.process(&input, TRANSFER));
		if CLASSIFY_FUNCTION(input[0], input[1])[0] == network.process(&input, TRANSFER)[0].round() {
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