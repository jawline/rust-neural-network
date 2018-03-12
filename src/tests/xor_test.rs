use rand;
use plot;
use train;
use rand::Rng;
use network::Network;
use steps::Tanh;
use data::{NormalRange, NormalizedSet};

pub const RANDOM_INPUT: &'static Fn() -> Vec<f64> = &|| {
	let mut rng = rand::thread_rng();
	[rng.gen::<f64>().abs() * 1000.0, rng.gen::<f64>().abs() * 1000.0].to_vec()
};

pub const CLASSIFY_FUNCTION: &'static Fn(&[f64]) -> Vec<f64> = &|d| {
	let left = d[0] * 1000.0;
	let right = d[1] * 1000.0;
	[(left as i64 ^ right as i64) as f64 / 1000.0].to_vec()
};

fn unmap(p: &[Vec<f64>], range: &NormalRange) -> Vec<Vec<f64>> {
	p.iter().map(|i| range.reverse(i)).collect()
}

#[test]
fn network_xor() {

    let mut network = Network::build(2, &[4, 1]);

    let step_fn = Tanh{};
    let data_range = NormalRange::new(&[0.0, 0.0], &[1000.0, 1000.0]);
    let points: Vec<Vec<f64>> = (0..50000).map(|_| RANDOM_INPUT()).collect();

    let training_set = NormalizedSet::with_bounds(
    	&points,
    	data_range.clone()
    );

    let points: Vec<Vec<f64>> = (0..500).map(|_| RANDOM_INPUT()).collect();

    let active_set = NormalizedSet::with_bounds(
    	&points,
    	data_range.clone()
    );

    train::train_network(&mut network,
    	0.01,
    	|rate, cur, prev| if cur >= prev { rate * 0.9 } else { rate },
    	&training_set.data,
    	|a, err| a == 0 || (a < 15 && err > 300.0),
    	CLASSIFY_FUNCTION,
    	&step_fn);

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();

    println!("Done Training");

	for input in &active_set.data {
		let expected = CLASSIFY_FUNCTION(input);
		let found = network.process(input, &step_fn);
		let error = expected.iter().zip(found.iter()).fold(0.0, |l, (&e, &f)| l + (e - f).abs());
		if error < 0.25 {
			&mut good_points
		} else {
			println!("Failed {:?} vs {:?}", expected, found);
			&mut bad_points
		}.push(input.clone());
	}

	let percentage_failed = bad_points.len() as f64 / active_set.data.len() as f64;
	let failed = percentage_failed > 0.2;

	println!("Percentage Failed: {}", percentage_failed * 100.0);

	if failed {
		plot::plot3("XOR", unmap(&good_points, &data_range), unmap(&bad_points, &data_range));
	}

	assert!(!failed);
}