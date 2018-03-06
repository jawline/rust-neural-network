use neuron::Neuron;
use network::Network;

pub fn train_perceptron<F, K, R>(p: &mut Neuron, rounds: usize, factor: f64, next_input: R, classifier: K, step: F) 
	where F: Copy + Fn(&Neuron, f64) -> f64,
		  K: Copy + Fn(f64, f64) -> f64,
		  R: Copy + Fn() -> Vec<f64> {
	for _ in 0..rounds {
		let input = next_input();
		let found = p.activate(&input, step);
		let expected = classifier(input[0], input[1]);
		p.adjust(expected - found, factor);
	}
}

pub fn train_network<F, K, L>(p: &mut Network, rounds: usize, factor: f64, training_set: &[Vec<f64>], classifier: K, step: F, step_deriv: L) 
	where F: Copy + Fn(&Neuron, f64) -> f64,
		  K: Copy + Fn(f64, f64) -> Vec<f64>,
		  L: Copy + Fn(f64) -> f64 {

	let learn_rate = factor;

	for round in 0..rounds {

		let mut sum_error: f64 = 0.0;
		
		for input in training_set {
			let expected = classifier(input[0], input[1]);
			let found = p.process(input, step);
			let delta_found: Vec<f64> = expected.iter().zip(found.iter()).map(|(exp, found)| exp - found).collect();
			sum_error += delta_found.iter().fold(0.0, |l, n| l + n.abs());

			let deltas = p.backpropogate(delta_found, step_deriv);
			p.adjust_weights(&deltas, learn_rate);
		}

		println!("round={} error={} rate={}", round, sum_error, learn_rate);
	}
}

pub fn train_network_epoch<F, K, R, L>(p: &mut Network, rounds: usize, items_per_round: usize, factor: f64, next_input: R, classifier: K, step: F, step_deriv: L) 
	where F: Copy + Fn(&Neuron, f64) -> f64,
		  K: Copy + Fn(f64, f64) -> Vec<f64>,
		  R: Copy + Fn() -> Vec<f64>,
		  L: Copy + Fn(f64) -> f64 {

	let learn_rate = factor;

	let training_set: Vec<(Vec<f64>, Vec<f64>)> = (0..items_per_round).map(|_| {
		let input = next_input();
		let expected = classifier(input[0], input[1]);
		(input, expected)
	}).collect();

	for round in 0..rounds {

		let mut sum_error: f64 = 0.0;
		let mut epoch_deltas = Vec::new();
		
		for &(ref input, ref expected) in training_set.iter() {
			let found = p.process(input, step);
			let delta_found: Vec<f64> = expected.iter().zip(found.iter()).map(|(exp, found)| exp - found).collect();
			sum_error += delta_found.iter().fold(0.0, |l, n| l + n.abs());

			let deltas = p.backpropogate(delta_found, step_deriv);
			epoch_deltas.push(deltas);
		}

		epoch_deltas.iter().for_each(|deltas| p.adjust_weights(deltas, learn_rate));

		println!("round={} error={} rate={}", round, sum_error, learn_rate);
	}
}