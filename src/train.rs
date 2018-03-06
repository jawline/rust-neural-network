use neuron::Neuron;
use network::Network;

pub fn train_perceptron<F, K, R>(p: &mut Neuron, rounds: usize, factor: f64, next_input: R, classifier: K, step: F) 
	where     F: Copy + Fn(&Neuron, f64) -> f64,
		  K: Copy + Fn(f64, f64) -> f64,
		  R: Copy + Fn() -> Vec<f64> {
	for _ in 0..rounds {
		let input = next_input();
		let found = p.activate(&input, step);
		let expected = classifier(input[0], input[1]);
		p.adjust(expected - found, factor);
	}
}

pub fn train_network<Stepper, Classifier, StepDerivitive>(p: &mut Network, rounds: usize, factor: f64, training_set: &[Vec<f64>],
			      apply_epoch: bool, classifier: Classifier, step: Stepper, step_deriv: L) 
	where     Stepper: Copy + Fn(&Neuron, f64) -> f64,
		  StepDerivitive: Copy + Fn(f64) -> f64,
		  Classifier: Copy + Fn(f64, f64) -> Vec<f64> {

	for round in 0..rounds {

		let mut sum_error: f64 = 0.0;
		let mut epoch_delta: Vec<f64> = Vec::new();
		
		for input in training_set {
			let expected = classifier(input[0], input[1]);
			let found = p.process(input, step);
			let delta_found: Vec<f64> = expected.iter().zip(found.iter()).map(|(exp, found)| exp - found).collect();
			sum_error += delta_found.iter().fold(0.0, |l, n| l + n.abs());

			let deltas = p.backpropogate(delta_found, step_deriv);
			
			if !apply_epoch {
				p.adjust_weights(&deltas, learn_rate);
			} else {
				epoch_delta = epoch_deltas.iter().zip(deltas.iter()).map(|(last, next)| last + next);
			}
		}
		
		if apply_epoch {
			p.adjust_weights(epoch_delta);
		}

		println!("round={} error={} rate={}", round, sum_error, learn_rate);
	}
}
