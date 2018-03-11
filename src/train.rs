use neuron::Neuron;
use network::Network;
use steps::StepFn;

pub fn train_perceptron<F: StepFn, K, R>(p: &mut Neuron, rounds: usize, factor: f64, next_input: R, classifier: K, step: &F) 
	where K: Copy + Fn(f64, f64) -> f64,
		  R: Copy + Fn() -> Vec<f64> {
	for _ in 0..rounds {
		let input = next_input();
		let found = p.activate(&input, step);
		let expected = classifier(input[0], input[1]);
		p.adjust(expected - found, factor);
	}
}

pub fn train_network<F: StepFn, Classifier, ExitTest>(p: &mut Network, learn_rate: f64, training_set: &[Vec<f64>],
			      apply_epoch: bool, exit_test: ExitTest, classifier: Classifier, step: &F)
	where Classifier: Copy + Fn(f64, f64) -> Vec<f64>,
		  ExitTest: Fn(usize, &[f64]) -> bool {
			  
	let mut round_errors: Vec<f64> = Vec::new();
	let mut rounds = 0;
			  
	while exit_test(rounds, &round_errors) {

		let mut sum_error: f64 = 0.0;
		let mut epoch_delta: Vec<Vec<f64>> = Vec::new();
		
		for input in training_set {
			let expected = classifier(input[0], input[1]);
			let found = p.process(input, step);
			let delta_found: Vec<f64> = expected.iter().zip(found.iter()).map(|(exp, found)| exp - found).collect();
			sum_error += delta_found.iter().fold(0.0, |l, n| l + n.powi(2)) / delta_found.len() as f64;

			let deltas = p.backpropogate(delta_found, step);
			
			if !apply_epoch {
				p.adjust_weights(&deltas, learn_rate);
			}
		}
		
		if apply_epoch {
			println!("TODO: Apply epoch mode");
		}

		println!("round={} error={} rate={}", rounds, sum_error, learn_rate);
		
		round_errors.push(sum_error);
		rounds += 1;
	}
}
