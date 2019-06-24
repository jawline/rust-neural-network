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

pub fn train_network<F: StepFn, LearnSlope, Classifier, ExitTest>(p: &mut Network, learn_rate: f64, learn_slope: LearnSlope, training_set: &[Vec<f64>],
		exit_test: ExitTest, classifier: &Classifier, step: &F)
	where Classifier: Fn(&[f64]) -> Vec<(f64, f64)>,
		  ExitTest: Fn(usize, f64) -> bool,
		  LearnSlope: Fn(f64, f64, f64) -> f64 {
			  
	let mut round_errors: Vec<f64> = Vec::new();
	let mut rounds = 0;
	let mut prev_error = 0.0;
	let mut learn_rate = learn_rate;
			  
	while exit_test(rounds, prev_error) {

    use std::iter::once;
		let mut sum_error: f64 = 0.0;
		
		for input in training_set {
			let expected = classifier(input);
			let found = p.process(input, step);
			let delta_found: Vec<f64> = expected
        .iter()
        .flat_map(|(x, y)| once(x).chain(once(y)))
        .zip(found.iter())
        .map(|(e1, f1)| (*e1 - f1)).collect();
			sum_error += delta_found.iter().fold(0.0, |l, n| l + n.powi(2)) / delta_found.len() as f64;

			let deltas = p.backpropogate(delta_found, step);
			p.adjust_weights(&deltas, learn_rate);
		}

		learn_rate = learn_slope(learn_rate, sum_error, prev_error);

		println!("round={} error={} rate={}", rounds, sum_error, learn_rate);
		
		round_errors.push(sum_error);
		rounds += 1;
		prev_error = sum_error;
	}
}
