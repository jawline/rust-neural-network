use neuron::Neuron;
use network::Network;

pub fn train_perceptron<F, K, R>(p: &mut Neuron, rounds: usize, factor: f64, next_input: R, classifier: K, step: F) 
	where F: Copy + Fn(&Neuron, f64) -> f64,
		  K: Copy + Fn(f64, f64) -> f64,
		  R: Copy + Fn() -> Vec<f64> {
	for _ in 0..rounds {
		let input = next_input();
		let found = p.process(&input, step);
		let expected = classifier(input[0], input[1]);
		p.adjust(expected - found, factor);
	}
}

pub fn train_network<F, K, R, L>(p: &mut Network, rounds: usize, factor: f64, next_input: R, classifier: K, step: F, step_deriv: L) 
	where F: Copy + Fn(&Neuron, f64) -> f64,
		  K: Copy + Fn(f64, f64) -> f64,
		  R: Copy + Fn() -> Vec<f64>,
		  L: Copy + Fn(f64) -> f64 {
	for _ in 0..rounds {
		let input = next_input();
		let found = p.process(&input, step);
		let expected = classifier(input[0], input[1]);
		p.adjust(expected - found, factor, step_deriv);
	}
}