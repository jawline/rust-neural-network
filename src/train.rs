use neuron::Neuron;
use rand;
use rand::Rng;

pub fn train_perceptron<F, K>(p: &mut Neuron, rounds: usize, factor: f64, classifier: K, step: F) 
	where F: Copy + Fn(&Neuron, f64) -> f64,
		  K: Copy + Fn(f64, f64) -> f64 {
	let mut rng = rand::thread_rng();
	for _ in 0..rounds {
		let input = [rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 120.0].to_vec();
		let found = p.process(input.clone(), step);
		let expected = classifier(input[0], input[1]);
		p.adjust(input, expected - found, factor);
	}
}