use neuron::Neuron;

use rand;
use rand::Rng;

pub fn random_neuron(num_inputs: usize) -> Neuron {
	let mut rng = rand::thread_rng();
	Neuron::new((0..num_inputs).map(|_| rng.gen::<f64>()).collect(), rng.gen::<f64>())
}