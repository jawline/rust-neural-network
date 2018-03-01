extern crate rand;
use rand::Rng;

struct Neuron {
	weights: Vec<f32>,
	bias: f32

}

impl Neuron {
	fn new(weights: Vec<f32>, bias: f32) -> Neuron {
		Neuron {
			weights: weights.clone(),
			bias: bias
		}
	}

	fn step(self: &Neuron, sum: f32) -> f32 {
		if sum > 0.0 { 1.0 } else { 0.0 }
	}

	fn process(self: &Neuron, input: Vec<f32>) -> f32 {
		let sum = self.bias + input.iter().zip(self.weights.iter()).fold(0.0, |last, (input, weight)| last + (input * weight));
		self.step(sum)
	}

	fn debug(self: &Neuron) {
		println!("Perceptron {:?} {}", self.weights, self.bias);
	}
}

fn main() {

	let mut rng = rand::thread_rng();

    println!("Perceptron");

    let perceptron = Neuron::new([rng.gen::<f32>(), rng.gen::<f32>()].to_vec(), rng.gen::<f32>());
    
    perceptron.debug();

    perceptron.process(Vec::new());

	if rng.gen() { // random bool
	    println!("i32: {}, u32: {}", rng.gen::<i32>(), rng.gen::<u32>())
	}

}