pub struct Neuron {
	weights: Vec<f64>,
	bias: f64
}

impl Neuron {

	pub fn new(weights: Vec<f64>, bias: f64) -> Neuron {
		Neuron {
			weights: weights.clone(),
			bias: bias
		}
	}

	pub fn process<F>(self: &Neuron, input: Vec<f64>, step: F) -> f64
		where F: Fn(&Neuron, f64) -> f64 {
		let sum = self.bias + input.iter().zip(self.weights.iter()).fold(0.0, |last, (input, weight)| last + (input * weight));
		step(self, sum)
	}

	pub fn adjust(self: &mut Neuron, inputs: Vec<f64>, delta: f64, learn_rate: f64) {
		self.weights = self.weights.iter().zip(inputs.iter()).map(|(weight, input)| weight + (input * delta * learn_rate)).collect();
		self.bias += delta * learn_rate;
	}

	pub fn debug(self: &Neuron) {
		println!("Perceptron {:?} {}", self.weights, self.bias);
	}
}