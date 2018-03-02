#[derive(Clone)]
pub struct Neuron {
	pub weights: Vec<f64>,
	pub bias: f64,
	pub output: f64,
	pub delta: f64,
	pub inputs: Vec<f64>
}

impl Neuron {

	pub fn new(weights: Vec<f64>, bias: f64) -> Neuron {
		Neuron {
			weights: weights,
			bias: bias,
			output: 0.0,
			delta: 0.0,
			inputs: Vec::new()
		}
	}

	pub fn activate<F>(self: &mut Neuron, input: &Vec<f64>, step: F) -> f64
		where F: Fn(&Neuron, f64) -> f64 {

		let sum = self.bias + input.iter()
			.zip(self.weights.iter())
			.fold(0.0, |last, (input, weight)| last + (input * weight));

		self.output = step(self, sum);
		self.inputs = input.clone();
		self.output
	}

	pub fn adjust(self: &mut Neuron, delta: f64, learn_rate: f64) {
		self.weights = self.weights
			.iter()
			.zip(self.inputs.iter())
			.map(|(weight, input)| weight + (input * delta * learn_rate)).collect();
		self.bias += delta * learn_rate;
	}
}