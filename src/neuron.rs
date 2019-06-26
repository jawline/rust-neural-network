use rand;
use rand::Rng;
use steps::StepFn;

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct Neuron {
	pub weights: Vec<f64>,
	pub bias: f64,
	pub output: f64,
	pub inputs: Vec<f64>
}

impl Neuron {
	pub fn new(weights: Vec<f64>, bias: f64) -> Neuron {
    let n_weights = weights.len();
		Neuron {
			weights: weights,
			bias: bias,
			output: 0.0,
			inputs: (0..n_weights).map(|_| 0.0).collect()
		}
	}

	pub fn activate<F: StepFn>(self: &mut Neuron, input: &[f64], step: &F) -> f64 {
    input.iter().enumerate().for_each(|(i, ref v)| self.inputs[i] = **v);
		let sum = self.bias + input.iter()
			.zip(self.weights.iter())
			.fold(0.0, |last, (input, weight)| last + (input * weight));
		self.output = step.transfer(sum);
		self.output
	}

	pub fn adjust(self: &mut Neuron, delta: f64, learn_rate: f64) {
		self.weights
			.iter_mut()
			.zip(self.inputs.iter())
			.for_each(|(ref mut weight, input)| **weight = **weight + (delta * learn_rate * input));
		self.bias += delta * learn_rate;
	}

	pub fn random(num_inputs: usize) -> Neuron {
		let mut rng = rand::thread_rng();
		Neuron::new((0..num_inputs).map(|_| rng.gen::<f64>().abs()).collect(), rng.gen::<f64>())
	}
}
