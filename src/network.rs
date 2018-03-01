use neuron::Neuron;
use steps::StepFn;

pub struct Layer {
	neurons: Vec<Neuron>,
	links: Vec<(usize, usize)>
}

impl Layer {
	pub fn new() -> Layer {
		Layer {
			neurons: Vec::new(),
			links: Vec::new()
		}
	}

	pub fn add(self: &mut Layer, n: Neuron) {
		self.neurons.push(n);
	}

	pub fn link(self: &mut Layer, from: usize, to: usize) {
		self.links.push((from, to));
	}

	pub fn do_layer<F>(self: &Layer, inputs: Vec<f64>, links: Vec<(usize, usize)>, step: F) -> Vec<f64> 
	where F: Copy + Fn(&Neuron, f64) -> f64 {
		let mut results = Vec::new();

		for i in 0..self.neurons.len() {
			let target_inputs: Vec<f64> = links.iter()
				.filter(|&&(_, y)| y == i)
				.map(|&(x, _)| x)
				.map(|x| inputs[x])
				.collect();
			results.push(self.neurons[i].process(target_inputs, step));
		}
		
		results
	}
}