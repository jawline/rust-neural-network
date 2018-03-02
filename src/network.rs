use neuron::Neuron;

#[derive(Clone)]
pub struct Layer {
	neurons: Vec<Neuron>
}

impl Layer {
	pub fn new(neurons: &[Neuron]) -> Layer {
		Layer {
			neurons: neurons.to_vec()
		}
	}

	pub fn forward<F>(self: &mut Layer, inputs: &Vec<f64>, step: F) -> Vec<f64> 
	where F: Copy + Fn(&Neuron, f64) -> f64 {
		self.neurons.iter_mut().map(|ref mut neuron| {
			neuron.process(&inputs, step)
		}).collect()
	}
}

pub struct Network {
	layers: Vec<Layer>
}

impl Network {
	pub fn new(layers: &[Layer]) -> Network {
		Network {
			layers: layers.to_vec()
		}
	}

	pub fn process<F>(self: &mut Network, inputs: &Vec<f64>, step: F) -> f64
	where F: Copy + Fn(&Neuron, f64) -> f64 {
		self.layers.iter_mut().fold(inputs.clone(), |next_inputs, layer| {
			layer.forward(&next_inputs, step)
		})[0]
	}

	pub fn adjust<F>(self: &mut Network, delta: f64, learn_rate: f64, transfer_derivitive: F)
	where F: Copy + Fn(f64) -> f64 {

		let mut last_deltas = Vec::new();

		for cur_layer in (0..self.layers.len()).rev() {
			let mut errors = Vec::new();

			if cur_layer == self.layers.len() - 1 {
				//Final layer
				errors.push(delta)
			} else {
				let layer = &self.layers[cur_layer];
				let last_layer = &self.layers[cur_layer + 1];
				for neuron_idx in 0..layer.neurons.len() {

					let error = last_layer.neurons
							.iter()
							.enumerate()
							.fold(0.0, |last, (idx, neuron)| {
								last + (neuron.weights[neuron_idx] * last_deltas[idx])
							});

					errors.push(error);
				}
			}

			last_deltas = self.layers[cur_layer]
				.neurons
				.iter()
				.enumerate()
				.map(|(i, neuron)| errors[i] *transfer_derivitive(neuron.output))
				.collect();

			self.layers[cur_layer].neurons
				.iter_mut()
				.enumerate()
				.for_each(|(i, ref mut neuron)| {
					neuron.adjust(last_deltas[i], learn_rate)
				});
		}
	}
}