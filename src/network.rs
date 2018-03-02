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
			neuron.activate(&inputs, step)
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

	pub fn backpropogate<F>(self: &mut Network, delta: f64, learn_rate: f64, transfer_derivitive: F)
	where F: Copy + Fn(f64) -> f64 {

		let layer_deltas: Vec<Vec<f64>> = 
			(0..self.layers.len()).rev().map(|cur_layer| {
				let errors = if cur_layer == self.layers.len() - 1 {
					[delta].to_vec()
				} else {
					let neuron_count = *&self.layers[cur_layer].neurons.len();

					(0..neuron_count).map(|neuron_idx| {
						//Find the weighted error of every output that uses this neuron as an input
						*(&self.layers[cur_layer + 1]
								.neurons
								.iter()
								.fold(0.0, |last, neuron| {
									last + (neuron.weights[neuron_idx] * neuron.delta)
								}))
					}).collect()
				};

				//Update the deltas
				self.layers[cur_layer]
					.neurons
					.iter_mut()
					.enumerate()
					.map(|(i, ref mut neuron)| {
						errors[i] * transfer_derivitive(neuron.output)
					})
					.collect()
			}).collect();

		//Adjust each neuron
		self.layers
			.iter_mut()
			.zip(layer_deltas.iter().rev())
			.for_each(|(ref mut layer, ref layer_deltas)| {
				layer.neurons
					.iter_mut()
					.zip(layer_deltas.iter())
					.for_each(|(ref mut neuron, ref delta)| {
							neuron.adjust(**delta, learn_rate);
					});
			});

	}
}