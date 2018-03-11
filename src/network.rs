use neuron::Neuron;
use steps::StepFn;

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct Layer {
	neurons: Vec<Neuron>
}

impl Layer {
	pub fn new(neurons: &[Neuron]) -> Layer {
		Layer { neurons: neurons.to_vec() }
	}

	pub fn forward<F: StepFn>(self: &mut Layer, inputs: &Vec<f64>, step: &F) -> Vec<f64> {
		self.neurons.iter_mut().map(|ref mut neuron| {
			neuron.activate(&inputs, step)
		}).collect()
	}

	pub fn transfer_errors<F: StepFn>(self: &mut Layer, errors: &Vec<f64>, step: &F) -> Vec<f64>
	where F: StepFn {
		self.neurons
		    .iter_mut()
		    .enumerate()
		    .map(|(i, ref mut neuron)| {
		    	errors[i] * step.derive(neuron.output)
		    }).collect()
	}
}

#[derive(RustcDecodable, RustcEncodable)]
pub struct Network {
	layers: Vec<Layer>
}

impl Network {
	pub fn new(layers: Vec<Layer>) -> Network {
		Network { layers: layers }
	}

	pub fn process<F: StepFn>(self: &mut Network, inputs: &Vec<f64>, step: &F) -> Vec<f64> {
		self.layers.iter_mut().fold(inputs.clone(), |next_inputs, layer| {
			layer.forward(&next_inputs, step)
		})
	}

	fn layer_weighted_error(self: &Network, last_deltas: &Vec<f64>, cur_layer: usize) -> Vec<f64> {
		
		let last_layer = cur_layer == self.layers.len() - 1;
		
		match last_layer {
			true => last_deltas.to_vec(),
			false => {
				let neuron_count = *&self.layers[cur_layer].neurons.len();
				(0..neuron_count).map(|neuron_idx| {
					//Find the weighted error of every output that uses this neuron as an input
					*(&self.layers[cur_layer + 1]
						.neurons
						.iter()
						.enumerate()
						.fold(0.0, |last, (i, neuron)| {
							last + (neuron.weights[neuron_idx] * last_deltas[i])
						}))
				}).collect()
			}
		}
	}

	pub fn backpropogate<F: StepFn>(self: &mut Network, delta: Vec<f64>, step: &F) -> Vec<Vec<f64>> 	{
		let mut last_deltas = delta;
		(0..self.layers.len()).rev().map(|cur_layer| {
			let error_ratios = self.layer_weighted_error(&last_deltas, cur_layer);
			last_deltas = self.layers[cur_layer].transfer_errors(&error_ratios, step);
			last_deltas.clone()
		}).collect()
	}

	pub fn adjust_weights(self: &mut Network, layer_deltas: &Vec<Vec<f64>>, learn_rate: f64) {
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

	fn build_layer(program_inputs: usize, size: usize, last_size: &mut usize) -> Layer {
		let items: Vec<Neuron> = (0..size).map(|_| Neuron::random(program_inputs)).collect();
		*last_size = size;
		Layer::new(&items)
	}

	pub fn build(inputs: usize, layer_sizes: &[usize]) -> Network {
		let mut last_size = inputs;

		Network::new(layer_sizes.iter().map(|size| {
			Network::build_layer(last_size, *size, &mut last_size)
		}).collect())
	}
}
