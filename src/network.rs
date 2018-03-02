use neuron::Neuron;

#[derive(Clone)]
pub struct Layer {
	neurons: Vec<Neuron>,
	links: Vec<(usize, usize)>
}

impl Layer {
	pub fn new(neurons: &[Neuron], links: &[(usize, usize)]) -> Layer {
		Layer {
			neurons: neurons.to_vec(),
			links: links.to_vec()
		}
	}

	pub fn add(self: &mut Layer, n: Neuron) {
		self.neurons.push(n);
	}

	pub fn link(self: &mut Layer, from: usize, to: usize) {
		self.links.push((from, to));
	}

	pub fn do_layer<F>(self: &mut Layer, inputs: &Vec<f64>, links: &Vec<(usize, usize)>, step: F) -> Vec<f64> 
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

pub struct Network {
	layers: Vec<Layer>
}

impl Network {
	pub fn new(layers: &[Layer]) -> Network {
		Network {
			layers: layers.to_vec()
		}
	}

	pub fn add(self: &mut Network, layer: Layer) {
		self.layers.push(layer);
	}

	pub fn process<F>(self: &mut Network, inputs: Vec<f64>, initial_links: Vec<(usize, usize)>, step: F) -> f64
	where F: Copy + Fn(&Neuron, f64) -> f64 {
		let mut current_inputs = inputs.clone();
		let mut current_links = initial_links.clone();

		for i in 0..self.layers.len() {
			current_inputs = self.layers[i].do_layer(&current_inputs, &current_links, step);
			current_links = self.layers[i].links.clone();
		}

		current_inputs[0]
	}

	fn trans_deriv(output: f64) -> f64 {
		output * (1.0 - output)
	}

	pub fn adjust(self: &mut Network, delta: f64, learn_rate: f64) {

		let mut last_deltas = Vec::new();

		for cur_layer in (0..self.layers.len()).rev() {
			let mut errors = Vec::new();

			if cur_layer == self.layers.len() - 1 {
				//Final layer
				errors.push(delta)
			} else {
				let layer = &self.layers[cur_layer];
				for neuron_idx in 0..layer.neurons.len() {

					let neuron = &layer.neurons[neuron_idx];

					let links = layer
									.links
									.iter()
									.filter(|&&(from, _)| from == neuron_idx)
									.map(|&(_, x)| x);
				
					let error = links
							.enumerate()
							.fold(0.0, |last, (link_id, next_idx)| {
								last + (neuron.weights[link_id] * last_deltas[next_idx])
							});

					errors.push(error);
				}
			}

			last_deltas = self.layers[cur_layer]
				.neurons
				.iter()
				.enumerate()
				.map(|(i, neuron)| errors[i] * Network::trans_deriv(neuron.output))
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