use neuron::Neuron;

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
}