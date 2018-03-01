extern crate rand;
use rand::{ThreadRng, Rng};

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

	fn adjust(self: &mut Neuron, inputs: Vec<f32>, delta: f32, learn_rate: f32) {
		self.weights = self.weights.iter().zip(inputs.iter()).map(|(weight, input)| weight + (input * delta * learn_rate)).collect();
		self.bias += delta * learn_rate;
	}

	fn debug(self: &Neuron) {
		println!("Perceptron {:?} {}", self.weights, self.bias);
	}
}

fn c1(_x: f32, y: f32) -> f32 {
	if y > 10.0 { 1.0 } else { 0.0 }
}

fn train(p: &mut Neuron, rounds: usize, factor: f32) {
	let mut rng = rand::thread_rng();
	(0..rounds).for_each(|_| {
		let input = [rng.gen::<f32>() * 100.0, rng.gen::<f32>() * 100.0].to_vec();
		let found = p.process(input.clone());
		let expected = c1(input[0], input[1]);
		p.adjust(input, expected - found, factor);
	});
}

fn make_guess(p: &Neuron, rng: &mut ThreadRng) -> bool {
	let input = [rng.gen::<f32>() * 100.0, rng.gen::<f32>() * 100.0].to_vec();
	let found = p.process(input.clone());
	found == c1(input[0], input[1])
}

fn main() {

	let mut rng = rand::thread_rng();

    println!("Perceptron");

    let mut perceptron = Neuron::new([rng.gen::<f32>(), rng.gen::<f32>()].to_vec(), rng.gen::<f32>());
    
    println!("Initial Weights (Before Training)");
    perceptron.debug();

    train(&mut perceptron, 1000, 0.05);

    println!("After Training");
    perceptron.debug();

    perceptron.process(Vec::new());

    let mut bad = 0;
    let attempts = 100000;

	for _ in 0..attempts {
		if !make_guess(&perceptron, &mut rng) {
			bad += 1;
		}
	}

	println!("{} in {} ({}%) fail", bad, attempts, (bad as f32 / attempts as f32) * 100.0)
}