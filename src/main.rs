extern crate rand;
extern crate gnuplot;

mod neuron;

use neuron::Neuron;
use rand::Rng;
use gnuplot::{Figure, Caption, Color};

fn cf(x: f64) -> f64 {
	1.0 * x + 4.0
}

fn c1(x: f64, y: f64) -> f64 {
	if cf(x) > y { 1.0 } else { 0.0 }
}

fn train(p: &mut Neuron, rounds: usize, factor: f64) {
	let mut rng = rand::thread_rng();
	(0..rounds).for_each(|_| {
		let input = [rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 120.0].to_vec();
		let found = p.process(input.clone());
		let expected = c1(input[0], input[1]);
		p.adjust(input, expected - found, factor);
	});
}

fn make_guess(p: &Neuron, input: &Vec<f64>) -> bool {
	let found = p.process(input.clone());
	found == c1(input[0], input[1])
}

fn extract(points: &Vec<Vec<f64>>, index: usize) -> Vec<f64> {
	points.iter().map(|x| x[index]).collect()
}

fn main() {

	let mut rng = rand::thread_rng();

    println!("Perceptron");

    let mut perceptron = Neuron::new([rng.gen::<f64>(), rng.gen::<f64>()].to_vec(), rng.gen::<f64>());
    
    println!("Initial Weights (Before Training)");
    perceptron.debug();

    train(&mut perceptron, 1000, 0.9);

    println!("After Training");
    perceptron.debug();

    perceptron.process(Vec::new());

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();
    let attempts = 500;

	let mut fg = Figure::new();

	for _ in 0..attempts {
		let input = [rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 120.0].to_vec();
		if !make_guess(&perceptron, &input) {
			bad_points.push(input);
		} else {
			good_points.push(input);
		}
	}

	println!("{} in {} ({}%) fail", bad_points.len(), attempts, (bad_points.len() as f64 / attempts as f64) * 100.0);

	let x = [0.0f64, 100.0];
	let y = [cf(0.0), cf(100.0)];

	fg.axes2d().lines(&x, &y, &[Caption("The dividing line"), Color("black")])
		.points(
			&(extract(&good_points, 0)),
			&(extract(&good_points, 1)),
			&[Color("green")]
		).points(
			&(extract(&bad_points, 0)),
			&(extract(&bad_points, 1)),
			&[Color("red")]
		);
	fg.show();
}