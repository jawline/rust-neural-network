#![allow(dead_code)]

extern crate rustc_serialize;
extern crate csv;
extern crate rand;
extern crate gnuplot;
extern crate rust_graph;

mod data;
mod neuron;
mod train;
mod plot;
mod steps;
mod network;

#[cfg(test)]
mod tests;


use rand::distributions::{Sample, Range};
use network::Network;
use steps::Tanh;
use data::{NormalRange, NormalizedSet};

fn gen_random_elements(size: usize) -> Vec<Vec<f64>> {
    let mut between = Range::new(0.0, 255.0);
    let mut rng = rand::thread_rng();

    let mut r_list = Vec::new();
    
    for _ in 0..size {
        let mut list_item = Vec::new();
        
        for _ in 0..(35 * 35 * 3) {
            list_item.push(between.sample(&mut rng) as f64);
        }

        r_list.push(list_item);
    }

    r_list
}

fn main() {

    let a: Vec<f64> = data::load_binary("./training/chars/a.data", 35 * 35 * 3).unwrap().iter().map(|x| *x as f64).collect();

    let classify = |input:&[f64]| {
        [input.iter().zip(a.iter()).fold(0.0, |l, (i, a)| a - i)].to_vec()
    };

    let mut network = Network::build(35 * 35 * 3, &[3, 1]);

    let step_fn = Tanh { };

    let min_input = [0.0; 35 * 35 * 3];
    let max_input = [255.0; 35 * 35 * 3];
    let elems: Vec<Vec<f64>> = gen_random_elements(100)
        .iter()
        .map(|a| a.clone())
        .collect();
    let training_data = NormalizedSet::with_bounds(&elems, NormalRange::new(&min_input, &max_input));

    println!("Training");

    train::train_network(&mut network,
    	0.01,
    	|rate, cur, prev| if cur >= prev { rate * 0.99 } else { rate },
    	&training_data.data,
    	|a, err| a < 10,
    	&classify,
    	&step_fn);

    println!("Done Training");

    let testing_data = NormalizedSet::with_bounds(&[a.clone(), min_input.to_vec()], NormalRange::new(&min_input, &max_input));

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();

	for input in &testing_data.data {
		let expected = classify(input);
		let found = network.process(input, &step_fn);
		let error = expected.iter().zip(found.iter()).fold(0.0, |l, (&e, &f)| l + (e - f).abs());
		if error == 0.0 {
            println!("Good {:?} vs {:?}", expected, found);
			&mut good_points
		} else {
			println!("Failed {:?} vs {:?}", expected, found);
			&mut bad_points
		}.push(input.clone());
	}
}
