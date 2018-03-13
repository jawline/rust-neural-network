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
use steps::{Sigmoid, Tanh, ReLU };
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

fn mix(l1: &[Vec<f64>], l2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut res = Vec::new();
    
    for i in 0..l1.len() {
        res.push(l1[i].clone());
        res.push(l2[i].clone());
    }

    res
}

fn main() {

    let a: Vec<f64> = data::load_binary("./training/chars/a.data", 35 * 35 * 3).unwrap().iter().map(|x| *x as f64).collect();
    let c: Vec<f64> = data::load_binary("./training/chars/c.data", 35 * 35 * 3).unwrap().iter().map(|x| *x as f64).collect();

    let a_norm: Vec<f64> = a.iter().map(|i| i / 255.0).collect();

    let classify = |input:&[f64]| {
        [if input.iter().eq(a_norm.iter()) { 97.0 / 255.0 } else { 0.0 }].to_vec()
    };

    let mut network = Network::build(35 * 35 * 3, &[4, 1]);

    let step_fn = Tanh {};

    let min_input = [0.0; 35 * 35 * 3];
    let max_input = [255.0; 35 * 35 * 3];
    let a_list: Vec<Vec<f64>> = std::iter::repeat(&a).take(1000).map(|a| a.clone()).collect();
    let elems: Vec<Vec<f64>> = mix(&gen_random_elements(1000), &a_list);
    let training_data = NormalizedSet::with_bounds(&elems, NormalRange::new(&min_input, &max_input));

    println!("Training on {} elements", training_data.data.len());

    train::train_network(&mut network,
    	0.1,
    	|rate, cur, prev| if cur >= prev { rate * 0.75 } else { rate },
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
            println!("Good {:?} vs {:?}", expected[0] * 255.0, found[0] * 255.0);
			&mut good_points
		} else {
            println!("Bad {:?} vs {:?}", expected[0] * 255.0, found[0] * 255.0);
			&mut bad_points
		}.push(input.clone());
	}
}
