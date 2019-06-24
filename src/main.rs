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

use std::iter;
use rand::Rng;
use rand::distributions::{Uniform};
use network::Network;
use steps::{Sigmoid, Tanh, ReLU };
use data::{NormalRange, NormalizedSet};

fn gen_random_elements(size: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_|
      (0..(35*35*3)).map(|_| rng.sample(Uniform::new(0.0, 255.0))).collect()
    ).collect()
}

fn mix(l1: &[Vec<f64>], l2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut res = Vec::new();
    for i in 0..l1.len() {
        res.push(l1[i].clone());
        if i < l2.len() {
            res.push(l2[i].clone());
        }
    }
    res
}

fn main() {

    let a: Vec<f64> = data::load_binary("./training/chars/a.data", 35 * 35 * 3).unwrap().iter().map(|x| *x as f64).collect();
    let a_2: Vec<f64> = data::load_binary("./training/chars/a_2.data", 35 * 35 * 3).unwrap().iter().map(|x| *x as f64).collect();
    let c: Vec<f64> = data::load_binary("./training/chars/c.data", 35 * 35 * 3).unwrap().iter().map(|x| *x as f64).collect();

    let a_norm: Vec<f64> = a.iter().map(|i| i / 255.0).collect();
    let a_2_norm: Vec<f64> = a_2.iter().map(|i| i / 255.0).collect();
    let c_norm: Vec<f64> = c.iter().map(|i| i / 255.0).collect();

    let classify = |input:&[f64]| {
        [if input.iter().eq(a_norm.iter()) {
          (1.0, 0.0) 
        } else if input.iter().eq(a_2_norm.iter()) {
          (1.0, 0.0)
        } else if input.iter().eq(c_norm.iter()) {
          (0.0, 1.0) 
        } else {
          (0.0, 1.0)
        }].to_vec()
    };

    let mut network = Network::build(35 * 35 * 3, &[10, 2]);

    let step_fn = Tanh{};

    let min_input = [0.0; 35 * 35 * 3];
    let max_input = [255.0; 35 * 35 * 3];
    let a_list: Vec<Vec<f64>> = std::iter::repeat(&a).take(50).map(|a| a.clone()).collect();
    let a_2_list: Vec<Vec<f64>> = std::iter::repeat(&a_2).take(50).map(|a| a.clone()).collect();
    let c_list: Vec<Vec<f64>> = std::iter::repeat(&c).take(50).map(|a| c.clone()).collect();
    let elems: Vec<Vec<f64>> = mix(&gen_random_elements(1000), &a_list);
    let elems: Vec<Vec<f64>> = mix(&elems, &c_list);
    let elems: Vec<Vec<f64>> = mix(&elems, &a_2_list);
    let training_data = NormalizedSet::with_bounds(&elems, NormalRange::new(&min_input, &max_input));

    println!("Training on {} elements", training_data.data.len());

    train::train_network(&mut network,
    	0.1,
    	|rate, cur, prev| if cur >= prev { rate * 0.75 } else { rate },
    	&training_data.data,
    	|a, err| a < 20,
    	&classify,
    	&step_fn);

    println!("Done Training");

    let testing_data = training_data;

    let mut good_points = Vec::new();
    let mut bad_points = Vec::new();

	for input in &[a_norm.clone(), a_2_norm.clone(), c_norm.clone()] {
    use std::iter::once;
		let expected = classify(input);
		let found = network.process(input, &step_fn);
		let error = expected
      .iter()
      .flat_map(|(x1, x2)| once(x1).chain(once(x2)))
      .zip(found.iter()).fold(0.0, |l, (&e, &f)| l + (e - f).abs().round());
		if error == 0.0 {
            println!("Good {:?} vs {:?}", expected, found);
			&mut good_points
		} else {
            println!("Bad {:?} vs {:?}", expected, found);
			&mut bad_points
		}.push(input.clone());
	}
}
