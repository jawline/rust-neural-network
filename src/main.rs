extern crate csv;
extern crate rand;
extern crate gnuplot;
extern crate rust_graph;

mod data;
mod neuron;
mod train;
mod circle;
mod plot;
mod steps;
mod network;

#[cfg(test)]
mod tests;

use data::{ write_data, load_data };
use circle::gen_even_set;

fn main() {

	write_data("./weighted.csv", &gen_even_set(5000));
}