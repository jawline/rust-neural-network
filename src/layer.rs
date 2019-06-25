use steps::StepFn;
use std::marker::PhantomData;
use rand::Rng;
use rand::distributions::{Uniform};
use neuron::Neuron;

pub trait Layer {
  fn input_size(&self) -> usize;
  fn output_size(&self) -> usize;
  fn process<F: StepFn>(&mut self, input: &[f64], output: &mut [f64], step: &F);
  fn transfer_errors<F: StepFn>(&mut self, last: &[f64], next: &mut [f64], step: &F);
}

pub struct InputLayer {
  pub size: usize
}

impl InputLayer {
  pub fn new(size: usize) -> InputLayer {
    InputLayer { size: size }
  }
}

impl Layer for InputLayer {
  fn input_size(&self) -> usize {
    self.size
  }

  fn output_size(&self) -> usize {
    self.size
  }

  fn process<F: StepFn>(&mut self, input: &[f64], output: &mut [f64], step: &F) {
    for i in 0..self.input_size() {
      output[i] = input[i]
    }
  }

  fn transfer_errors<F: StepFn>(&mut self, last: &[f64], next: &mut [f64], step: &F) {
    //Input layers ignore errors
    for i in 0..self.input_size() {
      next[i] = last[i];
    } 
  }
}

pub struct NeuronLayer {
  pub neurons: Vec<Neuron>,
  is: usize
}

impl NeuronLayer {
  pub fn new(input_size: usize, output_size: usize) -> NeuronLayer {
    NeuronLayer {
      neurons: (0..output_size).map(|_| Neuron::random(input_size)).collect(),
      is: input_size
    }
  }
}

impl Layer for NeuronLayer {
  fn input_size(&self) -> usize { self.is }
  fn output_size(&self) -> usize { self.neurons.len() }

  fn process<F: StepFn>(&mut self, input: &[f64], output: &mut [f64], step: &F) {
    self
      .neurons
      .iter_mut()
      .enumerate()
      .for_each(|(i, ref mut neuron)| {
        output[i] = neuron.activate(input, step)
      });
  }

  fn transfer_errors<F: StepFn>(&mut self, last: &[f64], next: &mut [f64], step: &F) { 
		self.neurons
		    .iter_mut()
		    .enumerate()
		    .for_each(|(i, ref mut neuron)| {
		    	next[i] = last[i] * step.derive(neuron.output)
		    });
  }
}
