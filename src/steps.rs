use neuron::Neuron;

pub trait StepFn {
  fn transfer(&Neuron, f64) -> f64;
  fn derive(&Neuron, f64) -> f64;
};

pub struct Heaviside {}
pub struct Sigmoid {}

impl StepFn for Heaviside {
  fn transfer(n:&Neuron, v:f64) -> f64 { if v > 0.5 { 1.0 } else { 0.0 } }
  fn derive(n:&Neuron, v:f64) -> f64 { v }
}

impl StepFn for Sigmoid {
  fn transfer(n:&Neuron, v:f64) -> f64 { 1.0 / (1.0 + -v.exp()) }
  fn derive(n:&Neuron, v:f64) -> f64 { v * (1.0 - v) }
}

pub const HEAVISIDE : &'static Fn(&Neuron, f64) -> f64 = &|_, v| if v > 0.0 { 1.0 } else { 0.0 };
pub const HEAVISIDE_DERIVITIVE : &'static Fn(&Neuron, f64) -> f64 = &|_, v| if v > 0.0 { 1.0 } else { 0.0 };

pub const TRANSFER : &'static Fn(&Neuron, f64) -> f64 = &|_, v| 1.0 / (1.0 + (-v).exp());
pub const TRANSFER_DERIVITIVE : &'static Fn(f64) -> f64 = &|v| v * (1.0 - v);
