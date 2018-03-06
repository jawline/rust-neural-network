use neuron::Neuron;

pub trait StepFn {
  fn transfer(&Neuron, f64) -> f64,
  fn derive(&Neuron, f64) -> f64
};

pub const HEAVISIDE : &'static Fn(&Neuron, f64) -> f64 = &|_, v| if v > 0.0 { 1.0 } else { 0.0 };
pub const HEAVISIDE_DERIVITIVE : &'static Fn(&Neuron, f64) -> f64 = &|_, v| if v > 0.0 { 1.0 } else { 0.0 };

pub const TRANSFER : &'static Fn(&Neuron, f64) -> f64 = &|_, v| 1.0 / (1.0 + (-v).exp());
pub const TRANSFER_DERIVITIVE : &'static Fn(f64) -> f64 = &|v| v * (1.0 - v);
