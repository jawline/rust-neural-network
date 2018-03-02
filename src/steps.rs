use neuron::Neuron;

pub const HEAVISIDE : &'static Fn(&Neuron, f64) -> f64 = &|_, v| if v > 0.0 { 1.0 } else { 0.0 };
pub const TRANSFER : &'static Fn(&Neuron, f64) -> f64 = &|_, v| 1.0 / (1.0 + (-v).exp());