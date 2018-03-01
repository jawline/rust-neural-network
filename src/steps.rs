use neuron::Neuron;

pub const HEAVISIDE : &'static Fn(&Neuron, f64) -> f64 = &|_, v| if v > 0.0 { 1.0 } else { 0.0 };