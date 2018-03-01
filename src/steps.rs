use neuron::Neuron;

pub type StepFn = Fn(&Neuron, f64) -> f64;

pub const HEAVISIDE : &'static Fn(&Neuron, f64) -> f64 = &|_, v| if v > 0.0 { 1.0 } else { 0.0 };