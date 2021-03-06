pub trait StepFn {
  fn transfer(&self, f64) -> f64;
  fn derive(&self, f64) -> f64;
}

pub struct Heaviside {}
pub struct Sigmoid {}
pub struct Tanh {}

pub struct ReLU {
	pub scalar: f64
}

impl StepFn for Heaviside {
  fn transfer(&self, v:f64) -> f64 { if v > 0.5 { 1.0 } else { 0.0 } }
  fn derive(&self, _:f64) -> f64 { 1.0 }
}

impl StepFn for Sigmoid {
  fn transfer(&self, v:f64) -> f64 { 1.0 / (1.0 + -v.exp()) }
  fn derive(&self, v:f64) -> f64 { v * (1.0 - v) }
}

impl StepFn for Tanh {
  fn transfer(&self, v:f64) -> f64 { v.tanh() }
  fn derive(&self, v:f64) -> f64 { 1.0 - v.tanh().powi(2) }
}

impl StepFn for ReLU {
  
  fn transfer(&self, v:f64) -> f64 { 
  	if v > 0.0 {
  		v
  	} else {
  		self.scalar * v
  	}
  }
  
  fn derive(&self, v:f64) -> f64 {
  	if v > 0.0 {
      1.0
    } else {
      0.0
    }
  }

}