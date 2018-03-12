use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::io::Read;
use csv;

#[derive(Clone)]
pub struct NormalRange {
	pub min: Vec<f64>,
	pub max: Vec<f64>
}

impl NormalRange {
	
	pub fn new(min: &[f64], max: &[f64]) -> NormalRange {
		NormalRange {
			min: min.to_vec(),
			max: max.to_vec()
		}
	}

	pub fn point(self: &NormalRange, p: &[f64]) -> Vec<f64> {
		p.iter()
		 .enumerate()
		 .map(|(i, p)| (p - self.min[i]) / (self.max[i] - self.min[i]))
		 .collect()
	}

	pub fn reverse(self: &NormalRange, p: &[f64]) -> Vec<f64> {
		let mut result = Vec::with_capacity(p.len());

		for i in 0..p.len() {
			result.push((p[i] * (self.max[i] - self.min[i])) + self.min[i]);
		}

		result
	}
}

pub struct NormalizedSet {
	pub data: Vec<Vec<f64>>,
	pub range: NormalRange
}

impl NormalizedSet {

	fn normalize_data(set: &[Vec<f64>], range: &NormalRange) -> Vec<Vec<f64>> {
		set.iter().map(|i| range.point(i)).collect()
	}

	pub fn with_bounds(set: &[Vec<f64>], range: NormalRange) -> NormalizedSet {
		NormalizedSet {
			data: NormalizedSet::normalize_data(set, &range),
			range: range
		}
	}
}

pub fn load_binary(file_path: &str, size: usize) -> Result<Vec<u8>, Box<Error>> {
    let mut f = File::open(file_path)?;
    let mut buffer = Vec::with_capacity(size);
    f.read(&mut buffer)?;
    Ok(buffer)
}

pub fn load_data(file_path: &str) -> Result<Vec<Vec<f64>>, Box<Error>> {
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file).has_headers(false);
    let mut results = Vec::new();

    for line in rdr.records() {
        results.push(line?.iter().map(|x| x.parse::<f64>().unwrap()).collect());
    }

    Ok(results)
}

pub fn write_data(file_path: &str, data: &[Vec<f64>]) -> Result<(), Box<Error>> {
	let mut f = File::create(file_path)?;

	for line in data {

		let mut first = true;
		
		for word in line {
			
			if !first {
				write!(f, ",")?;
			}

			write!(f, "{}", word.to_string())?;
			first = false;
		}

		write!(f, "\n")?;
	}

	Ok(())
}
