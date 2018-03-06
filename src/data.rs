use std::error::Error;
use std::fs::File;
use std::io::Write;
use csv;

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