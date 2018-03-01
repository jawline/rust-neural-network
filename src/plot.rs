use gnuplot::{Figure, Caption, Color};

fn extract(points: &Vec<Vec<f64>>, index: usize) -> Vec<f64> {
	points.iter().map(|x| x[index]).collect()
}

pub fn plot(maxx: f64, miny: f64, maxy: f64, good_points: Vec<Vec<f64>>, bad_points: Vec<Vec<f64>>) {

	let x = [0.0f64, maxx];
	let y = [miny, maxy];

	let mut fg = Figure::new();

	fg.axes2d().lines(&x, &y, &[Caption("The dividing line"), Color("black")])
		.points(
			&(extract(&good_points, 0)),
			&(extract(&good_points, 1)),
			&[Color("green")]
		).points(
			&(extract(&bad_points, 0)),
			&(extract(&bad_points, 1)),
			&[Color("red")]
		);
	fg.show();
}