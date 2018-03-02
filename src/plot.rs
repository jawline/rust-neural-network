use gnuplot::{Figure, Caption, Color, AxesCommon};

fn extract(points: &Vec<Vec<f64>>, index: usize) -> Vec<f64> {
	points.iter().map(|x| x[index]).collect()
}

pub fn plot(title: &str, maxx: f64, miny: f64, maxy: f64, good_points: Vec<Vec<f64>>, bad_points: Vec<Vec<f64>>) {

	let x = [0.0f64, maxx];
	let y = [miny, maxy];

	let mut fg = Figure::new();

	fg.axes2d()
		.set_title(title, &[])
		.lines(&x, &y, &[Caption("The dividing line"), Color("black")])
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

pub fn plot2(title: &str, good_points: Vec<Vec<f64>>, bad_points: Vec<Vec<f64>>) {

	let mut fg = Figure::new();

	fg.axes2d()
		.set_title(title, &[])
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