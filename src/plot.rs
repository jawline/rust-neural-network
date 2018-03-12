use rust_graph::Graph;

fn extract(points: &Vec<Vec<f64>>) -> Vec<(f64, f64)> {
	points.iter().map(|v| (v[0], v[1])).collect()
}

pub fn plot(title: &str, maxx: f64, miny: f64, maxy: f64, good_points: Vec<Vec<f64>>, bad_points: Vec<Vec<f64>>) {
	Graph::new(title)
		.line(((0.0, miny), (maxx, maxy)), "black")
		.points(&extract(&bad_points), "red")
		.points(&extract(&good_points), "green")
		.show();
}

pub fn plot2(title: &str, guide_points: Vec<Vec<f64>>, good_points: Vec<Vec<f64>>, bad_points: Vec<Vec<f64>>) {
	Graph::new(title)
		.points(&extract(&guide_points), "blue")
		.points(&extract(&bad_points), "red")
		.points(&extract(&good_points), "green")
		.show();
}

pub fn plot3(title: &str, good_points: Vec<Vec<f64>>, bad_points: Vec<Vec<f64>>) {
	Graph::new(title)
		.points(&extract(&bad_points), "red")
		.points(&extract(&good_points), "green")
		.show();
}