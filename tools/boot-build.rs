extern crate pythia;

use pythia::build::*;

fn main() {
  let trip = triple();
  println!("{}/{}", trip.arch, trip.system);
}
