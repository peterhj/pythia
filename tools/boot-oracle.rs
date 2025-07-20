extern crate pythia;

use pythia::oracle::*;

fn main() {
  let aoi = ApproxOracleInterface::init();
  println!("DEBUG: default model = {:?}", aoi.default_model());
  println!("DEBUG: default timeout = {}", aoi.default_timeout());
  println!("DEBUG: concurrency = {}", aoi.concurrency());
  println!("DEBUG: len = {}", aoi.len());
  //let req = ApproxOracleRequest{key: 0, query: "hi".into()};
  let item = aoi.poll_test();
  println!("DEBUG: item = {:?}", item);
  // TODO
  let req = ApproxOracleRequest{key: None, query: vec!["Hi!".into()], .. Default::default()};
  aoi.put(req);
  let item = aoi.poll();
  println!("DEBUG: item = {:?}", item);
}
