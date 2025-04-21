extern crate pythia;

use pythia::journal::*;
use pythia::oracle::*;

fn main() {
  let mut journal = DevelJournal_::cold_start();
  journal.append(&BootTest);
  let oraclei = ApproxOracleInterface::init();
  println!("DEBUG: default model = {:?}", oraclei.default_model());
  println!("DEBUG: default timeout = {}", oraclei.default_timeout());
  println!("DEBUG: concurrency = {}", oraclei.concurrency());
  println!("DEBUG: len = {}", oraclei.len());
  /*
  //let req = ApproxOracleRequest{key: 0, query: "hi".into()};
  let item = oraclei.poll_test().unwrap();
  println!("DEBUG: item = {:?}", item);
  journal.append(&item);
  */
  // TODO
  //let model = Default::default();
  //let model = ApproxOracleModel::DeepSeek_V3_Chat_20241226;
  //let model = ApproxOracleModel::DeepSeek_R1_20250120;
  let model = ApproxOracleModel::DeepSeek_V3_Chat_20250324;
  //let req = ApproxOracleRequest{key: 0, query: "Hi!".into()};
  let req = ApproxOracleRequest{key: 0, query: "I'm interested in learning about the Test Anything Protocol (TAP).".into(), model};
  oraclei.put(req);
  let item = oraclei.poll().unwrap();
  println!("DEBUG: item = {:?}", item);
  journal.append(&item);
}
