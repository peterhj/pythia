extern crate pythia;

use pythia::clock::{Timedelta, Timestamp};
use pythia::interp::*;
use pythia::interp_test::*;
use pythia::smp::{init_smp};
use pythia::tap::*;
use pythia::test_data::*;

fn main() {
  let test_data_cfg = TestDataConfig::test_last();
  println!("DEBUG: boot: test data config = {:?}", test_data_cfg);
  let prover = InterpTestsProver::from(test_data_cfg);
  EchoTAPParser::parse(prover);
}
