extern crate pythia;

use pythia::store::*;

fn main() {
  let mut store = DevelStore_::cold_start();
  store.append(&BootTest);
}
