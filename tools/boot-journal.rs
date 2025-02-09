extern crate pythia;

use pythia::journal::*;

fn main() {
  let mut journal = DevelJournal_::cold_start();
  journal.append(&BootTest);
}
