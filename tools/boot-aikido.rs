extern crate pythia;

use pythia::aikido::*;

fn main() {
  let mut repo = Repo::root();
  let mut frame = Frame::root();
  frame.debug_print_status(&repo);
  repo.debug_dump();
  let mut frame = frame.fresh(&mut repo);
  frame.debug_print_status(&repo);
  repo.debug_dump();
  {
    let mut workcopy = frame.work();
    let mut s = workcopy.mut_data().mut_string();
    *s = "print(\"Hello world!\")\n".to_string();
  }
  frame.debug_print_status(&repo);
  repo.debug_dump();
  frame.commit(&mut repo);
  frame.debug_print_status(&repo);
  repo.debug_dump();
}
