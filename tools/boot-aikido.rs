extern crate pythia;

use pythia::aikido::*;

fn main() {
  let mut store = Store::root();
  let mut frame = Frame::root();
  frame.debug_print_status(&store);
  store.debug_print_digest();
  let mut frame = frame.fresh(&mut store);
  frame.debug_print_status(&store);
  store.debug_print_digest();
  {
    let mut workcopy = frame.modify();
    let mut s = workcopy.mut_data().mut_string();
    *s = "print(\"Hello world!\")\n".to_string();
  }
  frame.debug_print_status(&store);
  store.debug_print_digest();
  frame.commit(&mut store);
  frame.debug_print_status(&store);
  store.debug_print_digest();
}
