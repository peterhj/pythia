use pyo3::prelude::*;
use pyo3::types::{PyDict};

pub fn _new_tokenizer() -> PyObject {
  Python::with_gil(|py| {
    let tokenizers = PyModule::import_bound(py, "_tokenizers").unwrap();
    // TODO: why does this `getattr` not need `py`, but the `getattr`
    // in the second block (below) does need `py`? wtf?
    let tokenizer_cls = tokenizers.getattr("Tokenizer").unwrap();
    let tok = tokenizer_cls.call_method1("from_file", ("tokenizer.json",)).unwrap();
    let tok = tok.into_py(py);
    // NB: Let's test the tokenizer.
    let encode = tok.getattr(py, "encode").unwrap();
    let args = ("Hello world! ! !!!",);
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("add_special_tokens", false).unwrap();
    let result = encode.call_bound(py, args, Some(&kwargs)).unwrap();
    let result = result.getattr(py, "ids").unwrap();
    let enc = result.extract::<Vec<i32>>(py).unwrap();
    println!("DEBUG: _new_tokenizer: enc = {:?}", enc);
    tok
  })
}

pub struct Tokenizer {
  inner: PyObject,
}

impl Default for Tokenizer {
  fn default() -> Tokenizer {
    let inner = _new_tokenizer();
    Tokenizer{inner}
  }
}

impl Tokenizer {
  pub fn encode(&self, s: &str) -> Vec<i64> {
    Python::with_gil(|py| {
      let encode = self.inner.getattr(py, "encode").unwrap();
      let args = (s,);
      let kwargs = PyDict::new_bound(py);
      kwargs.set_item("add_special_tokens", false).unwrap();
      let result = encode.call_bound(py, args, Some(&kwargs)).unwrap();
      let result = result.getattr(py, "ids").unwrap();
      let enc = result.extract::<Vec<i64>>(py).unwrap();
      //println!("DEBUG: _new_tokenizer: enc = {:?}", enc);
      enc
    })
  }
}
