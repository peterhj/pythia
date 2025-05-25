extern crate pyo3;
extern crate serde;
extern crate serde_json;

use pyo3::prelude::*;
use serde::*;
use serde_json::{Deserializer, Value};

use std::io::{Cursor, Read};

// see: https://github.com/serde-rs/json/issues/632

pub fn deserialize_value<R: Read>(reader: R) -> Result<Value, ()> {
  let mut des = Deserializer::from_reader(reader);
  let v = Value::deserialize(&mut des).map_err(|_| ())?;
  Ok(v)
}

#[pymodule]
#[pyo3(name = "_extmod")]
pub fn _extmod(mod_: &Bound<'_, PyModule>) -> PyResult<()> {
  {
    let json_mod = PyModule::new(mod_.py(), "json")?;
    json_mod.add_function(wrap_pyfunction!(reloads2, &json_mod)?)?;
    mod_.add_submodule(&json_mod)?;
  }
  Ok(())
}

#[pyfunction]
#[pyo3(name = "reloads2")]
pub fn reloads2(s: &str) -> u64 {
  let mut reader = Cursor::new(s.as_bytes());
  let _v = deserialize_value(&mut reader);
  reader.position()
}
