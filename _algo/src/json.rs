pub use serde_json::{Value as JsonValue};
pub use serde_json_fmt::{JsonFormat};
use serde::{Deserialize};
use serde_json::{Deserializer};

use std::io::{Read};

// see: https://github.com/serde-rs/json/issues/632

pub fn deserialize_json_value<R: Read>(reader: R) -> Result<JsonValue, ()> {
  let mut des = Deserializer::from_reader(reader);
  let v = JsonValue::deserialize(&mut des).map_err(|_| ())?;
  Ok(v)
}
