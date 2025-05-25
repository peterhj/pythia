extern crate serde;
extern crate serde_json;

use serde::*;
use serde_json::{Deserializer, Value};

use std::io::{Cursor, Read};

// see: https://github.com/serde-rs/json/issues/632

fn deserialize_part<R: Read>(data: R) -> Result<Value, ()> {
  let mut des = Deserializer::from_reader(data);
  let v = Value::deserialize(&mut des).map_err(|_| ())?;
  Ok(v)
}

fn main() {
  let t = b"{\"hello\": \"world\"}";
  println!("{}", t.len());
  let t = b"{\"hello\": \"world\"} \"goodbye\"";
  println!("{}", t.len());
  let mut r = Cursor::new(t);
  println!("{}", r.position());
  let v = deserialize_part(&mut r);
  println!("{}", r.position());
  println!("{:?}", v);
}
