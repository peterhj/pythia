extern crate fxhash2;
#[cfg(feature = "pyo3")]
extern crate pyo3;
extern crate rclite;
extern crate serde;
extern crate smol_str;
extern crate triomphe;

pub use fxhash2::{FxHashMap, FxHashSet};
pub use smol_str::{SmolStr};

pub use std::collections::{BTreeMap, BTreeSet};
pub use std::rc::{Rc};
pub use std::sync::{Arc};

pub mod arc;
pub mod blake2s;
pub mod cell;
pub mod extract;
pub mod rc;
pub mod str;
pub mod token;

pub trait OptionExt<T> {
  fn push(&mut self, val: T) -> Option<T>;
  fn pop(&mut self) -> Option<T>;
}

impl<T> OptionExt<T> for Option<T> {
  fn push(&mut self, val: T) -> Option<T> {
    let prev_val = self.take();
    *self = Some(val);
    prev_val
  }

  fn pop(&mut self) -> Option<T> {
    let prev_val = self.take();
    *self = None;
    prev_val
  }
}
