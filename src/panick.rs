use crate::algo::str::{safe_ascii};

use serde::{Serialize};
use serde::ser::{Serializer, SerializeStruct};

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::panic::{Location};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Loc(pub &'static Location<'static>);

impl Serialize for Loc {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    let mut state = serializer.serialize_struct("Loc", 3)?;
    state.serialize_field("file",
        &format!("{}", safe_ascii(self.0.file().as_bytes()))
    )?;
    state.serialize_field("line",
        &format!("{}", self.0.line())
    )?;
    state.serialize_field("col",
        &format!("{}", self.0.column())
    )?;
    state.end()
  }
}

impl Debug for Loc {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "Loc({}:{}:{})",
        safe_ascii(self.0.file().as_bytes()),
        self.0.line(),
        self.0.column(),
    )
  }
}

#[track_caller]
pub fn loc() -> Loc {
  Loc(Location::caller())
}

macro_rules! loc_ {
  () => {
    Loc(Location::caller())
  };
}
pub(crate) use loc_;
