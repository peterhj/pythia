use crate::algo::str::{safe_ascii};

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::panic::{Location};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Loc(pub &'static Location<'static>);

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

macro_rules! _errorln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.verbose >= 0;
    if print {
      println!($($arg)*);
    }
    print
  }};
}
pub(crate) use _errorln;

macro_rules! _warningln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.verbose >= 1;
    if print {
      println!($($arg)*);
    }
    print
  }};
}
pub(crate) use _warningln;

macro_rules! _infoln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.verbose >= 2;
    if print {
      println!($($arg)*);
    }
    print
  }};
}
pub(crate) use _infoln;

macro_rules! _debugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.verbose >= 3;
    if print {
      println!($($arg)*);
    }
    print
  }};
}
pub(crate) use _debugln;

macro_rules! _vdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.verbose >= 4;
    if print {
      println!($($arg)*);
    }
    print
  }};
}
pub(crate) use _vdebugln;

macro_rules! _vvdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.verbose >= 5;
    if print {
      println!($($arg)*);
    }
    print
  }};
}
pub(crate) use _vvdebugln;

macro_rules! _vvvdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.verbose >= 6;
    if print {
      println!($($arg)*);
    }
    print
  }};
}
pub(crate) use _vvvdebugln;

macro_rules! _traceln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.verbose >= 7;
    if print {
      println!($($arg)*);
    }
    print
  }};
}
pub(crate) use _traceln;
