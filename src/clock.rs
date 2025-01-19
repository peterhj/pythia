use time::{Duration, Timespec, Tm, get_time};

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Sub};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Timedelta {
  inner: Duration,
}

impl Display for Timedelta {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}.{:09}", self.inner.num_seconds(), self.inner.nanos_mod_sec())
  }
}

impl Default for Timedelta {
  fn default() -> Timedelta {
    Timedelta::zero()
  }
}

impl Timedelta {
  pub fn zero() -> Timedelta {
    Timedelta{inner: Duration::zero()}
  }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Timestamp {
  inner: Timespec,
}

impl Timestamp {
  pub fn fresh() -> Timestamp {
    Timestamp{inner: get_time()}
  }

  // TODO: type.
  pub fn utc(&self) -> Tm {
    self.inner.utc()
  }
}

impl Sub<Timestamp> for Timestamp {
  type Output = Timedelta;

  fn sub(self, other: Timestamp) -> Timedelta {
    Timedelta{inner: self.inner - other.inner}
  }
}
