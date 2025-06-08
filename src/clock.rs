use crate::algo::{SmolStr};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use serde::de::{Deserializer, Error as DError};
use serde::ser::{Serializer};
use time::{Duration, Timespec, Tm, get_time};

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::num::{ParseIntError};
use std::ops::{Sub};
use std::str::{FromStr};

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

impl Serialize for Timestamp {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&self.to_string())
  }
}

impl<'d> Deserialize<'d> for Timestamp {
  fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Timestamp, D::Error> {
    let s = SmolStr::deserialize(deserializer)?;
    s.parse().map_err(|e| <D::Error as DError>::custom(format!("{:?}", e)))
  }
}

#[cfg(feature = "pyo3")]
impl<'py> FromPyObject<'py> for Timestamp {
  fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
    let s: String = obj.extract()?;
    match s.parse() {
      Ok(this) => Ok(this),
      Err(e) => panic!("bug: {:?}", e),
    }
  }
}

#[derive(Debug)]
pub enum Rfc3339NsecParseError {
  Split,
  ParseInt(ParseIntError),
}

impl From<ParseIntError> for Rfc3339NsecParseError {
  fn from(e: ParseIntError) -> Rfc3339NsecParseError {
    Rfc3339NsecParseError::ParseInt(e)
  }
}

impl FromStr for Timestamp {
  type Err = Rfc3339NsecParseError;

  fn from_str(s: &str) -> Result<Timestamp, Rfc3339NsecParseError> {
    //println!("DEBUG: Timestamp::from_str: s = {:?}", s);

    /*// Version 1 (courtesy of DeepSeek-V3).
    let parts: Vec<&str> = s.split(&['-', 'T', ':', '.', '+', 'Z'][..]).collect();

    if parts.len() < 6 {
      //return Err(ParseIntError::InvalidDigit);
      return Err(Rfc3339NsecParseError::Split);
    }

    let year = parts[0].parse::<u32>()?;
    let month = parts[1].parse::<u8>()?;
    let day = parts[2].parse::<u8>()?;
    let hour = parts[3].parse::<u8>()?;
    let minute = parts[4].parse::<u8>()?;
    let second = parts[5].parse::<u8>()?;

    let nanosecond = if parts.len() > 6 {
      let nanos = parts[6];
      let len = nanos.len().min(9); // Limit to nanoseconds
      format!("{:0<9}", &nanos[..len]).parse::<u32>()?
    } else {
      0
    };

    /*let offset_hours = if parts.len() > 7 {
      parts[7].parse::<i8>()?
    } else {
      0
    };

    let offset_minutes = if parts.len() > 8 {
      parts[8].parse::<i8>()?
    } else {
      0
    };*/
    */

    // Version 2 (courtesy of DeepSeek-V3).
    let date_time: Vec<&str> = s.split('T').collect();
    if date_time.len() != 2 {
      return Err(Rfc3339NsecParseError::Split);
    }

    let date_parts: Vec<&str> = date_time[0].split('-').collect();
    if date_parts.len() != 3 {
      return Err(Rfc3339NsecParseError::Split);
    }

    let time_parts: Vec<&str> = date_time[1].split(&[':', '.', '+', '-'][..]).collect();
    if time_parts.len() < 3 {
      return Err(Rfc3339NsecParseError::Split);
    }

    let year = date_parts[0].parse::<u32>()?;
    let month = date_parts[1].parse::<u8>()?;
    let day = date_parts[2].parse::<u8>()?;
    //println!("DEBUG: Timestamp::from_str: year = {}", year);

    let hour = time_parts[0].parse::<u8>()?;
    let minute = time_parts[1].parse::<u8>()?;
    let second = time_parts[2].parse::<u8>()?;

    let nanosecond = if time_parts.len() > 3 {
      let nanos = time_parts[3];
      let len = nanos.len().min(9); // Limit to nanoseconds
      format!("{:0<9}", &nanos[..len]).parse::<u32>()?
    } else {
      0
    };

    let tm = Tm{
      tm_year: (year as i32) - 1900,
      tm_mon: (month as i32) - 1,
      tm_mday: day as _,
      tm_hour: hour as _,
      tm_min: minute as _,
      tm_sec: second as _,
      tm_nsec: nanosecond as _,
      tm_utcoff: 0,
      tm_isdst: -1,
      tm_yday: -1,
      tm_wday: -1,
    };

    Ok(Timestamp{inner: tm.to_timespec()})
  }
}

impl Debug for Timestamp {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", self.inner.utc().rfc3339_nsec())
  }
}

impl Timestamp {
  pub fn fresh() -> Timestamp {
    Timestamp{inner: get_time()}
  }

  // TODO: type.
  pub fn utc(&self) -> Tm {
    self.inner.utc()
  }

  pub fn to_string(&self) -> String {
    format!("{}", self.utc().rfc3339_nsec())
  }
}

impl Sub<Timestamp> for Timestamp {
  type Output = Timedelta;

  fn sub(self, other: Timestamp) -> Timedelta {
    Timedelta{inner: self.inner - other.inner}
  }
}
