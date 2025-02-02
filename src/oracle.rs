use crate::_extlib::{_EXTLIB};
use crate::algo::str::{SafeStr};
use crate::clock::{Timestamp};

use pyo3::prelude::*;
use pyo3::{IntoPyObjectExt};
use serde::{Serialize, Deserialize};
use serde::de::{Deserializer};

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::str::{FromStr};

#[derive(Clone, Copy, Serialize, Default, Debug)]
//#[derive(Clone, Copy, Serialize, Deserialize, Default, Debug)]
//#[serde(untagged)]
pub enum ApproxOracleModel {
  #[default]
  #[serde(rename = "deepseek-v3-chat-20241226")]
  DeepSeek_V3_Chat_20241226,
  #[serde(rename = "deepseek-r1-20250120")]
  DeepSeek_R1_20250120,
}

impl<'d> Deserialize<'d> for ApproxOracleModel {
  fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<ApproxOracleModel, D::Error> {
    struct _Visitor;

    impl<'de> serde::de::Visitor<'de> for _Visitor {
      type Value = ApproxOracleModel;

      fn expecting(&self, formatter: &mut Formatter) -> FmtResult {
        formatter.write_str("a string representing ApproxOracleModel")
      }

      fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
      where E: serde::de::Error,
      {
        match value.parse() {
          Ok(this) => Ok(this),
          //_ => panic!("bug"),
          _ => Err(E::custom(format!("unknown ApproxOracleModel variant: {}", value)))
        }
      }
    }

    deserializer.deserialize_str(_Visitor)
  }
}

impl<'py> FromPyObject<'py> for ApproxOracleModel {
  fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
    let s: String = obj.extract()?;
    //println!("DEBUG: ApproxOracleModel::extract_bound: s = {:?}", s);
    match s.parse() {
      Ok(this) => Ok(this),
      _ => panic!("bug"),
    }
  }
}

impl FromStr for ApproxOracleModel {
  type Err = ();

  fn from_str(s: &str) -> Result<ApproxOracleModel, ()> {
    Ok(match s {
      "deepseek-v3-chat-20241226" => ApproxOracleModel::DeepSeek_V3_Chat_20241226,
      "deepseek-r1-20250120" => ApproxOracleModel::DeepSeek_R1_20250120,
      _ => return Err(())
    })
  }
}

#[derive(Clone, Serialize, Deserialize, IntoPyObject, Debug)]
pub struct ApproxOracleRequest {
  pub key: i64,
  pub query: SafeStr,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleResExtra {
  pub data: SafeStr,
  pub t0:   Timestamp,
  pub t1:   Timestamp,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleExtraItem {
  pub res:  Option<ApproxOracleResExtra>,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleItem<K=i64> {
  pub key: K,
  pub query: SafeStr,
  pub thinking: Option<SafeStr>,
  pub value: SafeStr,
  pub timestamp: Timestamp,
  pub model: ApproxOracleModel,
  pub extra: Option<ApproxOracleExtraItem>,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleTestItem {
  pub timestamp: Timestamp,
  pub model: ApproxOracleModel,
}

pub struct ApproxOracleInterface {
  this: PyObject,
}

impl ApproxOracleInterface {
  pub fn init() -> ApproxOracleInterface {
    let this = _EXTLIB._approx_oracle_interface();
    ApproxOracleInterface{this}
  }

  pub fn default_model(&self) -> String {
    Python::with_gil(|py| -> PyResult<_> {
      self.this
          .getattr(py, "default_model")?
          .extract::<_>(py)
    }).unwrap()
  }

  pub fn default_timeout(&self) -> i32 {
    Python::with_gil(|py| -> PyResult<_> {
      self.this
          .getattr(py, "default_timeout")?
          .extract::<_>(py)
    }).unwrap()
  }

  pub fn concurrency(&self) -> i32 {
    Python::with_gil(|py| -> PyResult<_> {
      self.this
          .getattr(py, "concurrency")?
          .extract::<_>(py)
    }).unwrap()
  }

  pub fn len(&self) -> usize {
    Python::with_gil(|py| -> PyResult<_> {
      self.this
          .call_method0(py, "__len__")?
          .extract::<_>(py)
    }).unwrap()
  }

  pub fn put(&self, item: ApproxOracleRequest) -> () {
    Python::with_gil(|py| -> PyResult<_> {
      self.this
          .call_method1(py, "put", (item.into_pyobject(py)?,))
    }).unwrap();
  }

  pub fn get(&self) -> Option<ApproxOracleItem> {
    Python::with_gil(|py| -> PyResult<_> {
      let item = self.this
          .call_method0(py, "get")?
          .into_bound_py_any(py)?;
      match item.extract() {
        Err(e) => panic!("bug: {:?}", e),
        Ok(item) => Ok(item)
      }
    }).unwrap()
  }

  pub fn get_test(&self) -> Option<ApproxOracleTestItem> {
    Python::with_gil(|py| -> PyResult<_> {
      let item = self.this
          .call_method0(py, "get_test")?
          .into_bound_py_any(py)?;
      match item.extract() {
        Err(e) => panic!("bug: {:?}", e),
        Ok(item) => Ok(item)
      }
    }).unwrap()
  }
}
