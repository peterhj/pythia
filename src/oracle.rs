use crate::_extlib::{_EXTLIB};
use crate::algo::{BTreeMap};
use crate::algo::str::{SafeStr};
use crate::clock::{Timestamp};
use crate::journal::{JournalExt, JournalEntryExt, JournalEntrySort_};

use pyo3::prelude::*;
use pyo3::{IntoPyObjectExt};
use serde::{Serialize, Deserialize};
use serde::de::{Deserializer};

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::str::{FromStr};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Default, Debug)]
//#[derive(Clone, Copy, Serialize, Deserialize, Default, Debug)]
//#[serde(untagged)]
pub enum ApproxOracleModel {
  #[default]
  #[serde(rename = "default")]
  Default,
  #[serde(rename = "deepseek-v3-chat-20250324")]
  DeepSeek_V3_Chat_20250324,
  #[serde(rename = "deepseek-v3-chat-20241226")]
  DeepSeek_V3_Chat_20241226,
  #[serde(rename = "deepseek-v3-chat-20250324-hyperbolic")]
  DeepSeek_V3_Chat_20250324_Hyperbolic,
  #[serde(rename = "deepseek-v3-chat-20241226-together")]
  DeepSeek_V3_Chat_20241226_Together,
  #[serde(rename = "deepseek-r1-20250528")]
  DeepSeek_R1_20250528,
  #[serde(rename = "deepseek-r1-20250120")]
  DeepSeek_R1_20250120,
  #[serde(rename = "xai-grok-3-mini-beta-20250418")]
  XAI_Grok_3_Mini_Beta_20250418,
  #[serde(rename = "xai-grok-3-beta-20250418")]
  XAI_Grok_3_Beta_20250418,
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
      "deepseek-v3-chat-20250324" |
      "\"deepseek-v3-chat-20250324\"" => {
        ApproxOracleModel::DeepSeek_V3_Chat_20250324
      }
      "deepseek-r1-20250528" |
      "\"deepseek-r1-20250528\"" => {
        ApproxOracleModel::DeepSeek_R1_20250528
      }
      "xai-grok-3-mini-beta-20250418" |
      "\"xai-grok-3-mini-beta-20250418\"" => {
        ApproxOracleModel::XAI_Grok_3_Mini_Beta_20250418
      }
      "xai-grok-3-beta-20250418" |
      "\"xai-grok-3-beta-20250418\"" => {
        ApproxOracleModel::XAI_Grok_3_Beta_20250418
      }
      _ => return Err(())
    })
  }
}

#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct ApproxOracleRequest {
  pub key: i64,
  pub query: SafeStr,
  pub model: ApproxOracleModel,
}

impl ApproxOracleRequest {
  pub fn _into_py(self) -> ApproxOracleRequestIntoPy {
    ApproxOracleRequestIntoPy{
      key: self.key,
      query: self.query,
      // FIXME: this double-quotes the enum-as-str.
      model: serde_json::to_string(&self.model).unwrap().into(),
    }
  }
}

#[derive(Clone, Serialize, Deserialize, IntoPyObject, Debug)]
pub struct ApproxOracleRequestIntoPy {
  pub key: i64,
  pub query: SafeStr,
  pub model: SafeStr,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleResponseItem {
  pub data: SafeStr,
  pub t0:   Timestamp,
  pub t1:   Timestamp,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleSampleItem {
  pub temperature: Option<f64>,
  pub top_p: Option<f64>,
  pub top_k: Option<i64>,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleExceptItem {
  pub exc_type: SafeStr,
  pub exc_str: SafeStr,
  pub stack_trace: SafeStr,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleExtraItem {
  pub res:  Option<ApproxOracleResponseItem>,
  pub exc:  Option<ApproxOracleExceptItem>,
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleItem<K=Option<SafeStr>> {
  //pub timestamp: Option<Timestamp>,
  // TODO: optional key incompat w/ kqmap (below).
  pub key: K,
  //pub key: Option<K>,
  pub query: SafeStr,
  pub tag: Option<SafeStr>,
  pub ctr: i64,
  pub model: ApproxOracleModel,
  pub sample: Option<ApproxOracleSampleItem>,
  pub think: Option<SafeStr>,
  pub value: Option<SafeStr>,
  pub extra: Option<ApproxOracleExtraItem>,
}

impl JournalEntryExt for ApproxOracleItem {
  fn _sort(&self) -> JournalEntrySort_ {
    JournalEntrySort_::ApproxOracle
  }

  fn _maybe_as_approx_oracle_item(&self) -> Option<&ApproxOracleItem> {
    Some(self)
  }
}

impl ApproxOracleItem {
  pub fn _into_key_item(&self) -> ApproxOracleKeyItem {
    ApproxOracleKeyItem{
      key: self.key.clone(),
      query: self.query.clone(),
      ctr: self.ctr,
      model: self.model,
    }
  }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleKeyItem {
  pub key: Option<SafeStr>,
  pub query: SafeStr,
  // NB: tag should _not_ be part of key.
  /*pub tag: Option<SafeStr>,*/
  pub ctr: i64,
  pub model: ApproxOracleModel,
  // TODO: should sample be part of key or value?
  /*pub sample: Option<ApproxOracleSampleItem>,*/
}

#[derive(Clone, Serialize, Deserialize, FromPyObject, Debug)]
pub struct ApproxOracleTestItem {
  pub timestamp: Timestamp,
  pub model: ApproxOracleModel,
}

impl JournalEntryExt for ApproxOracleTestItem {
  fn _sort(&self) -> JournalEntrySort_ {
    JournalEntrySort_::ApproxOracleTest
  }
}

pub struct ApproxOracleWorker {
  this: PyObject,
}

impl Clone for ApproxOracleWorker {
  fn clone(&self) -> ApproxOracleWorker {
    let this = Python::with_gil(|py| {
      self.this.clone_ref(py)
    });
    ApproxOracleWorker{this}
  }
}

impl ApproxOracleWorker {
  pub fn init(concurrency: u32) -> ApproxOracleWorker {
    let this = _EXTLIB._approx_oracle_worker(concurrency);
    ApproxOracleWorker{this}
  }
}

pub struct ApproxOracleInterface {
  this: PyObject,
}

impl ApproxOracleInterface {
  pub fn init() -> ApproxOracleInterface {
    let this = _EXTLIB._approx_oracle_interface();
    ApproxOracleInterface{this}
  }

  pub fn init_with_worker(worker: ApproxOracleWorker) -> ApproxOracleInterface {
    let this = _EXTLIB._approx_oracle_interface_with_worker(worker.this);
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
          .call_method1(py, "put", (item._into_py().into_pyobject(py)?,))
    }).unwrap();
  }

  pub fn poll(&self) -> Option<ApproxOracleItem> {
    Python::with_gil(|py| -> PyResult<_> {
      let item = self.this
          .call_method0(py, "poll")?
          .into_bound_py_any(py)?;
      match item.extract() {
        Err(e) => panic!("bug: {:?}", e),
        Ok(item) => Ok(item)
      }
    }).unwrap()
  }

  pub fn poll_test(&self) -> Option<ApproxOracleTestItem> {
    Python::with_gil(|py| -> PyResult<_> {
      let item = self.this
          .call_method0(py, "poll_test")?
          .into_bound_py_any(py)?;
      match item.extract() {
        Err(e) => panic!("bug: {:?}", e),
        Ok(item) => Ok(item)
      }
    }).unwrap()
  }
}

pub struct ApproxOracleIndex {
  iface: ApproxOracleInterface,
  // FIXME: probably want a better data structure.
  kqmap: BTreeMap<(Option<SafeStr>, SafeStr), (i64, ApproxOracleItem)>,
}

impl ApproxOracleIndex {
  pub fn init() -> ApproxOracleIndex {
    let iface = ApproxOracleInterface::init();
    let kqmap = BTreeMap::new();
    ApproxOracleIndex{
      iface,
      kqmap,
    }
  }

  pub fn put(&self, item: ApproxOracleRequest) -> () {
    self.iface.put(item)
  }

  pub fn poll(&self) -> Option<ApproxOracleItem> {
    self.iface.poll()
  }

  pub fn commit<J: JournalExt>(&mut self, journal: &mut J, item: &ApproxOracleItem) -> i64 {
    let result = journal.append(item);
    let jnum = result.eid;
    self.kqmap.insert((item.key.clone(), item.query.clone()), (jnum, item.clone()));
    jnum
  }

  pub fn get(&self, key: Option<&SafeStr>, query: &SafeStr) -> Option<ApproxOracleItem> {
    // TODO: temporary default key=0.
    // FIXME: tuple of borrowed str?
    match self.kqmap.get(&(key.cloned(), query.clone())) {
      None => None,
      Some(&(_jnum, ref item)) => {
        Some(item.clone())
      }
    }
  }
}
