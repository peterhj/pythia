use once_cell::sync::{Lazy};
use pyo3::prelude::*;
use pyo3_ffi::{c_str};

pub static _EXTLIB: Lazy<Extlib> = Lazy::new(|| Extlib::init());

pub struct Extlib {
  approx_oracle: Py<PyModule>,
}

impl Extlib {
  pub fn init() -> Extlib {
    let approx_oracle_py = c_str!(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/_extlib/_approx_oracle.py"
    )));
    let approx_oracle: Py<_> = Python::with_gil(|py| -> PyResult<Py<_>> {
      Ok(PyModule::from_code(py,
          approx_oracle_py,
          c_str!("_extlib._approx_oracle"),
          c_str!("_extlib._approx_oracle"),
      )?.unbind())
    }).unwrap();
    Extlib{
      approx_oracle,
    }
  }

  pub fn _approx_oracle_worker_cls(&self) -> PyObject {
    Python::with_gil(|py| -> PyResult<Py<_>> {
      self.approx_oracle.getattr(py, "ApproxOracleWorker")
    }).unwrap()
  }

  pub fn _approx_oracle_worker(&self, concurrency: u32) -> PyObject {
    Python::with_gil(|py| -> PyResult<_> {
      let cls = self.approx_oracle.getattr(py, "ApproxOracleWorker")?;
      cls.call1(py, (concurrency,))
    }).unwrap()
  }

  pub fn _approx_oracle_interface_cls(&self) -> PyObject {
    Python::with_gil(|py| -> PyResult<Py<_>> {
      self.approx_oracle.getattr(py, "ApproxOracleInterface")
    }).unwrap()
  }

  pub fn _approx_oracle_interface(&self) -> PyObject {
    Python::with_gil(|py| -> PyResult<_> {
      let cls = self.approx_oracle.getattr(py, "ApproxOracleInterface")?;
      cls.call0(py)
    }).unwrap()
  }

  pub fn _approx_oracle_interface_with_worker(&self, worker_obj: PyObject) -> PyObject {
    Python::with_gil(|py| -> PyResult<_> {
      let cls = self.approx_oracle.getattr(py, "ApproxOracleInterface")?;
      cls.call1(py, (worker_obj,))
    }).unwrap()
  }
}
