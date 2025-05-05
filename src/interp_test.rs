// TODO: temporarily disabled lint for debugging.
#![allow(unused_variables)]

use crate::clock::{Timedelta, Timestamp};
use crate::interp::*;
use crate::tap::*;
use crate::test_data::*;

use term_colors::{Colorize};

use std::io::{Read, Write, Error as IoError};

pub struct InterpTestItem {
  pub key:  String,
  pub src:  String,
  pub vdst: Option<String>,
  pub snapshot_dst: Option<String>,
}

pub enum TestResult {
  OK(Timedelta, Timedelta, Yield_),
  Check(Timedelta, Timedelta, InterpCheck),
}

#[derive(Default)]
pub struct InterpTestsProver {
  conf: TestDataConfig,
}

impl From<TestDataConfig> for InterpTestsProver {
  fn from(conf: TestDataConfig) -> InterpTestsProver {
    InterpTestsProver{conf}
  }
}

impl InterpTestsProver {
  pub fn _prove_item<W: Write + ?Sized>(&self, rank: usize, item: &InterpTestItem, writer: &mut W) -> Result<(), IoError> {
    let mut interp = FastInterp::default();
    let snapshot = self.conf.init_snapshot_file(&item.key);
    let _ = interp.set_snapshot_writer(snapshot);
    let tap_writer = wrap_tap_writer(Vec::<u8>::new());
    let _ = interp.set_tap_writer(tap_writer);
    // TODO: debugging.
    if item.key.contains("interp_choice_if_")
    {
      interp.set_trace();
    }
    match interp.pre_init() {
      Err(check) => {
        let t1 = Timestamp::fresh();
        writeln!(writer, "{} {} - {:?}", "not ok".red().bold(), rank, &item.key)?;
        writeln!(writer, "# check = {:?}", check)?;
        return Ok(());
      }
      Ok(_) => {}
    }
    match interp.cold_start(&item.src) {
      Err(check) => {
        let t1 = Timestamp::fresh();
        writeln!(writer, "{} {} - {:?}", "not ok".red().bold(), rank, &item.key)?;
        writeln!(writer, "# check = {:?}", check)?;
        return Ok(());
      }
      Ok(_) => {}
    }
    let t1 = Timestamp::fresh();
    let yield_ = match interp.interp_() {
      Err(check) => {
        let t2 = Timestamp::fresh();
        writeln!(writer, "{} {} - {:?}", "not ok".red().bold(), rank, &item.key)?;
        writeln!(writer, "# check = {:?}", check)?;
        return Ok(());
      }
      Ok(yield_) => yield_
    };
    let t2 = Timestamp::fresh();
    /*let flatinterp = interp.flatten_();
    self.conf.set_vector_file(&item.key, &flatinterp.vectorize());*/
    writeln!(writer, "{} {} - {:?}", "ok".green(), rank, &item.key)?;
    if yield_ == Yield_::Quiescent {
    } else {
      println!("# yield = {:?}", yield_);
    }
    // FIXME: cannot "unwrap" Box<dyn Write> to get lines again.
    let tap_writer = interp.unset_tap_writer();
    Ok(())
  }
}

impl TAPProver for InterpTestsProver {
  fn prove<W: Write + ?Sized>(&self, writer: &mut W) -> Result<(), IoError> {
    let mut ctr = 0;
    for (idx, key) in self.conf.keys().iter().enumerate() {
      let mut f = self.conf.get_source_file(key);
      let mut src = String::new();
      f.read_to_string(&mut src).unwrap();
      let vdst = if let Some(mut f) = self.conf.maybe_get_vector_file(key) {
        let mut vdst = String::new();
        f.read_to_string(&mut vdst).unwrap();
        Some(vdst)
      } else {
        None
      };
      let item = InterpTestItem{
        key: key.to_string(),
        src,
        vdst,
        // FIXME
        snapshot_dst: None,
      };
      self._prove_item(idx + 1, &item, writer)?;
      ctr += 1;
    }
    writeln!(writer, "1..={}", ctr)?;
    Ok(())
  }
}
