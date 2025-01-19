extern crate pythia;
extern crate term_colors;

use pythia::clock::{Timedelta, Timestamp};
use pythia::interp::*;
use term_colors::{Colorize};

use std::env::{args};
use std::fs::{File};
use std::io::{Read};

enum TestResult {
  OK(Timedelta, Timedelta, Yield_),
  Check(Timedelta, Timedelta, InterpCheck),
}

fn _interp(src: &str) -> TestResult {
  let t0 = Timestamp::fresh();
  let mut interp = FastInterp::default();
  match interp.pre_init() {
    Err(check) => {
      let t1 = Timestamp::fresh();
      return TestResult::Check(t1-t0, Timedelta::default(), check);
    }
    Ok(_) => {}
  }
  match interp.load_(src) {
    Err(check) => {
      let t1 = Timestamp::fresh();
      return TestResult::Check(t1-t0, Timedelta::default(), check);
    }
    Ok(_) => {}
  }
  let t1 = Timestamp::fresh();
  let yield_ = match interp.interp_() {
    Err(check) => {
      let t2 = Timestamp::fresh();
      return TestResult::Check(t1-t0, t2-t1, check);
    }
    Ok(yield_) => yield_
  };
  let t2 = Timestamp::fresh();
  return TestResult::OK(t1-t0, t2-t1, yield_);
}

fn main() {
  let argv: Vec<_> = args().collect();
  let mut args_err = false;
  let mut v = false;
  let mut src_path = None;
  for arg in (&argv[1 .. ]).iter() {
    if arg.starts_with("-") {
      if arg == "-v" {
        v = true;
      } else {
        args_err = true;
        break;
      }
    } else {
      src_path = Some(arg.to_string());
    }
  }
  if args_err || src_path.is_none() {
    println!("usage: interp [-v] <source.pythia>");
    return;
  }
  let mut file = File::open(src_path.as_ref().unwrap()).unwrap();
  let mut src = String::new();
  file.read_to_string(&mut src).unwrap();
  drop(file);
  let res = _interp(&src);
  if v {
    match res {
      TestResult::OK(dt0, dt1, yield_) => {
        println!("DEBUG: interp: dt0 = {} s", dt0);
        println!("DEBUG: interp: dt1 = {} s", dt1);
        if yield_ == Yield_::Quiescent {
          println!("DEBUG: interp: {}", "ok".green().bold());
        } else {
          println!("DEBUG: interp: {} = {:?}", "ok".green().bold(), yield_);
        }
      }
      TestResult::Check(dt0, dt1, check) => {
        println!("DEBUG: interp: dt0 = {} s", dt0);
        println!("DEBUG: interp: dt1 = {} s", dt1);
        println!("DEBUG: interp: {} = {:?}", "check".red().bold(), check);
      }
    }
  }
}
