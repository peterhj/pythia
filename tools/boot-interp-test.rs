extern crate pythia;
//extern crate rayon;
extern crate term_colors;

use pythia::clock::{Timedelta, Timestamp};
use pythia::interp::*;
use pythia::smp::{init_smp};
use pythia::test_data::*;
//use rayon::prelude::*;
use term_colors::{Colorize};

enum TestResult {
  OK(Timedelta, Timedelta, Yield_),
  Check(Timedelta, Timedelta, InterpCheck),
}

fn main() {
  init_smp();
  let test_data_cfg = TestDataConfig::default();
  println!("DEBUG: boot: test data config = {:?}", test_data_cfg);
  //let mut results = Vec::new();
  let results: Vec<_> = test_data_cfg.iter_interp_tests().enumerate()
  .map(|(idx, item)| {
    println!("DEBUG: boot: test[{idx}]: key = {:?}", item.key);
    let t0 = Timestamp::fresh();
    let mut interp = FastInterp::default();
    interp.set_debug();
    if idx == 5
       || item.key.contains("interp_ident_")
       || item.key.contains("interp_eq_")
       || item.key.contains("interp_fun_")
       //|| item.key.contains("interp_obj_")
       || item.key.contains("interp_choice_")
    {
      interp.set_trace();
      interp.set_parser_debug();
    }
    match interp.pre_init() {
      Err(check) => {
        let t1 = Timestamp::fresh();
        return (item.key, item.src, TestResult::Check(t1-t0, Timedelta::default(), check));
      }
      Ok(_) => {}
    }
    match interp.load_(&item.src) {
      Err(check) => {
        let t1 = Timestamp::fresh();
        return (item.key, item.src, TestResult::Check(t1-t0, Timedelta::default(), check));
      }
      Ok(_) => {}
    }
    let t1 = Timestamp::fresh();
    let yield_ = match interp.interp_() {
      Err(check) => {
        let t2 = Timestamp::fresh();
        return (item.key, item.src, TestResult::Check(t1-t0, t2-t1, check));
      }
      Ok(yield_) => yield_
    };
    let t2 = Timestamp::fresh();
    let flatinterp = interp.flatten_();
    println!("DEBUG: boot: test[{idx}]: flat = {:?}", flatinterp);
    test_data_cfg.set_vector_file(&item.key, &flatinterp.vectorize());
    return (item.key, item.src, TestResult::OK(t1-t0, t2-t1, yield_));
  }).collect();
  //.collect_into_vec(&mut results);
  let mut ok_ct = 0;
  let mut check_ct = 0;
  for (idx, (key, src, res)) in results.into_iter().enumerate() {
    println!("DEBUG: boot: test[{idx}]: key = {:?}", key);
    println!("DEBUG: boot: test[{idx}]: src = {:?}", src);
    match res {
      TestResult::OK(dt0, dt1, yield_) => {
        println!("DEBUG: boot: test[{idx}]: dt0 = {} s", dt0);
        println!("DEBUG: boot: test[{idx}]: dt1 = {} s", dt1);
        if yield_ == Yield_::Quiescent {
          println!("DEBUG: boot: test[{idx}]: {}", "ok".green().bold());
        } else {
          println!("DEBUG: boot: test[{idx}]: {} = {:?}", "ok".green().bold(), yield_);
        }
        ok_ct += 1;
      }
      TestResult::Check(dt0, dt1, check) => {
        println!("DEBUG: boot: test[{idx}]: dt0 = {} s", dt0);
        println!("DEBUG: boot: test[{idx}]: dt1 = {} s", dt1);
        println!("DEBUG: boot: test[{idx}]: {} = {:?}", "check".red().bold(), check);
        check_ct += 1;
      }
    }
  }
  if check_ct > 0 {
    println!("DEBUG: boot: {} {} / {} {} / {} total",
        check_ct, "check".red().bold(),
        ok_ct, "ok",
        check_ct + ok_ct
    );
  } else {
    println!("DEBUG: boot: {} {} / {} {} / {} total",
        check_ct, "check",
        ok_ct, "ok".green().bold(),
        check_ct + ok_ct
    );
  }
}
