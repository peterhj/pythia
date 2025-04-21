use crate::build::{CWD};

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{PathBuf};

const INTERP_TESTS: &'static str = include_str!("../data/test/interp.txt");

pub fn test_data_root() -> PathBuf {
  let mut p = PathBuf::from(CWD);
  p.push("data/test");
  p
}

#[derive(Debug)]
pub struct TestDataConfig {
  pub root: PathBuf,
  pub keys: Vec<String>,
}

impl Default for TestDataConfig {
  fn default() -> TestDataConfig {
    TestDataConfig::interp_tests()
  }
}

impl TestDataConfig {
  pub fn interp_tests() -> TestDataConfig {
    let mut keys = Vec::new();
    for line in INTERP_TESTS.lines() {
      keys.push(line.into());
    }
    TestDataConfig{
      root: test_data_root(),
      keys,
    }
  }

  pub fn test_1() -> TestDataConfig {
    TestDataConfig::interp_test_1()
  }

  pub fn interp_test_1() -> TestDataConfig {
    let mut keys = Vec::new();
    for line in INTERP_TESTS.lines() {
      keys.push(line.into());
      break;
    }
    TestDataConfig{
      root: test_data_root(),
      keys,
    }
  }

  pub fn test_last() -> TestDataConfig {
    TestDataConfig::interp_test_last()
  }

  pub fn interp_test_last() -> TestDataConfig {
    let mut keys = Vec::new();
    for line in INTERP_TESTS.lines().rev() {
      keys.push(line.into());
      break;
    }
    TestDataConfig{
      root: test_data_root(),
      keys,
    }
  }

  pub fn keys(&self) -> &[String] {
    &self.keys
  }

  pub fn get_source(&self, key: &str) -> String {
    let mut f = self.get_source_file(key);
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    s
  }

  pub fn get_source_file(&self, key: &str) -> File {
    let mut p = self.root.clone();
    p.push(key);
    File::open(&p).unwrap()
  }

  pub fn init_snapshot_file(&self, key: &str) -> Box<File> {
    let mut p = self.root.clone();
    p.push("_snapshot");
    p.push(key);
    p.set_extension("jsonl");
    Box::new(OpenOptions::new()
      .write(true).create(true).truncate(true)
      .open(&p).unwrap())
  }

  /*pub fn set_snapshot_file(&self, key: &str, val: &str) {
    let mut f = self.init_snapshot_file(key);
    f.write_all(val.as_bytes()).unwrap();
  }

  pub fn maybe_get_snapshot_file(&self, key: &str) -> Option<File> {
    let mut p = self.root.clone();
    p.push("_snapshot");
    p.push(key);
    p.set_extension("jsonl");
    File::open(&p).ok()
  }*/

  pub fn maybe_get_vector_file(&self, key: &str) -> Option<File> {
    let mut p = self.root.clone();
    p.push("__v");
    p.push(key);
    p.set_extension("json");
    File::open(&p).ok()
  }

  pub fn set_vector_file(&self, key: &str, val: &str) {
    let mut p = self.root.clone();
    p.push("__v");
    p.push(key);
    p.set_extension("json");
    let mut f = OpenOptions::new()
      .write(true).create(true).truncate(true)
      .open(&p).unwrap();
    f.write_all(val.as_bytes()).unwrap();
  }

  pub fn maybe_get_review_file(&self, key: &str) -> Option<File> {
    let mut p = self.root.clone();
    p.push("__rev");
    p.push(key);
    p.set_extension("json");
    File::open(&p).ok()
  }

  pub fn set_review_file(&self, key: &str, val: &str) {
    let mut p = self.root.clone();
    p.push("__rev");
    p.push(key);
    p.set_extension("json");
    let mut f = OpenOptions::new()
      .write(true).create(true).truncate(true)
      .open(&p).unwrap();
    f.write_all(val.as_bytes()).unwrap();
  }

  pub fn iter_interp_tests(&self) -> impl Iterator<Item=TestItem> + '_ {
    INTERP_TESTS.lines().map(|key| {
      let mut f = self.get_source_file(key);
      let mut src = String::new();
      f.read_to_string(&mut src).unwrap();
      let vdst = if let Some(mut f) = self.maybe_get_vector_file(key) {
        let mut vdst = String::new();
        f.read_to_string(&mut vdst).unwrap();
        Some(vdst)
      } else {
        None
      };
      let rev = if let Some(_) = self.maybe_get_review_file(key) {
        unimplemented!();
      } else {
        TestReview::default()
      };
      TestItem{
        key: key.to_string(),
        src,
        vdst,
        rev,
      }
    })
  }
}

pub struct TestItem {
  pub key:  String,
  pub src:  String,
  pub vdst: Option<String>,
  pub rev:  TestReview,
}

pub struct TestReview {
  // TODO
  pub rev_ct: i8,
}

impl Default for TestReview {
  fn default() -> TestReview {
    TestReview{
      rev_ct: 0
    }
  }
}

pub fn interp_tests() -> Vec<(String, String, PathBuf)> {
  interp_tests_iter().collect()
}

pub fn interp_tests_iter() -> impl Iterator<Item=(String, String, PathBuf)> {
  INTERP_TESTS.lines().map(|line| {
    let mut p = PathBuf::from("data/test");
    let mut pv = PathBuf::from("data/test/__v");
    p.push(line);
    pv.push(line);
    pv.set_extension("json");
    println!("DEBUG: test_data::interp_tests: src={} vdst={}", p.display(), pv.display());
    let mut f = File::open(&p).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    (line.into(), s, pv)
  })
}
