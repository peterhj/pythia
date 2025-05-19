//use crate::algo::{BTreeMap};
use crate::str::{len_utf8};

use serde::{Deserialize};

use std::collections::{BTreeMap};

#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum TokenCategory {
  // TODO
  //Stop,
  Space,
  Term,
  Constant,
  Placeholder,
  Predicate,
  Control,
  LDelimiter,
  RDelimiter,
  Delimiter,
  Particle,
}

#[derive(Default)]
pub struct TokenCharTrieMap<V> {
  next: BTreeMap<(u32, char), BTreeMap<char, u32>>,
  map:  BTreeMap<(u32, char), V>,
  ctr:  u32,
}

impl<V> TokenCharTrieMap<V> {
  fn _fresh(&mut self) -> u32 {
    let x = self.ctr + 1;
    self.ctr = x;
    x
  }

  /*pub fn insert<K: AsRef<str>>(&mut self, key: K, value: V) {
    let key = key.as_ref();
    assert!(key.len() > 0);
    let prev_x = 0;
    let prev_c = Err(());
    let cur_x = 0;
    let cur = key.chars().next().unwrap();
    self._insert(prev_x, prev_c, cur_x, cur, key.get(len_utf8(cur as _) .. ).unwrap(), value)
  }

  pub fn _insert<K: AsRef<str>>(&mut self, prev_x: u32, prev_c: Result<char, ()>, cur_x: u32, cur_c: char, rest: &str, value: V) {
    if let Ok(prev_c) = prev_c {
      match self.next.get_mut(&(prev_x, prev_c)) {
        None => {
          let mut next_set = BTreeMap::new();
          next_set.insert(cur_c, cur_x);
          self.next.insert((prev_x, prev_c), next_set);
        }
        Some(next_set) => {
          next_set.insert(cur_c, cur_x);
        }
      }
    }
    if rest.is_empty() {
      self.map.insert((cur_x, value));
    } else {
      let next = rest.get(len_utf8(cur_c as _) .. ).unwrap();
      let c = next.chars().next().unwrap();
      let x = self._fresh();
      self._insert(cur_x, Ok(cur_c), x, c, next, value)
    }
  }*/

  pub fn insert<K: AsRef<str>>(&mut self, key: K, value: V) {
    let key = key.as_ref();
    assert!(key.len() > 0);
    let mut prev_cur = 0;
    let mut prev_c: Result<char, ()> = Err(());
    let mut cur = 0;
    for (pos, c) in key.char_indices() {
      if pos > 0 {
        match self.next.get_mut(&(prev_cur, prev_c.unwrap())) {
          Some(next_set) => {
            match next_set.get(&c) {
              Some(&next_cur) => {
                prev_cur = cur;
                cur = next_cur;
              }
              None => {
                prev_cur = cur;
                /*cur = self._fresh();*/
                cur = self.ctr + 1;
                self.ctr = cur;
                next_set.insert(c, cur);
              }
            }
          }
          None => {
            let mut next_set = BTreeMap::new();
            prev_cur = cur;
            /*cur = self._fresh();*/
            cur = self.ctr + 1;
            self.ctr = cur;
            next_set.insert(c, cur);
            self.next.insert((prev_cur, prev_c.unwrap()), next_set);
          }
        }
      }
      prev_c = Ok(c);
    }
    self.map.insert((cur, prev_c.unwrap()), value);
  }
}

impl<V: Clone> TokenCharTrieMap<V> {
  pub fn prefix_match<P: AsRef<str>>(&self, pat: P) -> Vec<(usize, V)> {
    let mut mats = Vec::new();
    let pat = pat.as_ref();
    let mut prev_cur = 0;
    let mut prev_c: Result<char, ()> = Err(());
    let mut cur = 0;
    for (pos, c) in pat.char_indices() {
      if pos > 0 {
        match self.next.get(&(prev_cur, prev_c.unwrap())) {
          Some(next_set) => {
            match next_set.get(&c) {
              Some(&next_cur) => {
                prev_cur = cur;
                cur = next_cur;
              }
              None => break
            }
          }
          None => break
        }
      }
      if let Some(v) = self.map.get(&(cur, c)) {
        mats.push((pos + len_utf8(c as _), v.clone()));
      }
      prev_c = Ok(c);
    }
    mats
  }
}
