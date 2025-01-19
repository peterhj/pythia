use crate::algo::{BTreeSet};

use std::io::{Cursor};
use std::process::{Command, Stdio};
use std::str::{from_utf8};

#[derive(Clone, Copy, Default, Debug)]
pub struct LscpuEntry {
  pub cpu:  u16,
  pub core: u16,
  pub sock: u8,
  pub node: u8,
}

#[derive(Clone, Debug)]
pub struct LscpuParse {
  pub entries: Vec<LscpuEntry>,
  pub core_ct: u16,
}

#[derive(Clone, Copy, Debug)]
enum LscpuParseState {
  Field(u8),
  Skip,
}

impl LscpuParse {
  pub fn open() -> Result<LscpuParse, ()> {
    let out = Command::new("lscpu")
        .arg("-p")
        .stdout(Stdio::piped())
        .output()
        .map_err(|_| ())?;
    if !out.status.success() {
      return Err(());
    }
    LscpuParse::parse(out.stdout)
  }

  pub fn parse<O: AsRef<[u8]>>(out: O) -> Result<LscpuParse, ()> {
    let out = out.as_ref();
    let mut entries = Vec::new();
    let mut e = LscpuEntry::default();
    let mut save = 0;
    let mut cursor = 0;
    let mut state = LscpuParseState::Field(0);
    loop {
      match state {
        LscpuParseState::Field(col) => {
          if cursor >= out.len() {
            if save == cursor && col == 0 {
              break;
            } else {
              return Err(());
            }
          }
          let x = out[cursor];
          if x == b'#' && save == cursor && col == 0 {
            cursor += 1;
            save = usize::max_value();
            state = LscpuParseState::Skip;
          } else if x >= b'0' && x <= b'9' {
            cursor += 1;
          } else if x == b',' {
            let s = from_utf8(&out[save .. cursor]).map_err(|_| ())?;
            match col {
              0 => e.cpu  = s.parse().map_err(|_| ())?,
              1 => e.core = s.parse().map_err(|_| ())?,
              2 => e.sock = s.parse().map_err(|_| ())?,
              3 => e.node = s.parse().map_err(|_| ())?,
              _ => unreachable!()
            }
            cursor += 1;
            if col < 3 {
              save = cursor;
              state = LscpuParseState::Field(col + 1);
            } else {
              entries.push(e);
              e = LscpuEntry::default();
              save = usize::max_value();
              state = LscpuParseState::Skip;
            }
          } else {
            return Err(());
          }
        }
        LscpuParseState::Skip => {
          if cursor >= out.len() {
            break;
          }
          let x = out[cursor];
          cursor += 1;
          if x == b'\n' {
            save = cursor;
            state = LscpuParseState::Field(0);
          }
        }
      }
    }
    let mut info = LscpuParse{
      entries,
      core_ct: 0,
    };
    if let Some(ct) = info._physical_core_count() {
      info.core_ct = ct;
    }
    Ok(info)
  }

  pub fn _physical_core_count(&self) -> Option<u16> {
    let mut core = BTreeSet::new();
    for e in self.entries.iter() {
      core.insert(e.core);
    }
    if core.len() == 0 {
      return None;
    }
    assert!(core.len() <= u16::max_value() as _);
    core.len().try_into().ok()
  }

  pub fn physical_core_count(&self) -> Option<u16> {
    if self.core_ct == 0 {
      None
    } else {
      Some(self.core_ct)
    }
  }
}
