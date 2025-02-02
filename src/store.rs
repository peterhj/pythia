use crate::clock::{Timestamp};

use byteorder::{LittleEndian as LE, ReadBytesExt, WriteBytesExt};
use once_cell::sync::{Lazy};
use serde::{Serialize, Deserialize};

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write, Seek, SeekFrom};
use std::path::{PathBuf};

pub static _STORE: Lazy<DevelStore_> = Lazy::new(|| DevelStore_::cold_start());

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum StoreSort_ {
  #[serde(rename = "approx-oracle")]
  ApproxOracle,
  #[serde(rename = "boot-test")]
  BootTest,
}

pub trait StoreSortExt {
  fn _sort(&self) -> StoreSort_;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BootTest;

impl StoreSortExt for BootTest {
  fn _sort(&self) -> StoreSort_ {
    StoreSort_::BootTest
  }
}

#[derive(Serialize, Debug)]
//#[derive(Serialize, Deserialize, Debug)]
pub struct StoreLogEntry_<I> {
  pub t:    Timestamp,
  pub sort: StoreSort_,
  pub item: I,
}

pub struct DevelStore_ {
  widx_mem: Vec<u32>,
  //wlog_mem: Vec<()>,
  widx_file: File,
  wlog_file: File,
}

impl DevelStore_ {
  pub fn cold_start() -> DevelStore_ {
    let t0 = Timestamp::fresh();
    let root_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/_store"
    ));
    let mut widx_path = root_path.clone();
    widx_path.push("_widx.bin");
    let mut wlog_path = root_path.clone();
    wlog_path.push("_wlog.jsonl");
    /*let mut wmeta_path = root_path.clone();
    wmeta_path.push("_wmeta.bin")
    let wmeta_file = File::open(&wmeta_path).unwrap();
    let mut wmeta_reader = BufReader::new(wmeta_file);
    let header = match wmeta_reader.read_u64::<LE>() {
      Err(_) => {
        // NB: no/malformed header, rebuild.
        truncate = true;
        break;
      }
      Ok(v) => v
    };*/
    let widx_size = match std::fs::metadata(&widx_path) {
      Err(_) => 0,
      Ok(f) => f.len() as usize
    };
    let wlog_size = match std::fs::metadata(&wlog_path) {
      Err(_) => 0,
      Ok(f) => f.len() as usize
    };
    let mut widx_mem = Vec::new();
    let mut truncate = false;
    loop {
      match (File::open(&widx_path), File::open(&wlog_path)) {
        (Ok(widx_file), Ok(wlog_file)) => {
          let mut widx_reader = BufReader::new(widx_file);
          let mut wlog_lines = BufReader::new(wlog_file).lines();
          loop {
            let mut buf: [u8; 4] = [0; 4];
            match widx_reader.read_u8() {
              Err(_) => {
                break;
              }
              Ok(v) => {
                buf[0] = v;
              }
            }
            let mut err = false;
            for p in 1 .. 4 {
              match widx_reader.read_u8() {
                Err(_) => {
                  err = true;
                  break;
                }
                Ok(v) => {
                  buf[p] = v;
                }
              }
            }
            if err {
              truncate = true;
              break;
            }
            let woff = u32::from_le_bytes(buf);
            let wpos = if widx_mem.len() <= 0 {
              0
            } else {
              widx_mem[widx_mem.len()-1]
            };
            let data = match wlog_lines.next() {
              None |
              Some(Err(_)) => {
                truncate = true;
                break;
              }
              Some(Ok(s)) => {
                if wpos + (s.len() as u32) + 1 != woff {
                  truncate = true;
                  break;
                } else {
                  s
                }
              }
            };
            widx_mem.push(woff);
          }
        }
        _ => {
        }
      }
      break;
    }
    println!("DEBUG: DevelStore_::_cold_start: widx len = {}", widx_mem.len());
    println!("DEBUG: DevelStore_::_cold_start: truncate = {:?}", truncate);
    if truncate {
      let t0_s = t0.to_string().replace(":", "_");
      let mut wlog_dst_path = root_path.clone();
      wlog_dst_path.push(format!("_wlog-{}.jsonl", t0_s));
      match (
          File::open(&wlog_path),
          OpenOptions::new()
          .write(true).create(true)
          .open(&wlog_dst_path)
      ) {
        (Ok(mut src_file), Ok(mut dst_file)) => {
          let mut err = false;
          if widx_mem.len() > 0 {
            match src_file.seek(SeekFrom::Start(widx_mem[widx_mem.len()-1] as u64)) {
              Err(_) => {
                err = true;
              }
              Ok(_) => {}
            }
          }
          if !err {
            let _ = std::io::copy(&mut src_file, &mut dst_file);
          }
        }
        _ => {}
      }
      let _widx_file = OpenOptions::new()
          .write(true).create(true)
          .open(&widx_path).unwrap();
      let _wlog_file = OpenOptions::new()
          .write(true).create(true)
          .open(&wlog_path).unwrap();
    }
    let widx_file = OpenOptions::new()
        .append(true).create(true)
        .open(&widx_path).unwrap();
    let wlog_file = OpenOptions::new()
        .append(true).create(true)
        .open(&wlog_path).unwrap();
    DevelStore_{
      widx_mem,
      widx_file,
      wlog_file,
    }
  }

  pub fn append<I: StoreSortExt + Serialize>(&mut self, item: &I) -> () {
    let t = Timestamp::fresh();
    let sort = item._sort();
    let entry = StoreLogEntry_{
      t,
      sort,
      item,
    };
    let s = serde_json::to_string(&entry).unwrap();
    let wpos = match self.widx_mem.len() {
      0 => 0,
      i => self.widx_mem[i-1],
    };
    let woff = wpos + (s.len() as u32) + 1;
    self.widx_mem.push(woff);
    self.widx_file.write_u32::<LE>(woff).unwrap();
    writeln!(&mut self.wlog_file, "{}", &s).unwrap();
  }
}
