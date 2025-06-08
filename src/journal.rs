use crate::algo::base64::{Base64Format};
use crate::algo::blake2s::{Blake2s};
//use crate::algo::hex::{HexFormat};
use crate::algo::json::{JsonValue};
use crate::clock::{Timestamp};
use crate::oracle::{ApproxOracleItem};

use byteorder::{LittleEndian as LE, ReadBytesExt, WriteBytesExt};
//use once_cell::sync::{Lazy};
use serde::{Serialize, Deserialize};
use serde_json_fmt::{JsonFormat};

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write, Seek, SeekFrom};
use std::net::{TcpStream};
use std::path::{PathBuf};
use std::str::{FromStr};

pub const HASH_SIZE: usize = 32;

//pub static _STORE: Lazy<DevelJournal_> = Lazy::new(|| DevelJournal_::cold_start());

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[non_exhaustive]
pub enum JournalEntrySort_ {
  #[serde(rename = "root")]
  _Root,
  #[serde(rename = "aikido")]
  Aikido,
  #[serde(rename = "approx-oracle")]
  ApproxOracle,
  #[serde(rename = "approx-oracle-test")]
  ApproxOracleTest,
  #[serde(rename = "boot-test")]
  BootTest,
  #[serde(rename = "test")]
  Test,
}

impl FromStr for JournalEntrySort_ {
  type Err = ();

  fn from_str(s: &str) -> Result<JournalEntrySort_, ()> {
    Ok(match s {
      "root" |
      "\"root\"" => {
        JournalEntrySort_::_Root
      }
      "aikido" |
      "\"aikido\"" => {
        JournalEntrySort_::Aikido
      }
      "approx-oracle" |
      "\"approx-oracle\"" => {
        JournalEntrySort_::ApproxOracle
      }
      "approx-oracle-test" |
      "\"approx-oracle-test\"" => {
        JournalEntrySort_::ApproxOracleTest
      }
      "boot-test" |
      "\"boot-test\"" => {
        JournalEntrySort_::BootTest
      }
      "test" |
      "\"test\"" => {
        JournalEntrySort_::Test
      }
      _ => return Err(())
    })
  }
}

pub struct RootSort_ {
}

impl RootSort_ {
  pub fn item_from_value(v: JsonValue) -> () {
    // FIXME
    //serde_json::from_value(_)
  }
}

pub struct AikidoSort_ {
}

impl AikidoSort_ {
  pub fn item_from_value(v: JsonValue) -> () {
    // FIXME
    //serde_json::from_value(_)
  }
}

pub struct ApproxOracleSort_ {
}

impl ApproxOracleSort_ {
  pub fn item_from_value(v: JsonValue) -> Result<ApproxOracleItem, String> {
    serde_json::from_value::<ApproxOracleItem>(v).map_err(|e| format!("{:?}", e))
  }
}

pub struct TestSort_ {
}

impl TestSort_ {
  pub fn item_from_value(v: JsonValue) -> Result<Test, String> {
    // FIXME
    let t = serde_json::from_value::<Test>(v).map_err(|e| format!("{:?}", e))?;
    Ok(t)
  }
}

pub trait JournalEntryExt {
  fn _sort(&self) -> JournalEntrySort_;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct _Root;

impl JournalEntryExt for _Root {
  fn _sort(&self) -> JournalEntrySort_ {
    JournalEntrySort_::_Root
  }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BootTest;

impl JournalEntryExt for BootTest {
  fn _sort(&self) -> JournalEntrySort_ {
    JournalEntrySort_::BootTest
  }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Test {
  pub hello: String,
}

impl JournalEntryExt for Test {
  fn _sort(&self) -> JournalEntrySort_ {
    JournalEntrySort_::Test
  }
}

#[derive(Clone, Debug)]
pub struct JournalEntryResult {
  pub eid:  i64,
  pub t:    Timestamp,
}

#[derive(Serialize, Debug)]
//#[derive(Serialize, Deserialize, Debug)]
pub struct JournalEntry_<I> {
  pub eid:  i64,
  pub t:    Timestamp,
  pub sort: JournalEntrySort_,
  pub item: I,
  pub hash: String,
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct JournalEntryNum {
  _idx: usize,
}

pub struct JournalBackend {
  widx_mem: Vec<u32>,
  //wlog_mem: Vec<()>,
  widx_file: File,
  wlog_file: File,
}

impl JournalBackend {
  pub fn cold_start() -> JournalBackend {
    let t0 = Timestamp::fresh();
    let root_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/_journal"
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
    let _widx_size = match std::fs::metadata(&widx_path) {
      Err(_) => 0,
      Ok(f) => f.len() as usize
    };
    let _wlog_size = match std::fs::metadata(&wlog_path) {
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
            let _data = match wlog_lines.next() {
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
    println!("DEBUG: JournalBackend::cold_start: widx len = {}", widx_mem.len());
    println!("DEBUG: JournalBackend::cold_start: truncate = {:?}", truncate);
    // FIXME: data "safe" truncate.
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
    let mut this = JournalBackend{
      widx_mem,
      widx_file,
      wlog_file,
    };
    if this.widx_mem.len() <= 0 {
      this.append_item(&_Root);
    }
    this
  }

  pub fn _append<S: AsRef<str>>(&mut self, s: S) -> JournalEntryNum {
    let s = s.as_ref();
    let widx = self.widx_mem.len();
    let wpos = match widx {
      0 => 0,
      i => self.widx_mem[i-1],
    };
    let woff = wpos + (s.len() as u32) + 1;
    self.widx_mem.push(woff);
    self.widx_file.write_u32::<LE>(woff).unwrap();
    writeln!(&mut self.wlog_file, "{}", s).unwrap();
    JournalEntryNum{_idx: widx}
  }

  pub fn append_item<I: JournalEntryExt + Serialize>(&mut self, item: &I) -> JournalEntryResult {
    let t = Timestamp::fresh();
    let widx = self.widx_mem.len();
    let eid: i64 = widx.try_into().unwrap();
    let sort = item._sort();
    let mut hval = Vec::with_capacity(HASH_SIZE);
    hval.resize(HASH_SIZE, 0);
    let hash = Base64Format::default().to_string(&hval);
    let eres = JournalEntryResult{
      eid,
      t,
    };
    let entry = JournalEntry_{
      eid,
      t,
      sort,
      item,
      hash,
    };
    let json_fmt = JsonFormat::new()
        .ascii(true)
        .colon(": ").unwrap()
        .comma(", ").unwrap();
    let mut s = json_fmt.to_string(&entry).unwrap();
    println!("DEBUG: JournalBackend::append_item: s = {:?}", s);
    let slen = s.len();
    //let hend = slen - (11 + HASH_SIZE * 2 + 2);
    let hend = slen - (12 + 43 + 2);
    println!("DEBUG: end = {:?}", s.get(hend .. ).unwrap());
    println!("DEBUG: len = {}", s.get(hend .. ).unwrap().len());
    println!("DEBUG: hlen = {}", hend);
    let mut h = Blake2s::new_hash();
    {
      let b = s.as_bytes();
      assert!(b.starts_with(b"{"));
      assert!(b[hend .. ].starts_with(b", \"hash\": \""));
      assert!(b.ends_with(b"\"}"));
      h.hash_bytes(&b[1 .. hend]);
    }
    unsafe {
      let mut s = s.as_mut_str();
      let mut b = s.as_bytes_mut();
      let hash = Base64Format::default().to_string(&h.finalize());
      (&mut b[hend + 11 .. slen - 2]).copy_from_slice(hash.as_bytes());
    }
    self._append(s);
    eres
  }
}

pub struct DevelJournal_ {
  backend: JournalBackend,
}

impl DevelJournal_ {
  pub fn cold_start() -> DevelJournal_ {
    let backend = JournalBackend::cold_start();
    let mut this = DevelJournal_{backend};
    if this.backend.widx_mem.len() <= 0 {
      this.append(&_Root);
    }
    this
  }
}

pub trait JournalExt {
  fn append<I: JournalEntryExt + Serialize>(&mut self, item: &I) -> JournalEntryResult where Self: Sized;
}

impl JournalExt for DevelJournal_ {
  fn append<I: JournalEntryExt + Serialize>(&mut self, item: &I) -> JournalEntryResult {
    let t = Timestamp::fresh();
    let widx = self.backend.widx_mem.len();
    let eid: i64 = widx.try_into().unwrap();
    let sort = item._sort();
    let mut hval = Vec::with_capacity(HASH_SIZE);
    hval.resize(HASH_SIZE, 0);
    let hash = Base64Format::default().to_string(&hval);
    let eres = JournalEntryResult{
      eid,
      t,
    };
    let entry = JournalEntry_{
      eid,
      t,
      sort,
      item,
      hash,
    };
    let json_fmt = JsonFormat::new()
        .ascii(true)
        .colon(": ").unwrap()
        .comma(", ").unwrap();
    let mut s = json_fmt.to_string(&entry).unwrap();
    let slen = s.len();
    let hend = slen - (11 + HASH_SIZE * 2 + 2);
    let mut h = Blake2s::new_hash();
    {
      let b = s.as_bytes();
      assert!(b.starts_with(b"{"));
      assert!(b[hend .. ].starts_with(b", \"hash\": \""));
      assert!(b.ends_with(b"\"}"));
      h.hash_bytes(&b[1 .. hend]);
    }
    unsafe {
      let mut s = s.as_mut_str();
      let mut b = s.as_bytes_mut();
      let hash = Base64Format::default().to_string(&h.finalize());
      (&mut b[hend + 11 .. slen - 2]).copy_from_slice(hash.as_bytes());
    }
    self.backend._append(s);
    eres
  }
}

/*pub struct JournalInterface {
  stream: TcpStream,
}

impl JournalInterface {
  pub fn new() -> JournalInterface {
    JournalInterface{
      stream: TcpStream::connect("127.0.0.1:9001").unwrap(),
    }
  }

  pub fn hi(&mut self) {
    self.stream.write_all(b"hi \n").unwrap();
    let mut buf = [0u8; 4];
    let nread = self.stream.read(&mut buf).unwrap();
    match &buf[ .. nread] {
      b"ok \n" => {}
      b"err\n" => {}
      _ => {
        // TODO
      }
    }
  }

  pub fn put(&mut self) {
  }

  pub fn get(&mut self) {
  }
}*/
