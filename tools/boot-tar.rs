extern crate pythia;

use pythia::algo::*;
use pythia::util::encoding::*;
use pythia::util::gzip::*;
use pythia::util::tar::*;

use std::cmp::{min};
use std::fs::{File};
use std::io::{Cursor, copy as iocopy};
use std::path::{PathBuf};

fn main() {
  let argv: Vec<_> = std::env::args().collect();
  let path = PathBuf::from(&argv[1]);
  let file = File::open(path).unwrap();
  let mut tar = TarFile::new(file);
  let mut it = tar.entry_iter();
  let mut utfs = HashMap::new();
  let mut non_utfs = HashMap::new();
  let mut encs = HashMap::new();
  let mut non_encs = HashMap::new();
  let mut h = BTreeMap::new();
  let mut n = 0;
  let mut n_tar = 0;
  let mut n_tar_shape = 0;
  let mut n_tar_chk = 0;
  let mut n_non_tar = 0;
  for (i, e) in it.enumerate() {
    if i == 0 {
    } else {
      n += 1;
    }
    if i == 1 {
      //println!("{:?}", e);
    }
    let e = e.unwrap();
    let p = e.path.as_os_str().to_str().unwrap();
    if p.ends_with(".gz") {
      // FIXME: only applies to older ids.
      let cat = p.get(5 .. p.len() - 10).unwrap();
      match h.get_mut(cat) {
        None => {
          h.insert(cat.to_string(), 1);
        }
        Some(ct) => {
          *ct += 1;
        }
      }
      match cat {
        _ => {
          let e_file = tar.slice_subreader_at(e.entry_pos, e.entry_sz);
          let mut e_file = GzipReader::new(e_file);
          let mut e_buf = Vec::new();
          iocopy(&mut e_file, &mut e_buf).unwrap();
          let mut tar = false;
          let mut tar_chk = false;
          let mut nonascii = false;
          /*for &x in &e_buf[ .. min(512, e_buf.len())] {
            if x == 0 {
              tar = true;
              break;
            }
          }*/
          if e_buf.len() >= 512 && e_buf.len() % 512 == 0 {
            n_tar_shape += 1;
            match check_tar_header_block(&e_buf[ .. 512]) {
              Ok(true) => {
                tar = true;
              }
              Ok(false) => {
                tar = true;
                tar_chk = true;
              }
              _ => {}
            }
          }
          if !tar {
            n_non_tar += 1;
            if cat == "cs" {
              println!("{:?}", e);
            }
            // TODO: chardet.
            let mut enc = EncodingDetector::new();
            nonascii = enc.feed(&e_buf, true);
            let (utf8_det, utf8_assess) = enc.guess_assess(None, true);
            let (non_utf8_det, non_utf8_assess) = enc.guess_assess(None, false);
            if utf8_assess {
              match utfs.get_mut(utf8_det) {
                None => {
                  utfs.insert(utf8_det, 1);
                }
                Some(ct) => {
                  *ct += 1;
                }
              }
            } else {
              match non_utfs.get_mut(utf8_det) {
                None => {
                  non_utfs.insert(utf8_det, 1);
                }
                Some(ct) => {
                  *ct += 1;
                }
              }
            }
            if non_utf8_assess {
              match encs.get_mut(non_utf8_det) {
                None => {
                  encs.insert(non_utf8_det, 1);
                }
                Some(ct) => {
                  *ct += 1;
                }
              }
            } else {
              match non_encs.get_mut(non_utf8_det) {
                None => {
                  non_encs.insert(non_utf8_det, 1);
                }
                Some(ct) => {
                  *ct += 1;
                }
              }
            }
          } else {
            n_tar += 1;
            if tar_chk {
              n_tar_chk += 1;
            }
            let e_buf = Cursor::new(e_buf);
            let e_file = TarFile::new(e_buf);
            for ee in e_file.entry_iter() {
              //println!("{:?}", ee);
              let ee = ee.unwrap();
              let ep = ee.path.as_os_str().to_str().unwrap();
              if ep.ends_with(".tex") {
                if cat == "cs" {
                  println!("{:?}", ee);
                }
              }
            }
          }
        }
      }
    }
  }
  println!("DEBUG: n = {}", n);
  println!("DEBUG: n tar = {}", n_tar);
  println!("DEBUG: n tar shape = {}", n_tar_shape);
  println!("DEBUG: n tar chk = {}", n_tar_chk);
  println!("DEBUG: n non tar = {}", n_non_tar);
  println!("{:?}", utfs);
  println!("{:?}", non_utfs);
  println!("{:?}", encs);
  println!("{:?}", non_encs);
  println!("{:?}", h);
}
