use crate::algo::cell::{RefCell};

use std::cmp::{min};
use std::ffi::{CStr};
use std::io::{Read, Seek, SeekFrom, Error as IoError};
use std::mem::{align_of, size_of};
use std::path::{PathBuf};
use std::slice::{from_raw_parts};
use std::str::{from_utf8};

pub const BLOCK_SZ: u64 = 512;

#[derive(Debug)]
pub enum Error {
  Io(IoError),
  CStrFromBytes,
  FromUtf8,
  FromStrRadix,
  Nul,
  Octal,
  ChksumStr,
  Chksum,
}

impl From<IoError> for Error {
  fn from(error: IoError) -> Error {
    Error::Io(error)
  }
}

// [Util-API] A record in the archive. Contains a minimal amount of metadata
// useful for out-of-core processing.
#[derive(Debug)]
pub struct TarEntry {
  pub header_pos:   u64,
  pub entry_pos:    u64,
  pub entry_sz:     u64,
  pub is_file:      bool,
  pub path:         PathBuf,
}

impl TarEntry {
  pub fn position(&self) -> u64 {
    self.entry_pos
  }

  pub fn size(&self) -> u64 {
    self.entry_sz
  }
}

pub fn check_tar_header_block(header_buf: &[u8]) -> Result<bool, ()> {
  let mut chksum = u32::max_value();
  for k in 148 .. 156 {
    if header_buf[k] == 0 {
      return Err(());
    } else if header_buf[k] == 0x20 {
    } else if header_buf[k] >= b'0' && header_buf[k] <= b'7' {
      let start = k;
      let mut end = 156;
      for k in start + 1 .. 156 {
        if header_buf[k] == 0 {
          end = k;
          break;
        } else if header_buf[k] == 0x20 {
          end = k;
          break;
        } else if header_buf[k] >= b'0' && header_buf[k] <= b'7' {
        } else {
          return Err(());
        }
      }
      let chksum_str = match from_utf8(&header_buf[start .. end]) {
        Err(_) => return Err(()),
        Ok(s) => s
      };
      chksum = match u32::from_str_radix(chksum_str, 8) {
        Err(_) => return Err(()),
        Ok(v) => v
      };
      break;
    } else {
      return Err(());
    }
  }
  let block_len = BLOCK_SZ as usize;
  let mut sum = 0;
  for k in 0 .. block_len {
    let x = if k >= 148 && k < 156 {
      0x20
    } else {
      header_buf[k] as u32
    };
    sum += x;
  }
  Ok(chksum == sum)
}

pub struct TarFile<F> {
  file: RefCell<F>,
}

impl<F> TarFile<F> {
  pub fn new(file: F) -> TarFile<F> {
    TarFile{file: RefCell::new(file)}
  }

  pub fn entry_iter<'a>(&'a self) -> TarEntryIter<'a, F> {
    TarEntryIter{
      file: Some(self),
      buf: Vec::new(),
      cur: 0,
    }
  }

  pub fn slice_subreader_at<'a>(&'a self, pos: u64, sz: u64) -> TarFileSlice<'a, F> {
    TarFileSlice{
      file: self,
      init: false,
      eof: false,
      err: false,
      pos,
      end: pos + sz,
    }
  }
}

pub struct TarFileSlice<'a, F> {
  file: &'a TarFile<F>,
  init: bool,
  eof: bool,
  err: bool,
  pos: u64,
  end: u64,
}

impl<'a, F: Read + Seek> Read for TarFileSlice<'a, F> {
  fn read(&mut self, buf: &mut [u8]) -> Result<usize, IoError> {
    if self.err {
      panic!("bug");
    }
    if self.eof {
      return Ok(0);
    }
    if self.pos >= self.end {
      self.eof = true;
      return Ok(0);
    }
    let mut file = self.file.file.borrow_mut();
    if !self.init {
      file.seek(SeekFrom::Start(self.pos))?;
      self.init = true;
    }
    let rem = min(buf.len() as u64, self.end - self.pos);
    let read_len = match file.read(&mut buf[ .. rem as usize]) {
      Err(e) => {
        self.err = true;
        return Err(e);
      }
      Ok(n) => n
    };
    assert!(read_len as u64 <= rem);
    self.pos += read_len as u64;
    Ok(read_len)
  }
}

pub struct TarEntryIter<'a, F> {
  file: Option<&'a TarFile<F>>,
  buf: Vec<u8>,
  cur: u64,
}

impl<'a, F: Read + Seek> TarEntryIter<'a, F> {
  pub fn eof(&self) -> bool {
    self.file.is_none()
  }

  pub fn _read_block_at(&mut self, pos: u64) -> Result<(), Error> {
    let block_len = BLOCK_SZ as usize;
    if self.buf.len() < block_len {
      self.buf.resize(block_len, 0);
    }
    let mut file = self.file.as_ref().unwrap().file.borrow_mut();
    file.seek(SeekFrom::Start(pos))?;
    file.read_exact(&mut self.buf[ .. block_len])?;
    Ok(())
  }

  pub fn _block_buf(&self) -> &[u8] {
    &self.buf
  }

  pub fn _block_buf_as_u64s(&self) -> &[u64] {
    let buf = self._block_buf();
    assert_eq!(0, buf.as_ptr().align_offset(align_of::<u64>()));
    assert_eq!(0, buf.len() % size_of::<u64>());
    unsafe { from_raw_parts(buf.as_ptr() as *const u64, buf.len() / size_of::<u64>()) }
  }
}

impl<'a, F: Read + Seek> Iterator for TarEntryIter<'a, F> {
  type Item = Result<TarEntry, Error>;

  fn next(&mut self) -> Option<Result<TarEntry, Error>> {
    let eof = self.eof();
    //println!("DEBUG: TarEntryIter::next: eof={:?} cur={}", eof, self.cur);
    if eof {
      return None;
    }
    let header_pos = self.cur;
    let entry_pos = header_pos + BLOCK_SZ;
    let mut halt = true;
    match self._read_block_at(self.cur) {
      Err(e) => return Some(Err(e)),
      Ok(_) => {}
    }
    for &x in self._block_buf_as_u64s() {
      if x != 0 {
        halt = false;
        break;
      }
    }
    if halt {
      self.file = None;
      return None;
    }
    let header_buf = self._block_buf();
    let mut path_len = 0;
    for k in 0 .. 100 {
      if header_buf[k] == 0 {
        path_len = k;
        break;
      }
    }
    let path_cstr = match CStr::from_bytes_with_nul(&header_buf[ .. path_len + 1]) {
      Err(_) => {
        //println!("DEBUG: TarEntryIter::next: error: path = {:?}", &header_buf[ .. 100]);
        return Some(Err(Error::CStrFromBytes));
      }
      Ok(s) => s
    };
    let path = PathBuf::from(path_cstr.to_str().unwrap());
    let mut entry_sz = 0;
    for k in 124 .. 136 - 1 {
      if header_buf[k] == 0 {
        break;
      } else if header_buf[k] == 0x20 {
      } else if header_buf[k] >= b'0' && header_buf[k] <= b'7' {
        let entry_sz_str = match from_utf8(&header_buf[k .. 136 - 1]) {
          Err(_) => {
            return Some(Err(Error::FromUtf8));
          }
          Ok(s) => s
        };
        match header_buf[136 - 1] {
          0 | 0x20 => {}
          _ => {
            //println!("DEBUG: TarEntryIter::next: error: nul = {:?}", header_buf[136 - 1]);
            return Some(Err(Error::Nul));
          }
        }
        entry_sz = match u64::from_str_radix(entry_sz_str, 8) {
          Err(_) => {
            //println!("DEBUG: TarEntryIter::next: error: size = {:?}", entry_sz_str);
            return Some(Err(Error::FromStrRadix));
          }
          Ok(v) => v
        };
        break;
      } else {
        return Some(Err(Error::Octal));
      }
    }
    match check_tar_header_block(&header_buf) {
      Err(_) => {
        return Some(Err(Error::ChksumStr));
      }
      Ok(false) => {
        return Some(Err(Error::Chksum));
      }
      Ok(true) => {}
    }
    let typeflag = header_buf[156];
    let is_file = match typeflag {
      b'0' | b'\0' => true,
      _ => false
    };
    self.cur = entry_pos + ((entry_sz + BLOCK_SZ - 1) / BLOCK_SZ) * BLOCK_SZ;
    Some(Ok(TarEntry{
      header_pos,
      entry_pos,
      entry_sz,
      is_file,
      path,
    }))
  }
}
