#[cfg(unix)]
use libc::{
  PROT_NONE, PROT_READ, PROT_WRITE,
  MAP_FAILED, MAP_ANON, MAP_PRIVATE,
  mmap, munmap,
};

use std::ffi::{c_void};
use std::fs::{File};
#[cfg(unix)] use std::os::unix::fs::{MetadataExt};
#[cfg(unix)] use std::os::unix::io::{AsRawFd};
use std::path::{Path};
use std::ptr::{null_mut};
use std::slice::{from_raw_parts};
use std::sync::{Arc};

#[derive(Clone)]
pub struct MmapFile {
  pub file: Arc<File>,
  pub ptr:  *mut c_void,
  pub size: usize,
}

impl Drop for MmapFile {
  #[cfg(not(unix))]
  fn drop(&mut self) {
    unimplemented!();
  }

  #[cfg(unix)]
  fn drop(&mut self) {
    // FIXME: weak count?
    if Arc::strong_count(&self.file) == 1 {
      assert!(!self.ptr.is_null());
      let ret = unsafe { munmap(self.ptr, self.size) };
      assert_eq!(ret, 0);
    }
  }
}

impl MmapFile {
  pub fn open<P: AsRef<Path>>(path: P) -> Result<MmapFile, ()> {
    let file = Arc::new(File::open(path).map_err(|_| ())?);
    MmapFile::from_file(&file)
  }

  #[cfg(not(unix))]
  pub fn from_file(_f: &Arc<File>) -> Result<MmapFile, ()> {
    unimplemented!();
  }

  #[cfg(unix)]
  pub fn from_file(f: &Arc<File>) -> Result<MmapFile, ()> {
    let size = f.metadata().unwrap().size();
    if size > usize::max_value() as u64 {
      return Err(());
    }
    let size = size as usize;
    let fd = f.as_raw_fd();
    let ptr = unsafe { mmap(null_mut(), size, PROT_READ, MAP_PRIVATE, fd, 0) };
    if ptr == MAP_FAILED {
      return Err(());
    }
    assert!(!ptr.is_null());
    Ok(MmapFile{ptr, size, file: f.clone()})
  }

  pub fn as_ptr(&self) -> *mut c_void {
    self.ptr
  }

  pub fn len(&self) -> usize {
    self.size
  }

  #[cfg(unix)]
  pub unsafe fn as_bytes_unsafe(&self) -> &[u8] {
    from_raw_parts(self.ptr as *mut u8 as *const u8, self.size)
  }

  #[cfg(all(unix, target_os = "macos"))]
  pub fn as_bytes(&self) -> &[u8] {
    // NB: some ~decade old anecdotal evidence suggests that writes
    // made by another proc are not visible to this current proc's
    // MAP_PRIVATE mapping;
    // see: <https://stackoverflow.com/questions/14670869/file-changes-after-a-mmap-in-os-x-ios>
    unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8, self.size) }
  }
}

#[cfg(all(unix, target_os = "macos"))]
impl AsRef<[u8]> for MmapFile {
  fn as_ref(&self) -> &[u8] {
    self.as_bytes()
  }
}
