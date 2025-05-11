/*#[cfg(target_os = "linux")]
use libc::{__errno_location};
#[cfg(target_os = "macos")]
use libc::{__error as __errno_location};
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
use libc::{__errno as __errno_location};*/
use libc::{_SC_PAGESIZE, sysconf};
use once_cell::sync::{Lazy};
#[cfg(feature = "rayon")]
use rayon::{ThreadPoolBuilder};

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;

pub static ONCE_SMP_INFO: Lazy<SmpInfo> = Lazy::new(|| SmpInfo::new());
thread_local! {
  pub static TL_SMP_INFO: SmpInfo = SmpInfo::global_clone();
}

#[derive(Clone)]
pub struct SmpInfo {
  pub sc_page_sz: Option<usize>,
  #[cfg(target_os = "linux")]
  pub lscpu: Option<self::linux::LscpuParse>,
  #[cfg(target_os = "macos")]
  pub sysctl: Option<self::macos::SysctlParse>,
}

impl SmpInfo {
  pub fn new() -> SmpInfo {
    let ret = unsafe { sysconf(_SC_PAGESIZE) };
    let sc_page_sz = if ret <= 0 {
      None
    } else {
      ret.try_into().ok()
    };
    #[cfg(target_os = "linux")]
    let lscpu = self::linux::LscpuParse::open().ok().map(|inner| inner.into());
    #[cfg(target_os = "macos")]
    let sysctl = self::macos::SysctlParse::open().ok().map(|inner| inner.into());
    SmpInfo{
      sc_page_sz,
      #[cfg(target_os = "linux")]
      lscpu,
      #[cfg(target_os = "macos")]
      sysctl,
    }
  }

  pub fn global_clone() -> SmpInfo {
    ONCE_SMP_INFO.clone()
  }

  pub fn tl_clone() -> SmpInfo {
    TL_SMP_INFO.with(|info| info.clone())
  }

  pub fn arch_page_size(&self) -> Option<usize> {
    self.sc_page_sz
  }

  pub fn physical_core_count(&self) -> Option<u16> {
    #[cfg(target_os = "linux")]
    if let Some(lscpu) = self.lscpu.as_ref() {
      return lscpu.physical_core_count();
    }
    #[cfg(target_os = "macos")]
    if let Some(sysctl) = self.sysctl.as_ref() {
      return sysctl.physical_core_count();
    }
    None
  }

  pub fn physical_memory_size(&self) -> Option<usize> {
    #[cfg(target_os = "macos")]
    if let Some(sysctl) = self.sysctl.as_ref() {
      return sysctl.physical_memory_size();
    }
    None
  }
}

#[cfg(not(feature = "rayon"))]
pub fn _init_rayon() {
}

#[cfg(feature = "rayon")]
pub fn _init_rayon() {
  let phy_core_ct = SmpInfo::tl_clone()
    .physical_core_count()
    .unwrap_or(1) as _;
  ThreadPoolBuilder::new()
    .num_threads(phy_core_ct)
    .build_global()
    .unwrap();
}

pub fn init_smp() {
  _init_rayon();
}
