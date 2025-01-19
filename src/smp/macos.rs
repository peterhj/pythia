use std::io::{BufRead, Cursor};
use std::process::{Command, Stdio};

#[derive(Clone, Copy)]
pub struct SysctlParse {
  pub core_ct: u16,
  pub mem_sz: usize,
}

impl SysctlParse {
  pub fn open() -> Result<SysctlParse, ()> {
    let out = Command::new("sysctl")
        .arg("-a")
        .stdout(Stdio::piped())
        .output()
        .map_err(|_| ())?;
    if !out.status.success() {
      return Err(());
    }
    SysctlParse::parse(out.stdout)
  }

  pub fn parse<O: AsRef<[u8]>>(out: O) -> Result<SysctlParse, ()> {
    let mut info = SysctlParse{
      core_ct: 0,
      mem_sz: 0,
    };
    let out = out.as_ref();
    for line in Cursor::new(out).lines() {
      let line = line.unwrap();
      if line.is_empty() {
        break;
      }
      let mut line_parts = line.split_ascii_whitespace();
      match (line_parts.next(), line_parts.next()) {
        (Some(key), Some(val)) => {
          match key {
            "hw.perflevel0.physicalcpu:" => {
              info.core_ct = val.parse().unwrap();
            }
            "hw.memsize_usable:" => {
              info.mem_sz = val.parse().unwrap();
            }
            _ => {}
          }
        }
        _ => {}
      }
    }
    Ok(info)
  }

  pub fn physical_core_count(&self) -> Option<u16> {
    if self.core_ct <= 0 {
      return None;
    }
    Some(self.core_ct)
  }

  pub fn physical_memory_size(&self) -> Option<usize> {
    if self.mem_sz <= 0 {
      return None;
    }
    Some(self.mem_sz)
  }
}
