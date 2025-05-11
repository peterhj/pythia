static LOWER_CHARS:     &'static [u8] = b"0123456789abcdef";
static UPPER_CHARS:     &'static [u8] = b"0123456789ABCDEF";
static REV_LOWER_CHARS: &'static [u8] = b"zyxwvutsrqponmlk";
static REV_UPPER_CHARS: &'static [u8] = b"ZYXWVUTSRQPONMLK";

#[inline]
pub fn encode_lower(bytes: &[u8]) -> String {
  let mut s = String::new();
  for &u in bytes.iter() {
    s.push(LOWER_CHARS[(u >> 4) as usize] as char);
    s.push(LOWER_CHARS[(u & 15) as usize] as char);
  }
  s
}

#[inline]
pub fn encode_upper(bytes: &[u8]) -> String {
  let mut s = String::new();
  for &u in bytes.iter() {
    s.push(UPPER_CHARS[(u >> 4) as usize] as char);
    s.push(UPPER_CHARS[(u & 15) as usize] as char);
  }
  s
}

#[inline]
pub fn encode_rev_lower(bytes: &[u8]) -> String {
  let mut s = String::new();
  for &u in bytes.iter() {
    s.push(REV_LOWER_CHARS[(u >> 4) as usize] as char);
    s.push(REV_LOWER_CHARS[(u & 15) as usize] as char);
  }
  s
}

#[inline]
pub fn encode_rev_upper(bytes: &[u8]) -> String {
  let mut s = String::new();
  for &u in bytes.iter() {
    s.push(REV_UPPER_CHARS[(u >> 4) as usize] as char);
    s.push(REV_UPPER_CHARS[(u & 15) as usize] as char);
  }
  s
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(u8)]
pub enum HexFormat {
  #[default]
  Lower,
  Upper,
  RevLower,
  RevUpper,
}

impl HexFormat {
  #[inline]
  pub fn lower(self) -> HexFormat {
    match self {
      HexFormat::Upper => {
        HexFormat::Lower
      }
      HexFormat::RevUpper => {
        HexFormat::RevLower
      }
      _ => self
    }
  }

  #[inline]
  pub fn upper(self) -> HexFormat {
    match self {
      HexFormat::Lower => {
        HexFormat::Upper
      }
      HexFormat::RevLower => {
        HexFormat::RevUpper
      }
      _ => self
    }
  }

  #[inline]
  pub fn rev(self) -> HexFormat {
    match self {
      HexFormat::Lower => {
        HexFormat::RevLower
      }
      HexFormat::Upper => {
        HexFormat::RevUpper
      }
      _ => self
    }
  }

  #[inline]
  pub fn to_string(self, bytes: &[u8]) -> String {
    match self {
      HexFormat::Lower => {
        encode_lower(bytes)
      }
      HexFormat::Upper => {
        encode_upper(bytes)
      }
      HexFormat::RevLower => {
        encode_rev_lower(bytes)
      }
      HexFormat::RevUpper => {
        encode_rev_upper(bytes)
      }
    }
  }
}
