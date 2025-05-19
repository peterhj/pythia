#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::{IntoPyObjectExt};
use serde::{Deserialize, Serialize};
use serde::de::{Deserializer};
use serde::ser::{Serializer};
use smol_str::{SmolStr};

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

const MAX_ONE_B: u32 = 0x80;
const MAX_TWO_B: u32 = 0x800;
const MAX_THREE_B: u32 = 0x10000;

// `len_utf8` and `utf8_char_width` below from rust libcore
// (Apache-2.0/MIT).

#[inline]
pub const fn len_utf8(code: u32) -> usize {
  if code < MAX_ONE_B {
    1
  } else if code < MAX_TWO_B {
    2
  } else if code < MAX_THREE_B {
    3
  } else {
    4
  }
}

// https://tools.ietf.org/html/rfc3629
const UTF8_CHAR_WIDTH: &[u8; 256] = &[
  // 1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 1
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 2
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 3
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 4
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 5
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 6
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 7
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 8
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 9
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // A
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // B
  0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // D
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // E
  4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // F
];

/// Given a first byte, determines how many bytes are in this UTF-8 character.
#[inline]
pub const fn utf8_char_width(b: u8) -> usize {
  UTF8_CHAR_WIDTH[b as usize] as usize
}

pub fn safe_ascii(s: &[u8]) -> SmolStr {
  let mut buf = String::new();
  for &x in s.iter() {
    if x <= 0x20 {
      buf.push(' ');
    } else if x < 0x7f {
      buf.push(x.try_into().unwrap());
    } else {
      buf.push('?');
    }
  }
  buf.into()
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SafeStr {
  raw:  SmolStr,
}

impl From<SmolStr> for SafeStr {
  fn from(raw: SmolStr) -> SafeStr {
    SafeStr{raw}
  }
}

impl From<String> for SafeStr {
  fn from(s: String) -> SafeStr {
    SafeStr{raw: s.into()}
  }
}

impl<'a> From<&'a str> for SafeStr {
  fn from(s: &'a str) -> SafeStr {
    SafeStr{raw: s.into()}
  }
}

#[cfg(feature = "pyo3")]
impl<'py> FromPyObject<'py> for SafeStr {
  fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
    let s: String = obj.extract()?;
    Ok(s.into())
  }
}

#[cfg(feature = "pyo3")]
impl<'py> IntoPyObject<'py> for SafeStr {
  type Target = PyAny;
  type Output = Bound<'py, Self::Target>;
  type Error  = PyErr;

  fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
    Ok(self.raw.into_bound_py_any(py)?)
  }
}

impl SafeStr {
  pub fn is_empty(&self) -> bool {
    self.raw.is_empty()
  }

  pub fn as_raw_str(&self) -> &str {
    self.raw.as_str()
  }

  pub fn set_raw_str<S: Into<SmolStr>>(&mut self, s: S) {
    self.raw = s.into();
  }
}

impl<'de> Deserialize<'de> for SafeStr {
  fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    let raw = SmolStr::deserialize(deserializer)?;
    Ok(SafeStr{raw})
  }
}

impl Serialize for SafeStr {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    self.raw.serialize(serializer)
  }
}

impl Debug for SafeStr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    // FIXME: escaping in debug fmt?
    write!(f, "{:?}", safe_ascii(self.raw.as_bytes()))
  }
}

impl Display for SafeStr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", safe_ascii(self.raw.as_bytes()))
  }
}

#[derive(Clone, Copy, Debug)]
pub struct StrParserConfig {
  pub delim: &'static str,
  pub lines: bool,
  pub tabs: bool,
}

impl Default for StrParserConfig {
  fn default() -> StrParserConfig {
    StrParserConfig{
      delim: "\"",
      lines: false,
      tabs: false,
    }
  }
}

impl StrParserConfig {
  pub fn parser_from_str_at<'this>(self, buf: &'this str, start: usize) -> StrParser<'this, str> {
    StrParser{
      cfg: self,
      buf,
      start,
      pos: start,
    }
  }
}

pub struct StrParser<'this, S: ?Sized> {
  cfg:  StrParserConfig,
  buf:  &'this S,
  start: usize,
  pos:  usize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ErrorCode {
  InvalidSyntax,
  InvalidNumber,
  EOFWhileParsingObject,
  EOFWhileParsingArray,
  EOFWhileParsingValue,
  EOFWhileParsingString,
  KeyMustBeAString,
  ExpectedColon,
  TrailingCharacters,
  TrailingComma,
  InvalidEscape,
  InvalidUnicodeCodePoint,
  LoneLeadingSurrogateInHexEscape,
  UnexpectedEndOfHexEscape,
  UnrecognizedHex,
  NotFourDigit,
  ControlCharacterInString(u8),
  NotUtf8,
  EmptyDelimiter,
  InvalidStartDelimiter,
  TruncatedStartDelimiter,
}

use self::ErrorCode::*;

// msg, line, col
//pub struct SyntaxError(ErrorCode, usize, usize);
pub type SyntaxError = ErrorCode;

impl<'this, S: ?Sized + AsRef<str>> StrParser<'this, S> {
  fn next_char(&mut self) -> Option<char> {
    let buf = self.buf.as_ref();
    if self.pos >= buf.len() {
      None
    } else {
      let c = buf.get(self.pos .. ).unwrap().chars().next();
      if let Some(c) = c {
        self.pos += len_utf8(c as _);
      }
      c
    }
  }

  fn error<E>(&self, reason: ErrorCode) -> Result<E, SyntaxError> {
    //Err(SyntaxError(reason, self.line, self.col))
    Err(reason)
  }

  fn decode_hex_escape(&mut self) -> Result<u16, SyntaxError> {
    let mut i = 0;
    let mut n = 0;
    while i < 4 {
      let c = self.next_char();
      if let Some(c) = c {
        n = match c {
          '0' ..= '9' => n * 16 + ((c as u16) - ('0' as u16)),
          'a' ..= 'f' => n * 16 + (10 + (c as u16) - ('a' as u16)),
          'A' ..= 'F' => n * 16 + (10 + (c as u16) - ('A' as u16)),
          _ => return self.error(InvalidEscape)
        };
      } else {
        return self.error(InvalidEscape);
      }

      i += 1;
    }

    Ok(n)
  }

  pub fn offset(&self) -> usize {
    self.pos - self.start
  }

  pub fn parse_str(&mut self) -> Result<String, SyntaxError> {
    let dc0 = self.cfg.delim.chars().next().ok_or_else(|| SyntaxError::EmptyDelimiter)?;
    let d_off = len_utf8(dc0 as _);
    let d_len = self.cfg.delim.len();

    {
      let buf = self.buf.as_ref();
      /*assert_eq!(self.pos, self.start);*/
      let next_pos = self.start + d_len;
      if buf.len() >= next_pos {
        if buf.get(self.pos .. next_pos).unwrap() ==
           self.cfg.delim
        {
          self.pos = next_pos;
          /*assert_eq!(self.pos, self.start + d_len);*/
        } else {
          return Err(SyntaxError::InvalidStartDelimiter);
        }
      } else {
        return Err(SyntaxError::TruncatedStartDelimiter);
      }
    }

    // FIXME: just recognize the string, not re-materialize it.
    let mut res = String::new();

    let mut escape = false;
    loop {
      let c = self.next_char();
      if c.is_none() {
        return self.error(EOFWhileParsingString);
      }

      if escape {
        match c {
          Some('"') => res.push('"'),
          Some('\\') => res.push('\\'),
          Some('/') => res.push('/'),
          Some('b') => res.push('\x08'),
          Some('f') => res.push('\x0c'),
          Some('n') => res.push('\n'),
          Some('r') => res.push('\r'),
          Some('t') => res.push('\t'),
          Some('u') => match self.decode_hex_escape()? {
            0xDC00 ..= 0xDFFF => {
              return self.error(LoneLeadingSurrogateInHexEscape)
            }

            // Non-BMP characters are encoded as a sequence of
            // two hex escapes, representing UTF-16 surrogates.
            n1 @ 0xD800 ..= 0xDBFF => {
              match (self.next_char(), self.next_char()) {
                (Some('\\'), Some('u')) => (),
                _ => return self.error(UnexpectedEndOfHexEscape),
              }

              let n2 = self.decode_hex_escape()?;
              if n2 < 0xDC00 || n2 > 0xDFFF {
                return self.error(LoneLeadingSurrogateInHexEscape)
              }
              let c = (((n1 - 0xD800) as u32) << 10 |
                   (n2 - 0xDC00) as u32) + 0x1_0000;
              res.push(char::from_u32(c).unwrap());
            }

            n => match char::from_u32(n as u32) {
              Some(c) => res.push(c),
              None => return self.error(InvalidUnicodeCodePoint),
            },
          },
          _ => return self.error(InvalidEscape),
        }
        escape = false;
      } else if c == Some('\\') {
        escape = true;
      } else {
        if c == Some(dc0) {
          let buf = self.buf.as_ref();
          let next_pos = self.pos + d_len - d_off;
          if buf.len() >= next_pos {
            if buf.get(self.pos .. next_pos).unwrap() ==
               self.cfg.delim.get(d_off .. ).unwrap()
            {
              self.pos = next_pos;
              return Ok(res);
            }
          }
        }
        match c {
          /*Some('"') => {
            let _ = self.next_char();
            return Ok(res);
          }*/
          Some(c) if c <= '\u{1F}' => {
            if self.cfg.lines && (c == '\n' || c == '\r') {
              res.push(c);
            } else if self.cfg.tabs && c == '\t' {
              res.push(c);
            } else {
              return self.error(ControlCharacterInString(c.try_into().unwrap()));
            }
          }
          Some(c) => {
            res.push(c);
          }
          None => unreachable!()
        }
      }
    }
  }
}
