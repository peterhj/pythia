use crate::algo::cell::{RefCell};
use crate::algo::str::{SafeStr};

use term_colors::{Colorize};

use std::any::{Any};
//use std::cell::{RefCell};
use std::io::{
  BufRead, Read, Write, Error as IoError,
  BufReader, BufWriter, Cursor, stdout
};

pub trait TAPProver {
  fn prove<W: Write + ?Sized>(&self, writer: &mut W) -> Result<(), IoError>;
}

pub trait TAPParser {
}

pub type DefaultTAPParser = EchoTAPParser;

#[derive(Default)]
pub struct EchoTAPParser {
}

impl EchoTAPParser {
  pub fn parse<P: TAPProver>(producer: P) /*-> EchoTAPParser */{
    //let mut buf = BufWriter::new(stdout().lock());
    let buf = Vec::<u8>::new();
    let mut buf = BufWriter::new(Cursor::new(buf));
    producer.prove(&mut buf).unwrap();
    let buf = buf.into_inner().unwrap();
    let buf = buf.into_inner();
    //println!("DEBUG: EchoTAPParser: {:?}", buf.as_bytes());
    let buf = BufReader::new(Cursor::new(buf));
    //const ok_prefix_1: &'static [u8] = &[27, 91, 51, 50, 109, 111, 107, 27, 91, 48, 109, 32];
    //const ok_prefix_2: &'static [u8] = &[27, 91, 49, 59, 51, 50, 109, 111, 107, 27, 91, 48, 109, 32];
    //const not_ok_prefix_1: &'static [u8] = &[27, 91, 51, 49, 109, 110, 111, 116, 32, 111, 107, 27, 91, 48, 109, 32];
    //const not_ok_prefix_2: &'static [u8] = &[27, 91, 49, 59, 51, 49, 109, 110, 111, 116, 32, 111, 107, 27, 91, 48, 109, 32];
    const OK_PREFIX_1: &'static str = "\u{1b}[32mok\u{1b}[0m ";
    const OK_PREFIX_2: &'static str = "\u{1b}[1;32mok\u{1b}[0m ";
    const NOT_OK_PREFIX_1: &'static str = "\u{1b}[31mnot ok\u{1b}[0m ";
    const NOT_OK_PREFIX_2: &'static str = "\u{1b}[1;31mnot ok\u{1b}[0m ";
    let mut ok_ct: usize = 0;
    let mut not_ok_ct: usize = 0;
    for line in buf.lines() {
      let line = line.unwrap();
      let line_buf = line.as_bytes();
      if line.starts_with("ok ")
      || line.starts_with(OK_PREFIX_1)
      || line.starts_with(OK_PREFIX_2)
      {
        ok_ct += 1;
      } else if line.starts_with("not ok ")
             || line.starts_with(NOT_OK_PREFIX_1)
             || line.starts_with(NOT_OK_PREFIX_2)
      {
        not_ok_ct += 1;
      } else if line_buf.starts_with(b"#") {
      } else if line_buf.starts_with(b"1..") {
      } else {
        println!("DEBUG: EchoTAPParser: {:?}", line.as_bytes());
        println!("DEBUG: EchoTAPParser: {:?}", line);
        println!("DEBUG: EchoTAPParser: {}", line);
        panic!("bug");
        //break;
      }
      println!("{}", line);
    }
    if not_ok_ct > 0 {
      println!("Result: {}", "FAIL".red().bold());
      println!("Failed {} / {} test programs.",
          not_ok_ct,
          not_ok_ct + ok_ct
      );
    } else {
      println!("All tests successful.");
      println!("Result: {}", "PASS".green().bold());
      println!("Passed {} test programs.", ok_ct);
    }
  }
}

// FIXME: Box<dyn Write> not flexible enough to unwrap...
pub fn wrap_tap_writer<W: 'static + Write>(writer: W) -> Box<dyn Write> {
  Box::new(writer)
}

pub trait TAPWriteBuffer: Write + Any {
  fn as_any(&self) -> &dyn Any;
  fn as_mut_any(&mut self) -> &mut dyn Any;
  fn writer(&mut self) -> &mut dyn Write;
}

impl<W: Write + Any> TAPWriteBuffer for W {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn as_mut_any(&mut self) -> &mut dyn Any {
    self
  }

  fn writer(&mut self) -> &mut dyn Write {
    self
  }
}

pub struct TAPOutput {
  pub writer:   RefCell<Box<dyn Write>>,
  pub verbose:  i8,
}

impl Default for TAPOutput {
  fn default() -> TAPOutput {
    TAPOutput::stdout()
  }
}

impl TAPOutput {
  pub fn stdout_writer() -> Box<dyn Write> {
    Box::new(std::io::stdout())
  }

  pub fn stdout() -> TAPOutput {
    TAPOutput{
      writer:   RefCell::new(Box::new(std::io::stdout())),
      verbose:  0,
    }
  }
}

macro_rules! _errorln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 0;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _errorln;

macro_rules! _warningln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 1;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _warningln;

macro_rules! _infoln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 2;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _infoln;

macro_rules! _debugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 3;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _debugln;

macro_rules! _vdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 4;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _vdebugln;

macro_rules! _vvdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 5;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _vvdebugln;

macro_rules! _vvvdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 6;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _vvvdebugln;

macro_rules! _traceln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 7;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _traceln;
