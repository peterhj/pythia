// TODO: temporarily disabled lint for debugging.
#![allow(unused_variables)]

use crate::algo::{SmolStr};
use crate::algo::cell::{RefCell};
use crate::algo::str::{SafeStr, StrParserConfig, StrParser};
use crate::panick::{Loc, loc};
use crate::tap::{TAPOutput, _debugln};

use bitflags::{bitflags, bitflags_match};
use regex::{Regex, RegexSet};

//use std::cell::{RefCell};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::io::{Write};
use std::mem::{replace};
use std::ops::{Range};
use std::panic::{Location};

pub type Span = Range<usize>;

pub trait Hull<Rhs> {
  fn hull(&self, rhs: Rhs) -> Self where Self: Sized;
}

impl Hull<Span> for Span {
  #[inline]
  fn hull(&self, rhs: Span) -> Span {
    self.hull(&rhs)
  }
}

impl<'a> Hull<&'a Span> for Span {
  #[inline]
  fn hull(&self, rhs: &'a Span) -> Span {
    let start = self.start.min(rhs.start);
    let end = self.end.max(rhs.end);
    Span{start, end}
  }
}

pub struct RegexMapBuilder<T> {
  pats: Vec<SmolStr>,
  funs: Vec<Box<dyn Fn(&str) -> T>>,
}

impl<T> RegexMapBuilder<T> {
  pub fn new() -> RegexMapBuilder<T> {
    RegexMapBuilder{
      pats: Vec::new(),
      funs: Vec::new(),
    }
  }

  pub fn push<'s, F: 'static + Fn(&str) -> T>(&mut self, pat: &'s str, fun: F) {
    self.pats.push(pat.into());
    self.funs.push(Box::new(fun));
  }
}

pub struct RegexMap<T> {
  set:  RegexSet,
  // FIXME: this could be compressed into one pointer word.
  pats: RefCell<Box<[Option<Regex>]>>,
  funs: Box<[Box<dyn Fn(&str) -> T>]>,
}

impl<T> From<RegexMapBuilder<T>> for RegexMap<T> {
  fn from(b: RegexMapBuilder<T>) -> RegexMap<T> {
    let set = RegexSet::new(b.pats).unwrap();
    let mut pats = Vec::with_capacity(b.funs.len());
    pats.resize(b.funs.len(), None);
    RegexMap{
      set,
      pats: RefCell::new(pats.into()),
      funs: b.funs.into(),
    }
  }
}

impl<T> RegexMap<T> {
  pub fn match_at(&self, text: &str, start: usize) -> Option<(Span, T)> {
    let haystack = text.get(start .. ).unwrap();
    let pat_idx = match self.set.matches_at(haystack, 0).into_iter().next() {
      None => return None,
      Some(pat_idx) => pat_idx
    };
    if self.pats.borrow()[pat_idx].is_none() {
      let pat_str = &self.set.patterns()[pat_idx];
      self.pats.borrow_mut()[pat_idx] = Regex::new(pat_str).unwrap().into();
    }
    let mat = self.pats.borrow()[pat_idx].as_ref().unwrap()
              .find_at(haystack, 0).unwrap();
    let mut span = mat.range();
    span.start += start;
    span.end += start;
    let val = (self.funs[pat_idx])(mat.as_str());
    Some((span, val))
  }
}

#[derive(Clone, Debug)]
pub enum Token {
  Indent(RawIndent),
  Space,
  NL,
  CR,
  Comment(SafeStr),
  Backslash,
  Comma,
  LDotDash,
  LDotEq,
  LDotTick,
  LDotParen,
  Ellipsis,
  DotDot,
  Dot,
  LQueryDash,
  RQueryDash,
  LQueryEq,
  RQueryEq,
  Query,
  Semi,
  //LColonTilde,
  LDeduct,
  LWalrus,
  LColonLt,
  LColonGt,
  LColonParen,
  Colon,
  RDotDash,
  RDeduct,
  RArrow,
  DashSlash,
  Dash,
  RDotEq,
  RWalrus,
  EqEq,
  Equal,
  SlashEq,
  Star,
  StarStar,
  TTTickUnquote,
  TTTickQuote,
  TTTick,
  TickUnquote,
  TickQuote,
  Tick,
  UTTTick,
  UTick,
  QTTTick,
  QTick,
  LParen,
  RParen,
  LBrack,
  RBrack,
  LCurly,
  RCurly,
  None,
  True,
  False,
  Unk,
  Par,
  And,
  As,
  Async,
  Await,
  Break,
  Case,
  Cases,
  //Choice,
  //Choices,
  Class,
  Continue,
  Def,
  Defclass,
  Defmatch,
  Defproc,
  Defrule,
  Del,
  Elif,
  Else,
  Enum,
  //Eval,
  Except,
  //Exec,
  Finally,
  For,
  Forall,
  Fresh,
  From,
  Global,
  If,
  Import,
  In,
  Is,
  Lambda,
  Let,
  Match,
  //Namespace,
  Nonlocal,
  Not,
  Of,
  Or,
  Pass,
  Quote,
  Raise,
  Return,
  Rule,
  Try,
  Unquote,
  While,
  With,
  Yield,
  IntLit(SafeStr),
  AtomLit(SafeStr, ()),
  PlaceIdent(SafeStr),
  Ident(SafeStr),
  DotIdent(SafeStr),
  // NB: deprecated syntax.
  /*ColonIdent(SafeStr),*/
  //_Utf8Error(Box<[u8]>),
  _Eof,
}

#[derive(Clone, Debug)]
pub struct SpanToken {
  pub span: Span,
  pub tok:  Token,
}

impl From<(Span, Token)> for SpanToken {
  fn from(t: (Span, Token)) -> SpanToken {
    SpanToken{span: t.0, tok: t.1}
  }
}

bitflags! {
  #[derive(Clone, Copy, PartialEq, Eq)]
  pub struct TokenizerFlag_: u8 {
    const BOL = 1;
    const EOF = 2;
    const PYTHON = 4;
    const PYTHIA = 8;
  }
}

pub struct Tokenizer<S> {
  imap: RegexMap<Token>,
  map:  RegexMap<Token>,
  buf:  S,
  pos:  usize,
  ind:  RawIndent,
  flag: TokenizerFlag_,
  //writer:   RefCell<Box<dyn Write>>,
  //verbose:  i8,
  tap:  TAPOutput,
}

impl<S> Tokenizer<S> {
  pub fn new(buf: S) -> Tokenizer<S> {
    let mut map = RegexMapBuilder::new();
    map.push(r"^[ \t]+", |_| Token::Space);
    //map.push(r"^\n", |_| Token::NL);
    //map.push(r"^\r", |_| Token::CR);
    //map.push(r"^\#", |_| Token::Comment(SafeStr::default()));
    map.push(r"^\\", |_| Token::Backslash);
    map.push(r"^,",  |_| Token::Comma);
    map.push(r"^\.-", |_| Token::LDotDash);
    map.push(r"^\.=", |_| Token::LDotEq);
    map.push(r"^\.`", |_| Token::LDotTick);
    map.push(r"^\.\(", |_| Token::LDotParen);
    map.push(r"^\.\.\.`", |_| Token::Ellipsis);
    map.push(r"^\.\.`", |_| Token::DotDot);
    map.push(r"^\.", |_| Token::Dot);
    map.push(r"^\?-", |_| Token::LQueryDash);
    map.push(r"^\?=", |_| Token::LQueryEq);
    map.push(r"^\?", |_| Token::Query);
    map.push(r"^;",  |_| Token::Semi);
    //map.push(r"^:\~", |_| Token::LColonTilde);
    map.push(r"^:\-", |_| Token::LDeduct);
    map.push(r"^:=", |_| Token::LWalrus);
    map.push(r"^:<", |_| Token::LColonLt);
    map.push(r"^:>", |_| Token::LColonGt);
    map.push(r"^:\(", |_| Token::LColonParen);
    map.push(r"^:",  |_| Token::Colon);
    map.push(r"^\-\.", |_| Token::RDotDash);
    map.push(r"^\-\?", |_| Token::RQueryDash);
    map.push(r"^\-:", |_| Token::RDeduct);
    map.push(r"^\->", |_| Token::RArrow);
    map.push(r"^\-/", |_| Token::DashSlash);
    map.push(r"^\-", |_| Token::Dash);
    map.push(r"^=\.", |_| Token::RDotEq);
    map.push(r"^=\?", |_| Token::RQueryEq);
    map.push(r"^=:", |_| Token::RWalrus);
    map.push(r"^==", |_| Token::EqEq);
    map.push(r"^=",  |_| Token::Equal);
    map.push(r"^/=", |_| Token::SlashEq);
    map.push(r"^\*\*", |_| Token::StarStar);
    map.push(r"^\*", |_| Token::Star);
    map.push(r"^```unquote", |_| Token::TTTickUnquote);
    map.push(r"^```quote", |_| Token::TTTickQuote);
    map.push(r"^```", |_| Token::TTTick);
    map.push(r"^`unquote", |_| Token::TickUnquote);
    map.push(r"^`quote", |_| Token::TickQuote);
    map.push(r"^`",  |_| Token::Tick);
    map.push(r"^u```", |_| Token::UTTTick);
    map.push(r"^u`", |_| Token::UTick);
    map.push(r"^q```", |_| Token::QTTTick);
    map.push(r"^q`", |_| Token::QTick);
    map.push(r"^\(", |_| Token::LParen);
    map.push(r"^\)", |_| Token::RParen);
    map.push(r"^\[", |_| Token::LBrack);
    map.push(r"^\]", |_| Token::RBrack);
    map.push(r"^\{", |_| Token::LCurly);
    map.push(r"^\}", |_| Token::RCurly);
    map.push(r"^None",  |_| Token::None);
    map.push(r"^True",  |_| Token::True);
    map.push(r"^False", |_| Token::False);
    map.push(r"^Unk",   |_| Token::Unk);
    map.push(r"^Para",  |_| Token::Par);
    map.push(r"^Par",   |_| Token::Par);
    map.push(r"^and", |_| Token::And);
    map.push(r"^async", |_| Token::Async);
    map.push(r"^as",  |_| Token::As);
    map.push(r"^await", |_| Token::Await);
    map.push(r"^break", |_| Token::Break);
    map.push(r"^cases", |_| Token::Cases);
    map.push(r"^case", |_| Token::Case);
    //map.push(r"^choices", |_| Token::Choices);
    //map.push(r"^choice", |_| Token::Choice);*/
    map.push(r"^class", |_| Token::Class);
    map.push(r"^continue", |_| Token::Continue);
    map.push(r"^defmatch", |_| Token::Defmatch);
    map.push(r"^defproc", |_| Token::Defproc);
    map.push(r"^defrule", |_| Token::Defrule);
    map.push(r"^def", |_| Token::Def);
    map.push(r"^del", |_| Token::Del);
    map.push(r"^elif", |_| Token::Elif);
    map.push(r"^else", |_| Token::Else);
    //map.push(r"^enum", |_| Token::Enum);
    //map.push(r"^eval", |_| Token::Eval);
    map.push(r"^except", |_| Token::Except);
    //map.push(r"^exec", |_| Token::Exec);
    map.push(r"^finally", |_| Token::Finally);
    map.push(r"^for", |_| Token::For);
    map.push(r"^fresh", |_| Token::Fresh);
    map.push(r"^from", |_| Token::From);
    map.push(r"^global", |_| Token::Global);
    map.push(r"^if", |_| Token::If);
    map.push(r"^import", |_| Token::Import);
    map.push(r"^in", |_| Token::In);
    map.push(r"^is", |_| Token::Is);
    map.push(r"^lambda", |_| Token::Lambda);
    map.push(r"^let", |_| Token::Let);
    map.push(r"^match", |_| Token::Match);
    map.push(r"^nonlocal", |_| Token::Nonlocal);
    map.push(r"^not", |_| Token::Not);
    map.push(r"^of", |_| Token::Of);
    map.push(r"^or", |_| Token::Or);
    map.push(r"^pass", |_| Token::Pass);
    map.push(r"^quote", |_| Token::Quote);
    map.push(r"^raise", |_| Token::Raise);
    map.push(r"^return", |_| Token::Return);
    map.push(r"^rule", |_| Token::Rule);
    map.push(r"^try", |_| Token::Try);
    map.push(r"^unquote", |_| Token::Unquote);
    map.push(r"^while", |_| Token::While);
    map.push(r"^with", |_| Token::With);
    map.push(r"^yield", |_| Token::Yield);
    let mut imap = RegexMapBuilder::new();
    imap.push(r"^[0-9]+", |s| Token::IntLit(s.into()));
    imap.push(r"^\-[0-9]+", |s| Token::IntLit(s.into()));
    imap.push(r"^\.[a-zA-Z_][a-zA-Z0-9_]*", |s| Token::DotIdent(s.into()));
    // NB: deprecated syntax.
    /*imap.push(r"^:[a-zA-Z_][a-zA-Z0-9_]*", |s| Token::ColonIdent(s.into()));*/
    imap.push(r"^[a-zA-Z_][a-zA-Z0-9_]*", |s| Token::Ident(s.into()));
    let tap = TAPOutput::default();
    Tokenizer{
      imap: imap.into(),
      map:  map.into(),
      buf,
      pos:  0,
      ind:  0,
      flag: TokenizerFlag_::BOL,
      //writer:   RefCell::new(Box::new(std::io::stdout())),
      //verbose:  0,
      tap,
    }
  }

  pub fn bol(&self) -> bool {
    bitflags_match!(self.flag & TokenizerFlag_::BOL, {
      TokenizerFlag_::BOL => true,
      _ => false
    })
  }

  pub fn set_bol(&mut self) {
    self.flag |= TokenizerFlag_::BOL;
  }

  pub fn unset_bol(&mut self) {
    self.flag &= !TokenizerFlag_::BOL;
  }

  pub fn eof(&self) -> bool {
    bitflags_match!(self.flag & TokenizerFlag_::EOF, {
      TokenizerFlag_::EOF => true,
      _ => false
    })
  }

  pub fn set_eof(&mut self) {
    self.flag |= TokenizerFlag_::EOF;
  }

  pub fn unset_eof(&mut self) {
    self.flag &= !TokenizerFlag_::EOF;
  }

  pub fn _pos(&self) -> Span {
    let start = self.pos;
    let end = self.pos;
    Span{start, end}
  }

  pub fn _advance(&mut self, o: usize) -> Span {
    let start = self.pos;
    self.pos += o;
    let end = self.pos;
    Span{start, end}
  }
}

impl<S: AsRef<str>> Iterator for Tokenizer<S> {
  type Item = SpanToken;

  fn next(&mut self) -> Option<SpanToken> {
    if self.eof() {
      let end = self.pos;
      let span = Span{start: end, end};
      let tok = Token::_Eof;
      return Some((span, tok).into());
    }
    let c = self.peek_char();
    if self.bol() {
      if let Some(c) = c {
        _debugln!(self, "DEBUG: Tokenizer::next: bol: c=0x{:x}", c as u32);
      } else {
        _debugln!(self, "DEBUG: Tokenizer::next: bol: eof");
      }
      self.unset_bol();
      if c == Some(' ') || c == Some('\t') {
        let mut indent = 0;
        let mut o = 0;
        for c in self.buf.as_ref().get(self.pos .. ).unwrap().chars() {
          match c {
            ' ' => {
              indent += 1;
              o += 1;
            }
            '\t' => {
              indent = (indent / 8 + 1) * 8;
              o += 1;
            }
            _ => break
          }
        }
        let span = self._advance(o);
        let tok = Token::Indent(indent);
        self.ind = indent;
        return Some((span, tok).into());
      }
      _debugln!(self, "DEBUG: Tokenizer::next:   non indent");
      let span = self._pos();
      let tok = Token::Indent(0);
      self.ind = 0;
      return Some((span, tok).into());
    }
    if c == Some('\n') {
      self.set_bol();
      let span = self._advance(1);
      let tok = Token::NL;
      return Some((span, tok).into());
    }
    else if c == Some('\r') {
      self.set_bol();
      let span = self._advance(1);
      let tok = Token::CR;
      return Some((span, tok).into());
    }
    // NB: re-enable python-style comments in python compat mode.
    else if /*self.compat && */c == Some('#') {
      let mut o = 0;
      for c in self.buf.as_ref().get(self.pos .. ).unwrap().chars() {
        o += 1;
        match c {
          '\n' | '\r' => {
            break;
          }
          _ => {}
        }
      }
      let span = self._advance(o);
      let tok = Token::Comment(self.buf.as_ref().get(span.clone()).unwrap().into());
      self.flag |= TokenizerFlag_::PYTHON;
      return Some((span, tok).into());
    }
    else if c == Some('-') {
      if let Some('-') = self.peek_char2() {
        let mut o = 0;
        for c in self.buf.as_ref().get(self.pos .. ).unwrap().chars() {
          o += 1;
          match c {
            '\n' | '\r' => {
              break;
            }
            _ => {}
          }
        }
        let span = self._advance(o);
        let tok = Token::Comment(self.buf.as_ref().get(span.clone()).unwrap().into());
        self.flag |= TokenizerFlag_::PYTHIA;
        return Some((span, tok).into());
      }
    }
    else if c == Some('_') {
      let mut ident = false;
      let mut o = 0;
      for c in self.buf.as_ref().get(self.pos .. ).unwrap().chars() {
        match c {
          '_' => {
            o += 1;
          }
          'a' ..= 'z' |
          'A' ..= 'Z' |
          '0' ..= '9' => {
            ident = true;
            break;
          }
          _ => break
        }
      }
      if !ident {
        let span = self._advance(o);
        let tok = Token::PlaceIdent(self.buf.as_ref().get(span.clone()).unwrap().into());
        return Some((span, tok).into());
      }
    }
    // FIXME: latex-delimited strings?
    else if c == Some('\"') {
      let c2 = self.peek_char2();
      let c3 = self.peek_char3();
      let (s, span) = if c2 == Some('\"') && c3 == Some('\"') {
        // FIXME: accept python-like block str indentation.
        //let indent = self.ind;
        let strconfig = StrParserConfig{
          delim: "\"\"\"",
          lines: true,
          tabs: true,
        };
        let mut strparser = strconfig.parser_from_str_at(self.buf.as_ref(), self.pos);
        let s: String = match strparser.parse_str() {
          Err(e) => {
            _debugln!(self, "DEBUG: Tokenizer::next: unhandled str parse error: {e:?}");
            panic!("BUG: Tokenizer::next: unhandled str parse error: {e:?}");
          }
          Ok(s) => s.into()
        };
        let span = self._advance(strparser.offset());
        (s, span)
      } else {
        let strconfig = StrParserConfig{
          delim: "\"",
          lines: false,
          tabs: false,
        };
        let mut strparser = strconfig.parser_from_str_at(self.buf.as_ref(), self.pos);
        let s: String = match strparser.parse_str() {
          Err(e) => {
            _debugln!(self, "DEBUG: Tokenizer::next: unhandled str parse error: {e:?}");
            panic!("BUG: Tokenizer::next: unhandled str parse error: {e:?}");
          }
          Ok(s) => s.into()
        };
        let span = self._advance(strparser.offset());
        (s, span)
      };
      let tok = Token::AtomLit(self.buf.as_ref().get(span.clone()).unwrap().into(), ());
      return Some((span, tok).into());
    }
    _debugln!(self, "DEBUG: Tokenizer::next: match pos={}", self.pos);
    match (self.imap.match_at(self.buf.as_ref(), self.pos),
           self.map.match_at(self.buf.as_ref(), self.pos))
    {
      (None, None) => {
        _debugln!(self, "DEBUG: Tokenizer::next:   non match");
        self.set_eof();
        let end = self.pos;
        let span = Span{start: end, end};
        let tok = Token::_Eof;
        return Some((span, tok).into());
      }
      (None, Some((span, tok))) |
      (Some((span, tok)), None) => {
        _debugln!(self, "DEBUG: Tokenizer::next:   found match: span={:?} tok={:?}", span, tok);
        self.pos = span.end;
        match &tok {
          &Token::NL | &Token::CR => {
            self.set_bol();
          }
          _ => {}
        }
        return Some((span, tok).into());
      }
      (Some((ispan, itok)), Some((span, tok))) => {
        assert_eq!(ispan.start, span.start);
        if ispan.end <= span.end {
          _debugln!(self, "DEBUG: Tokenizer::next:   found match: span={:?} tok={:?}", span, tok);
          self.pos = span.end;
          return Some((span, tok).into());
        }
        _debugln!(self, "DEBUG: Tokenizer::next:   found match: span={:?} tok={:?}", ispan, itok);
        self.pos = ispan.end;
        return Some((ispan, itok).into());
      }
    }
  }
}

impl<S: AsRef<str>> Tokenizer<S> {
  pub fn seek(&mut self, start: usize) {
    let buf_len = self.buf.as_ref().len();
    assert!(start <= buf_len);
    self.pos = start;
    if start == 0 {
      self.set_bol();
    } else {
      let v = self.buf.as_ref().as_bytes()[start-1];
      if v == b'\n' || v == b'\r' {
        self.set_bol();
      }
    }
    if start == buf_len {
      self.set_eof();
    } else {
      self.unset_eof();
    }
  }

  pub fn peek_char(&self) -> Option<char> {
    let mut cs = self.buf.as_ref().get(self.pos .. ).unwrap().chars();
    cs.next()
  }

  pub fn peek_char2(&self) -> Option<char> {
    let mut cs = self.buf.as_ref().get(self.pos .. ).unwrap().chars();
    let _ = cs.next()?;
    cs.next()
  }

  pub fn peek_char3(&self) -> Option<char> {
    let mut cs = self.buf.as_ref().get(self.pos .. ).unwrap().chars();
    let _ = cs.next()?;
    let _ = cs.next()?;
    cs.next()
  }
}

/*pub struct DebugTokenizer<S> {
  inner: Tokenizer<S>,
  verbose:  i8,
}

impl<S> DebugTokenizer<S> {
  pub fn new(buf: S) -> DebugTokenizer<S> {
    let inner = Tokenizer::new(buf);
    let verbose = 0;
    DebugTokenizer{inner, verbose}
  }

  pub fn _pos(&self) -> Span {
    self.inner._pos()
  }
}

impl<S: AsRef<str>> Iterator for DebugTokenizer<S> {
  type Item = SpanToken;

  fn next(&mut self) -> Option<SpanToken> {
    let item = self.inner.next();
    _debugln!(self, "DEBUG: Tokenizer::next: item={:?}", item);
    item
  }
}

impl<S: AsRef<str>> DebugTokenizer<S> {
  pub fn seek(&mut self, start: usize) {
    _debugln!(self, "DEBUG: Tokenizer::seek: start={}", start);
    self.inner.seek(start);
  }

  pub fn peek_char(&self) -> Option<char> {
    let c = self.inner.peek_char();
    if let Some(c) = c {
      _debugln!(self, "DEBUG: Tokenizer::peek_char: c=0x{:x}", c as u32);
    } else {
      _debugln!(self, "DEBUG: Tokenizer::peek_char: eof");
    }
    c
  }
}*/

// FIXME
pub type Ident = SafeStr;
pub type Lit = SafeStr;
pub type TermRef = Box<Term>;
pub type StmRef = Box<Stm>;

#[derive(Debug)]
pub enum Term {
  // TODO TODO
  Ident(Span, Ident),
  QualIdent(Span, TermRef, Ident),
  //StrLit(Span, Lit),
  AtomLit(Span, Lit),
  NoneLit(Span, Lit),
  BoolLit(Span, Lit),
  IntLit(Span, Lit),
  FloatLit(Span, Lit),
  ListLit(Span, Vec<TermRef>),
  Neg(Span, TermRef),
  //Tuple(Vec<TermRef>),
  Group(Span, TermRef),
  Bunch(Span, Vec<TermRef>),
  Query(Span, TermRef),
  Equal(Span, TermRef, TermRef),
  NEqual(Span, TermRef, TermRef),
  QEqual(Span, TermRef, TermRef),
  BindL(Span, TermRef, TermRef),
  BindR(Span, TermRef, TermRef),
  Subst(Span, TermRef, TermRef),
  RebindL(Span, TermRef, TermRef),
  RebindR(Span, TermRef, TermRef),
  Apply(Span, Vec<TermRef>),
  ApplyBindL(Span, TermRef, Vec<TermRef>),
  ApplyBindR(Span, Vec<TermRef>, TermRef),
  // FIXME: fold the head (middle) term into rterms/rtup.
  Effect(Span, TermRef, Vec<TermRef>),
}

impl Term {
  pub fn span(&self) -> Span {
    match self {
      &Term::Ident(ref span, ..) |
      &Term::QualIdent(ref span, ..) |
      //&Term::StrLit(ref span, ..) |
      &Term::AtomLit(ref span, ..) |
      &Term::NoneLit(ref span, ..) |
      &Term::BoolLit(ref span, ..) |
      &Term::IntLit(ref span, ..) |
      &Term::ListLit(ref span, ..) |
      &Term::Neg(ref span, ..) |
      &Term::Group(ref span, ..) |
      &Term::Bunch(ref span, ..) |
      &Term::Query(ref span, ..) |
      &Term::Equal(ref span, ..) |
      &Term::NEqual(ref span, ..) |
      &Term::QEqual(ref span, ..) |
      &Term::BindL(ref span, ..) |
      &Term::BindR(ref span, ..) |
      &Term::Subst(ref span, ..) |
      &Term::RebindL(ref span, ..) |
      &Term::RebindR(ref span, ..) |
      &Term::Apply(ref span, ..) |
      &Term::ApplyBindL(ref span, ..) |
      &Term::ApplyBindR(ref span, ..) |
      &Term::Effect(ref span, ..)
      => span.clone(),
      _ => unimplemented!()
    }
  }
}

#[derive(Debug)]
pub enum Stm {
  // TODO TODO
  Just(Span, TermRef),
  Comment(Span, ()),
  Pass(Span),
  Global(Span, Ident),
  Nonlocal(Span, Option<i16>, Ident),
  With(Span, TermRef, Vec<StmRef>),
  Try,
  If(Span, Vec<(TermRef, Vec<StmRef>)>, Option<Vec<StmRef>>),
  While,
  For,
  Match(Span, (), Vec<StmRef>),
  Def(Span, Option<DefPrefix>, (), Vec<StmRef>),
  // FIXME: def-like args (params) are more general than just
  // a list of idents; but this is a stopgap to parse something.
  Defproc(Span, Option<DefPrefix>, Ident, Vec<Option<Ident>>, Vec<StmRef>),
  Defmatch(Span, Option<DefPrefix>, Ident, Vec<Option<Ident>>, Vec<StmRef>),
  //Enum(Span, (), ),
  Cases(Span, (), Vec<StmRef>),
  Class(Span, (), Vec<StmRef>),
  Quote(Span, (), Vec<StmRef>),
  _EndQuote(Span),
}

#[derive(Clone, Copy, Debug)]
pub enum DefPrefix {
  // TODO
  //Global,
  Rule,
  //GlobalRule,
}

#[derive(Debug)]
pub struct Mod {
  pub span: Span,
  pub body: Vec<StmRef>,
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum StmStage {
  Try,
  Except,
  Finally,
  If,
  Elif,
  Else,
}

impl<'a> From<&'a Token> for StmStage {
  fn from(tok: &'a Token) -> StmStage {
    match tok {
      &Token::Elif => StmStage::Elif,
      &Token::Else => StmStage::Else,
      &Token::Except => StmStage::Except,
      &Token::Finally => StmStage::Finally,
      &Token::If => StmStage::If,
      &Token::Try => StmStage::Try,
      _ => panic!("bug")
    }
  }
}

pub type RawIndent = u32;
pub type RawBp = u16;

#[derive(Clone, Copy, Debug)]
pub enum StmIndent {
  Eq(RawIndent),
  Gt(RawIndent),
}

impl StmIndent {
  pub fn is_eq(&self) -> bool {
    match self {
      &StmIndent::Eq(_) => true,
      &StmIndent::Gt(_) => false,
    }
  }

  pub fn is_gt(&self) -> bool {
    match self {
      &StmIndent::Eq(_) => false,
      &StmIndent::Gt(_) => true,
    }
  }

  pub fn eq_to_gt(&self) -> StmIndent {
    match self {
      &StmIndent::Eq(indent) => {
        StmIndent::Gt(indent)
      }
      _ => panic!("bug")
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct StmCtx {
  _stage: Option<StmStage>,
  indent: StmIndent,
  //bp: RawBp,
}

impl Default for StmCtx {
  fn default() -> StmCtx {
    StmCtx{
      _stage: None,
      indent: StmIndent::Eq(0),
      //bp: 0,
    }
  }
}

impl StmCtx {
  pub fn term(&self) -> TermCtx {
    TermCtx{
      indent: match self.indent {
        StmIndent::Eq(indent) => indent,
        StmIndent::Gt(_) => {
          panic!("bug");
        }
      },
      bp: 0,
    }
  }
}

#[derive(Clone, Copy, Default, Debug)]
pub enum StmPrefix {
  #[default]
  _Nil,
  // TODO
  //Global,
  Rule,
}

impl StmPrefix {
  pub fn into_def(self) -> Option<DefPrefix> {
    match self {
      StmPrefix::_Nil => None,
      //StmPrefix::Global => Some(DefPrefix::Global),
      StmPrefix::Rule => Some(DefPrefix::Rule),
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct TermCtx {
  // FIXME: multi-line (...)-terms also need an original indent,
  // which is a possible lower bound for the last line indent
  // (e.g. the enclosing R-paren).
  //org_indent: RawIndent,
  indent: RawIndent,
  bp: RawBp,
}

#[derive(Clone, Debug)]
pub enum ParseError {
  Eof,
  Indent,
  EmptyBlock,
  ExpectedIdent,
  Expected(Token),
  Unexpected(Token),
  ExpectedStm,
  ExpectedBunch,
  ExpectedIntLit,
  InvalidIntLit,
  Unimpl_,
  Unimpl(Token),
  _Bot,
}

#[derive(Clone, Debug)]
pub struct ParseSpanError {
  pub span: Span,
  pub err:  ParseError,
  pub loc:  Loc,
}

impl From<(Span, ParseError)> for ParseSpanError {
  #[track_caller]
  fn from(t: (Span, ParseError)) -> ParseSpanError {
    let loc = loc();
    ParseSpanError{span: t.0, err: t.1, loc}
  }
}

pub type Parser<S> = FastParser<S>;

pub struct FastParser<S> {
  tokens:   Tokenizer<S>,
  //tokens:   DebugTokenizer<S>,
  //tab:  _,
  cur:      Option<SpanToken>,
  peek:     Option<SpanToken>,
  tap:      TAPOutput,
  //verbose:  i8,
}

impl<S: AsRef<str>> Parser<S> {
  pub fn new(buf: S) -> Parser<S> {
    let tokens = Tokenizer::new(buf);
    //let tokens = DebugTokenizer::new(buf);
    let cur = None;
    let peek = None;
    //let verbose = 0;
    let tap = TAPOutput::default();
    Parser{tokens, cur, peek, tap}
  }

  pub fn set_verbose(&mut self, v: i8) {
    self.tap.verbose = v;
  }

  pub fn set_debug(&mut self) {
    self.set_verbose(3);
  }

  pub fn lbp(&self, tok: &Token) -> RawBp {
    match tok {
      &Token::LDeduct |
      &Token::RDeduct => {
        100
      }
      // FIXME: LWalrus should be R-assoc.
      &Token::LWalrus |
      &Token::RWalrus => {
        110
      }
      &Token::Equal |
      &Token::SlashEq |
      &Token::LQueryEq => {
        120
      }
      // NB: deprecated syntax.
      /*&Token::ColonIdent(_) |*/
      &Token::LParen => {
        400
      }
      &Token::Comma => {
        800
      }
      &Token::Query => {
        // TODO: should be even higher?
        1200
      }
      &Token::DotIdent(_) => {
        1600
      }
      _ => 0
    }
  }

  pub fn restore(&mut self, span: &Span) {
    // FIXME FIXME: this does not restore the bol state.
    self.tokens.seek(span.start);
    self.cur = None;
    self.peek = None;
  }

  pub fn next(&mut self) {
    if let Some(t) = self.peek.take() {
      assert!(self.cur.is_some());
      self.cur = Some(t);
      return;
    }
    self.cur = self.tokens.next();
  }

  pub fn cur(&self) -> SpanToken {
    self.cur.clone().unwrap()
  }

  pub fn maybe_cur_span(&self) -> Option<Span> {
    self.cur.as_ref().map(|cur| cur.span.clone())
  }

  pub fn maybe_cur_tok(&self) -> Option<Token> {
    self.cur.as_ref().map(|cur| cur.tok.clone())
  }

  #[track_caller]
  pub fn cur_span(&self) -> Span {
    let loc = loc();
    match self.cur.as_ref() {
      None => {
        panic!("Parser::cur_span: no cursor (due to reset?): {:?}", loc);
        //println!("DEBUG: Parser::cur_span: no cursor (due to reset?): {:?}", loc);
        //Span{start: usize::max_value(), end: usize::max_value()}
      }
      Some(cur) => cur.span.clone()
    }
  }

  pub fn cur_tok(&self) -> Token {
    self.cur.as_ref().unwrap().tok.clone()
  }

  pub fn peek(&mut self) -> SpanToken {
    assert!(self.cur.is_some());
    if self.peek.is_none() {
      self.peek = self.tokens.next();
    }
    self.peek.clone().unwrap()
  }

  pub fn pos(&self) -> Span {
    match self.cur.as_ref() {
      None => self.tokens._pos(),
      Some(cur) => cur.span.clone()
    }
  }

  pub fn maybe_spaces_deprecated(&mut self) {
    self.next();
    let mut cur = self.cur();
    loop {
      match &cur.tok {
        // FIXME: these cases are probably wrong!
        // esp. w.r.t. multi-line expression blocks, which are indent-sensitive.
        &Token::Space |
        &Token::NL |
        &Token::CR |
        &Token::Comment(_) => {}
        &Token::_Eof => {
          break;
        }
        _ => {
          self.restore(&cur.span);
          break;
        }
      }
      self.next();
      cur = self.cur();
    }
  }

  pub fn mod_(&mut self) -> Result<Mod, ParseSpanError> {
    let mut body = Vec::new();
    loop {
      _debugln!(self, "DEBUG: Parser::mod_: stm...");
      match self.stm(StmCtx::default())? {
        None => break,
        Some((stm, _)) => {
          body.push(Box::new(stm));
        }
      }
    }
    let start = 0;
    let end = self.maybe_cur_span().map(|span| span.end).unwrap_or(0);
    let span = Span{start, end};
    Ok(Mod{span, body})
  }

  pub fn stm(&mut self, ctx: StmCtx) -> Result<Option<(Stm, StmCtx)>, ParseSpanError> {
    let stm_ctx = self._stm(ctx, StmPrefix::default())?;
    loop {
      self.next();
      let cur = self.cur();
      match cur.tok {
        Token::Space => {}
        Token::NL |
        Token::CR |
        Token::Comment(_) |
        Token::_Eof => {
          break;
        }
        _ => {
          self.restore(&cur.span);
          break;
        }
      }
    }
    Ok(stm_ctx)
  }

  pub fn term(&mut self, ctx: TermCtx) -> Result<Term, ParseSpanError> {
    self.maybe_term_spaces(ctx.indent)?;
    self.next();
    let mut cur = self.cur();
    match &cur.tok {
      &Token::Indent(0) => {
        self.next();
        cur = self.cur();
      }
      _ => {}
    };
    let mut lterm = self.term_nud(ctx.indent, cur.clone())?;
    _debugln!(self, "DEBUG: Parser::term: nud: tok={:?} lterm={:?}", &cur.tok, &lterm);
    loop {
      match self.maybe_term_spaces(ctx.indent) {
        Ok(_) => {}
        Err(_) => {
          // FIXME FIXME: right restore point?
          //self.restore(&cur.span);
          break;
        }
      }
      self.next();
      let mut next = self.cur();
      match &next.tok {
        &Token::Indent(0) => {
          self.next();
          next = self.cur();
        }
        _ => {}
      };
      if ctx.bp >= self.lbp(&next.tok) {
        _debugln!(self, "DEBUG: Parser::term: led: break");
        /*// FIXME: technically this restores to after the maybe spaces.
        //self.restore(&rest.span);
        self.restore(&cur.span);*/
        self.restore(&next.span);
        break;
      }
      cur = next;
      lterm = self.term_led(ctx.indent, lterm, cur.clone())?;
      _debugln!(self, "DEBUG: Parser::term: led: tok={:?} lterm={:?}", &cur.tok, &lterm);
    }
    Ok(lterm)
  }

  pub fn _stm(&mut self, ctx: StmCtx, prefix: StmPrefix) -> Result<Option<(Stm, StmCtx)>, ParseSpanError> {
    _debugln!(self, "DEBUG: Parser::stm: ctx={:?} cur? span={:?} tok={:?}", ctx, self.maybe_cur_span(), self.maybe_cur_tok());
    let mut this_ctx = ctx;
    loop {
      self.next();
      let cur = self.cur();
      match cur.tok {
        Token::_Eof => {
          return Ok(None);
        }
        Token::Indent(indent) => {
          match this_ctx.indent {
            StmIndent::Gt(this_indent) => {
              if indent <= this_indent {
                /*self.next();
                let peek = self.cur();
                self.restore(&peek.span);*/
                let peek = self.peek();
                match &peek.tok {
                  &Token::_Eof |
                  &Token::NL |
                  &Token::CR |
                  &Token::Comment(_) => {}
                  _ => {
                    match ctx.indent {
                      StmIndent::Gt(ctx_indent) => {
                        if indent <= ctx_indent {
                          return Ok(None);
                        }
                      }
                      _ => {}
                    }
                    _debugln!(self, "DEBUG: Parser::stm: indent: unexpected tok={:?}", &peek.tok);
                    return Err((cur.span, ParseError::Indent).into());
                  }
                }
              } else {
                this_ctx.indent = StmIndent::Eq(indent);
              }
            }
            StmIndent::Eq(this_indent) => {
              if indent != this_indent {
                /*self.next();
                let peek = self.cur();
                self.restore(&peek.span);*/
                let peek = self.peek();
                match &peek.tok {
                  &Token::_Eof |
                  &Token::NL |
                  &Token::CR |
                  &Token::Comment(_) => {}
                  _ => {
                    match ctx.indent {
                      StmIndent::Gt(ctx_indent) => {
                        if indent <= ctx_indent {
                          return Ok(None);
                        }
                      }
                      _ => {}
                    }
                    if indent < this_indent {
                      return Ok(None);
                    }
                    _debugln!(self, "DEBUG: Parser::stm: indent: unexpected tok={:?}", &peek.tok);
                    _debugln!(self, "DEBUG: Parser::stm: indent:   indent={:?}", indent);
                    _debugln!(self, "DEBUG: Parser::stm: indent:   Eq this indent={:?}", this_indent);
                    return Err((cur.span, ParseError::Indent).into());
                  }
                }
              }
            }
          }
        }
        Token::NL |
        Token::CR |
        Token::Comment(_) => {}
        _ => {
          self.restore(&cur.span);
          break;
        }
      }
    }
    _debugln!(self, "DEBUG: Parser::stm:   this ctx={:?}", this_ctx);
    if this_ctx.indent.is_gt() {
      return Err((self.cur_span(), ParseError::Indent).into());
    }
    self.next();
    let cur = self.cur();
    let cur = match &cur.tok {
      &Token::Indent(0) => {
        self.next();
        self.cur()
      }
      _ => cur
    };
    // TODO: allow `rule` prefix before def-like stm.
    match &cur.tok {
      &Token::Global => {
        _debugln!(self, "DEBUG: Parser::stm: global: tok={:?}", &cur.tok);
        let start = cur.span.clone();
        // FIXME: spaces are required here.
        self.maybe_spaces_deprecated();
        let term = self.term(this_ctx.term())?;
        let ident = match term {
          Term::Ident(_, ident) => ident,
          _ => {
            return Err((self.cur_span(), ParseError::ExpectedIdent).into());
          }
        };
        // FIXME: check that this really is an ident.
        let span = cur.span.hull(self.pos());
        return Ok(Some((Stm::Global(span, ident.into()), this_ctx)));
        /*
        // FIXME: multiple prefixes, support certain orders.
        return Err((self.cur_span(), ParseError::Unimpl(cur.tok.clone())).into());
        */
      }
      &Token::Nonlocal => {
        _debugln!(self, "DEBUG: Parser::stm: nonlocal: tok={:?}", &cur.tok);
        let start = cur.span.clone();
        // FIXME: spaces are required here.
        self.maybe_spaces_deprecated();
        // TODO: optional syntax extension for debruijn levels or indices.
        let mut static_scope = None;
        let save_pos = self.pos();
        self.next();
        let cur = self.cur();
        match &cur.tok {
          &Token::LBrack => {
            let term = self.term(this_ctx.term())?;
            let static_scope_: i16 = match term {
              Term::IntLit(_, s) => {
                match s.as_raw_str().parse() {
                  Err(_) => {
                    return Err((self.cur_span(), ParseError::InvalidIntLit).into());
                  }
                  Ok(v) => v
                }
              }
              _ => {
                return Err((self.cur_span(), ParseError::ExpectedIntLit).into());
              }
            };
            static_scope = Some(static_scope_);
            self.next();
            let cur = self.cur();
            match &cur.tok {
              &Token::RBrack => {}
              _ => {
                return Err((self.cur_span(), ParseError::Expected(Token::RBrack)).into());
              }
            }
            self.maybe_spaces_deprecated();
          }
          _ => {
            self.restore(&save_pos);
          }
        }
        let term = self.term(this_ctx.term())?;
        let ident = match term {
          Term::Ident(_, ident) => ident,
          _ => {
            return Err((self.cur_span(), ParseError::ExpectedIdent).into());
          }
        };
        // FIXME: check that this really is an ident.
        let span = cur.span.hull(self.pos());
        return Ok(Some((Stm::Nonlocal(span, static_scope, ident.into()), this_ctx)));
      }
      &Token::Rule => {
        match prefix {
          StmPrefix::_Nil => {
            self.maybe_spaces_deprecated();
            self.tokens.flag |= TokenizerFlag_::PYTHIA;
            return self._stm(this_ctx, StmPrefix::Rule);
          }
          StmPrefix::Rule => {
            return Err((cur.span, ParseError::Unexpected(cur.tok.clone())).into());
          }
        }
      }
      _ => {}
    }
    match &cur.tok {
      &Token::Elif |
      &Token::Else |
      &Token::Except |
      &Token::Finally |
      &Token::Try => {
        // FIXME: deprecated this compound-block stm parsing path;
        // see the If parsing block below.
        _debugln!(self, "DEBUG: Parser::stm: compound block tok={:?}", &cur.tok);
        let mut this_ctx = StmCtx{
          _stage: Some(StmStage::from(&cur.tok)),
          indent: this_ctx.indent.eq_to_gt(),
        };
        let _stm = match self.stm(this_ctx)? {
          None => {
            // FIXME: this is a fail.
            return Err((cur.span, ParseError::EmptyBlock).into());
          }
          Some((_, stm_ctx)) => {
            assert!(stm_ctx.indent.is_eq());
            this_ctx.indent = stm_ctx.indent;
          }
        };
        // FIXME: stage transitions for compound statements.
        loop {
          match self.stm(this_ctx)? {
            None => {
              break;
            }
            Some((_stm, _)) => {
              // TODO
              unimplemented!();
            }
          }
        }
      }
      _ => {}
    }
    match &cur.tok {
      &Token::None |
      &Token::True |
      &Token::False |
      &Token::IntLit(_) |
      &Token::AtomLit(..) |
      &Token::PlaceIdent(_) |
      &Token::Ident(_) |
      &Token::DotIdent(_) => {
        _debugln!(self, "DEBUG: Parser::stm: just terms: tok={:?}", &cur.tok);
        // FIXME: this_ctx.
        self.restore(&cur.span);
        _debugln!(self, "DEBUG: Parser::stm:   this_ctx={:?} ctx={:?}", this_ctx, ctx);
        _debugln!(self, "DEBUG: Parser::stm:   pos={:?} cur={:?} (before)", self.pos(), &self.cur);
        let t = self.term(this_ctx.term())?;
        //_debugln!(self, "DEBUG: Parser::stm:   term={:?}", &t);
        //_debugln!(self, "DEBUG: Parser::stm:   cur={:?} (after)", &self.cur);
        let span = cur.span.hull(self.pos());
        return Ok(Some((Stm::Just(span, t.into()), this_ctx)));
      }
      &Token::TTTick => {
        _debugln!(self, "DEBUG: Parser::stm: end block quote: tok={:?}", &cur.tok);
        self.tokens.flag |= TokenizerFlag_::PYTHIA;
        return Ok(Some((Stm::_EndQuote(cur.span.clone()), this_ctx)));
      }
      &Token::Pass => {
        _debugln!(self, "DEBUG: Parser::stm: pass: tok={:?}", &cur.tok);
        return Ok(Some((Stm::Pass(cur.span.clone()), this_ctx)));
      }
      &Token::If => {
        let org_tok = cur.tok.clone();
        _debugln!(self, "DEBUG: Parser::stm: if: tok={:?}", &cur.tok);
        let start = cur.span.clone();
        // FIXME: spaces are required here.
        self.maybe_spaces_deprecated();
        let cond = self.term(this_ctx.term())?;
        // FIXME: no spaces should occur here.
        self.maybe_spaces_deprecated();
        self.next();
        let cur = self.cur();
        match &cur.tok {
          &Token::Colon => {
          }
          _ => return Err((self.cur_span(), ParseError::Expected(Token::Colon)).into())
        }
        // TODO
        let mut inner_ctx = StmCtx{
          _stage: None,
          indent: this_ctx.indent.eq_to_gt(),
        };
        let mut body = Vec::new();
        let stm = match self.stm(inner_ctx)? {
          None => {
            return Err((self.cur_span(), ParseError::ExpectedStm).into());
          }
          Some((stm, stm_ctx)) => {
            assert!(stm_ctx.indent.is_eq());
            inner_ctx.indent = stm_ctx.indent;
            stm
          }
        };
        body.push(stm.into());
        loop {
          let stm = match self.stm(inner_ctx)? {
            None => {
              break;
            }
            Some((stm, _)) => stm
          };
          body.push(stm.into());
        }
        let mut cases = vec![(cond.into(), body)];
        //return Err((self.cur_span(), ParseError::Unimpl_).into());
        loop {
          let save_pos = self.pos();
          self.next();
          let cur = self.cur();
          let cur = match &cur.tok {
            &Token::Indent(indent) => {
              match this_ctx.indent {
                StmIndent::Eq(this_ctx_indent) => {
                  if indent > this_ctx_indent {
                    return Err((self.cur_span(), ParseError::Indent).into());
                  } else if indent < this_ctx_indent {
                    self.restore(&save_pos);
                    let span = start.hull(save_pos);
                    return Ok(Some((Stm::If(span, cases, None), this_ctx)));
                  }
                }
                _ => panic!("bug")
              }
              self.next();
              self.cur()
            }
            _ => cur
          };
          let cond = match &cur.tok {
            // FIXME: this explicit Eof case is a bit kludgy.
            &Token::_Eof => {
              let span = start.hull(save_pos);
              return Ok(Some((Stm::If(span, cases, None), this_ctx)));
            }
            &Token::Elif => {
              // FIXME: spaces are required here.
              self.maybe_spaces_deprecated();
              let cond = self.term(this_ctx.term())?;
              Some(cond)
            }
            &Token::Else => {
              None
            }
            _ => return Err((self.cur_span(), ParseError::Expected(Token::Else)).into())
          };
          self.next();
          let cur = self.cur();
          match &cur.tok {
            &Token::Colon => {
            }
            _ => return Err((self.cur_span(), ParseError::Expected(Token::Colon)).into())
          }
          let mut inner_ctx = StmCtx{
            _stage: None,
            indent: this_ctx.indent.eq_to_gt(),
          };
          let mut body = Vec::new();
          let stm = match self.stm(inner_ctx)? {
            None => {
              return Err((self.cur_span(), ParseError::ExpectedStm).into());
            }
            Some((stm, stm_ctx)) => {
              assert!(stm_ctx.indent.is_eq());
              inner_ctx.indent = stm_ctx.indent;
              stm
            }
          };
          body.push(stm.into());
          loop {
            let stm = match self.stm(inner_ctx)? {
              None => {
                break;
              }
              Some((stm, _)) => stm
            };
            body.push(stm.into());
          }
          if let Some(cond) = cond {
            cases.push((cond.into(), body));
          } else {
            let span = start.hull(self.pos());
            return Ok(Some((Stm::If(span, cases, Some(body)), this_ctx)));
          }
        }
      }
      &Token::With => {
        _debugln!(self, "DEBUG: Parser::stm: with ctx: tok={:?}", &cur.tok);
        let start = cur.span.clone();
        self.maybe_spaces_deprecated();
        let head = self.term(this_ctx.term())?;
        match &head {
          &Term::Ident(..) => {
          }
          &Term::Apply(..) => {
          }
          _ => return Err((self.cur_span(), ParseError::ExpectedIdent).into())
        }
        self.maybe_spaces_deprecated();
        self.next();
        let cur = self.cur();
        match &cur.tok {
          &Token::Colon => {
          }
          _ => return Err((self.cur_span(), ParseError::Expected(Token::Colon)).into())
        }
        // TODO
        let mut body = Vec::new();
        let mut inner_ctx = StmCtx{
          _stage: None,
          indent: this_ctx.indent.eq_to_gt(),
        };
        let stm = match self.stm(inner_ctx)? {
          None => {
            return Err((self.cur_span(), ParseError::ExpectedStm).into());
          }
          Some((stm, stm_ctx)) => {
            assert!(stm_ctx.indent.is_eq());
            inner_ctx.indent = stm_ctx.indent;
            stm
          }
        };
        body.push(stm.into());
        loop {
          let stm = match self.stm(inner_ctx)? {
            None => {
              break;
            }
            Some((stm, _)) => stm
          };
          body.push(stm.into());
        }
        let span = start.hull(self.pos());
        _debugln!(self, "DEBUG: Parser::stm: ok: with");
        return Ok(Some((Stm::With(span, head.into(), body), this_ctx)));
      }
      &Token::Defmatch |
      &Token::Defproc |
      &Token::Def => {
        let org_tok = cur.tok.clone();
        _debugln!(self, "DEBUG: Parser::stm: defmatch/defproc: tok={:?}", &cur.tok);
        let start = cur.span.clone();
        // FIXME: spaces are required here.
        self.maybe_spaces_deprecated();
        // FIXME: failure cases (?).
        let mut head = None;
        let mut params = Vec::new();
        match self.term(this_ctx.term())? {
          Term::Ident(_, s) => {
            head = Some(s);
          }
          Term::Apply(_, tup) => {
            match &*tup[0] {
              &Term::Ident(_, ref s) => {
                head = Some(s.clone());
              }
              _ => {}
            }
            for elt in tup[1 .. ].iter() {
              match &**elt {
                &Term::Ident(_, ref s) => {
                  params.push(Some(s.clone()));
                }
                _ => {
                  // FIXME
                  params.push(None);
                }
              }
            }
          }
          _ => return Err((self.cur_span(), ParseError::ExpectedIdent).into())
        }
        let head = head.unwrap();
        //let head = head.ok_or_else(|| (cur.span, ParseError::Unimpl(cur.tok.clone())).into())?;
        self.maybe_spaces_deprecated();
        self.next();
        let cur = self.cur();
        match &cur.tok {
          &Token::Colon => {
          }
          _ => return Err((self.cur_span(), ParseError::Expected(Token::Colon)).into())
        }
        // TODO
        let mut inner_ctx = StmCtx{
          _stage: None,
          indent: this_ctx.indent.eq_to_gt(),
        };
        let mut body = Vec::new();
        let stm = match self.stm(inner_ctx)? {
          None => {
            return Err((self.cur_span(), ParseError::ExpectedStm).into());
          }
          Some((stm, stm_ctx)) => {
            assert!(stm_ctx.indent.is_eq());
            inner_ctx.indent = stm_ctx.indent;
            stm
          }
        };
        body.push(stm.into());
        loop {
          let stm = match self.stm(inner_ctx)? {
            None => {
              break;
            }
            Some((stm, _)) => stm
          };
          body.push(stm.into());
        }
        let span = start.hull(self.pos());
        match &org_tok {
          &Token::Defmatch => {
            _debugln!(self, "DEBUG: Parser::stm: ok: defmatch");
            self.tokens.flag |= TokenizerFlag_::PYTHIA;
            return Ok(Some((Stm::Defmatch(span, prefix.into_def(), head, params, body), this_ctx)));
          }
          &Token::Defproc => {
            _debugln!(self, "DEBUG: Parser::stm: ok: defproc");
            self.tokens.flag |= TokenizerFlag_::PYTHIA;
            return Ok(Some((Stm::Defproc(span, prefix.into_def(), head, params, body), this_ctx)));
          }
          &Token::Def => {
            _debugln!(self, "DEBUG: Parser::stm: ok: def");
            self.tokens.flag |= TokenizerFlag_::PYTHON;
            return Ok(Some((Stm::Defproc(span, prefix.into_def(), head, params, body), this_ctx)));
          }
          _ => {}
        }
      }
      &Token::TTTickQuote => {
        _debugln!(self, "DEBUG: Parser::stm: block quote: tok={:?}", &cur.tok);
        let start = cur.span.clone();
        let mut inner_ctx = StmCtx{
          _stage: None,
          //indent: this_ctx.indent.eq_to_gt(),
          indent: this_ctx.indent,
        };
        let mut body = Vec::new();
        let stm = match self.stm(inner_ctx)? {
          None => {
            return Err((self.cur_span(), ParseError::ExpectedStm).into());
          }
          Some((stm, stm_ctx)) => {
            assert!(stm_ctx.indent.is_eq());
            inner_ctx.indent = stm_ctx.indent;
            stm
          }
        };
        body.push(stm.into());
        loop {
          let stm = match self.stm(inner_ctx)? {
            None => {
              break;
            }
            Some((stm, _)) => stm
          };
          match &stm {
            &Stm::_EndQuote(_) => {
              let span = start.hull(self.pos());
              _debugln!(self, "DEBUG: Parser::stm: ok: block quote");
              self.tokens.flag |= TokenizerFlag_::PYTHIA;
              return Ok(Some((Stm::Quote(span, (), body), this_ctx)));
            }
            _ => {}
          }
          body.push(stm.into());
        }
      }
      _ => {}
    }
    match &cur.tok {
      &Token::Break |
      &Token::Continue |
      &Token::For |
      &Token::Raise |
      &Token::Return |
      &Token::While |
      &Token::With => {
        _debugln!(self, "DEBUG: Parser::stm: block stm: tok={:?}", &cur.tok);
        // FIXME: this_ctx.
        // TODO
      }
      _ => {}
    }
    _debugln!(self, "DEBUG: Parser::stm: unimpl: cur.span={:?} cur.tok={:?}", cur.span, &cur.tok);
    return Err((cur.span, ParseError::Unimpl(cur.tok.clone())).into());
  }

  pub fn maybe_term_spaces(&mut self, ctx_indent: RawIndent) -> Result<(), ParseSpanError> {
    self.next();
    let mut cur = self.cur();
    let mut nl = false;
    loop {
      if nl {
        nl = false;
        if ctx_indent > 0 {
          match &cur.tok {
            &Token::Indent(indent) => {
              if ctx_indent > indent {
                // FIXME: okay to accept if the indent is followed by NL/CR;
                // but maybe we want to be super strict about indent?
                return Err((cur.span, ParseError::Indent).into());
              }
            }
            _ => {}
          }
          self.next();
          cur = self.cur();
          continue;
        }
      }
      match &cur.tok {
        &Token::Space => {}
        &Token::NL |
        &Token::CR |
        &Token::Comment(_) => {
          nl = true;
        }
        &Token::Indent(0) => {}
        &Token::Indent(_) => {
          // FIXME: maybe want a different ParseError here.
          return Err((cur.span, ParseError::Indent).into());
        }
        &Token::_Eof => {
          break;
        }
        _ => {
          self.restore(&cur.span);
          break;
        }
      }
      self.next();
      cur = self.cur();
    }
    Ok(())
  }

  pub fn term_nud(&mut self, ctx_indent: RawIndent, cur: SpanToken) -> Result<Term, ParseSpanError> {
    // TODO TODO
    let this_ctx = TermCtx{
      indent: ctx_indent,
      bp: 0,
      //bp: self.lbp(&tok),
    };
    match &cur.tok {
      &Token::Indent(0) => {
        panic!("bug: Parser::term_nud: unexpected Indent(0)");
      }
      &Token::Indent(_) |
      &Token::Space |
      &Token::NL |
      &Token::CR |
      &Token::Comment(_) => {
        panic!("bug: Parser::term_nud: tok={:?}", &cur.tok);
      }
      &Token::_Eof => {
        return Err((cur.span, ParseError::Eof).into());
      }
      &Token::None => {
        return Ok(Term::NoneLit(cur.span, "None".into()));
      }
      &Token::True => {
        return Ok(Term::BoolLit(cur.span, "True".into()));
      }
      &Token::False => {
        return Ok(Term::BoolLit(cur.span, "False".into()));
      }
      &Token::IntLit(ref s) => {
        //_debugln!(self, "DEBUG: Parser::term_led: int lit: tok={:?}", &cur.tok);
        return Ok(Term::IntLit(cur.span, s.clone()));
      }
      &Token::AtomLit(ref s, _) => {
        //return Ok(Term::StrLit(cur.span, s.clone()));
        return Ok(Term::AtomLit(cur.span, s.clone()));
      }
      &Token::PlaceIdent(ref s) => {
        // FIXME: place ident term?
        return Ok(Term::Ident(cur.span, s.clone()));
        //return Ok(Term::PlaceIdent(cur.span, ()));
      }
      &Token::Ident(ref s) => {
        // FIXME
        return Ok(Term::Ident(cur.span, s.clone()));
      }
      &Token::DashSlash => {
        self.maybe_term_spaces(ctx_indent)?;
        let rterm = self.term(this_ctx)?;
        return Ok(Term::Neg(cur.span, rterm.into()));
      }
      &Token::LParen => {
        let start = cur.span.clone();
        self.maybe_term_spaces(ctx_indent)?;
        let rterm = self.term(this_ctx)?;
        self.maybe_term_spaces(ctx_indent)?;
        self.next();
        let cur = self.cur();
        match &cur.tok {
          &Token::RParen => {}
          _ => {
            return Err((cur.span, ParseError::_Bot).into());
          }
        }
        let span = start.hull(self.pos());
        return Ok(Term::Group(span, rterm.into()));
      }
      &Token::LBrack => {
        let start = cur.span.clone();
        //let mut tup = vec![];
        self.next();
        let cur = self.cur();
        match &cur.tok {
          &Token::RBrack => {
            let span = start.hull(self.pos());
            return Ok(Term::ListLit(span, Vec::new()));
          }
          _ => {}
        }
        self.restore(&cur.span);
        let term = self.term(this_ctx)?;
        match term {
          Term::Bunch(_, tup) => {
            self.next();
            let cur = self.cur();
            match &cur.tok {
              &Token::RBrack => {}
              _ => {
                return Err((cur.span, ParseError::Expected(Token::RBrack)).into());
              }
            }
            let span = start.hull(self.pos());
            return Ok(Term::ListLit(span, tup));
          }
          _ => {
            return Err((cur.span, ParseError::ExpectedBunch).into());
          }
        }
      }
      _ => {}
    }
    _debugln!(self, "DEBUG: Parser::term_nud: unimpl: span={:?} tok={:?}", cur.span, &cur.tok);
    unimplemented!();
  }

  pub fn term_led(&mut self, ctx_indent: RawIndent, mut lterm: Term, cur: SpanToken) -> Result<Term, ParseSpanError> {
    let this_ctx = TermCtx{
      indent: ctx_indent,
      bp: self.lbp(&cur.tok),
    };
    match &cur.tok {
      &Token::Indent(_) |
      &Token::Space |
      &Token::NL |
      &Token::CR |
      &Token::Comment(_) => {
        panic!("bug");
      }
      &Token::_Eof => {
        return Err((cur.span, ParseError::Eof).into());
      }
      // NB: deprecated syntax.
      /*&Token::ColonIdent(ref s) => {
        let start = lterm.span();
        let span = start.hull(self.pos());
        return Ok(Term::QualIdent(span, lterm.into(), s.clone()));
      }*/
      &Token::Query => {
        // FIXME: if lterm is an apply tuple, might convert this into
        // an apply-query term.
        let span = lterm.span();
        return Ok(Term::Query(span, lterm.into()));
      }
      &Token::Equal => {
        //_traceln!(self, "DEBUG: Parser::term_led: Equal: tok={:?}", &cur.tok);
        let start = lterm.span();
        self.maybe_term_spaces(ctx_indent)?;
        let rterm = self.term(this_ctx)?;
        let span = start.hull(self.pos());
        return Ok(Term::Equal(span, lterm.into(), rterm.into()));
      }
      &Token::SlashEq => {
        //_traceln!(self, "DEBUG: Parser::term_led: SlashEq: tok={:?}", &cur.tok);
        let start = lterm.span();
        self.maybe_term_spaces(ctx_indent)?;
        let rterm = self.term(this_ctx)?;
        let span = start.hull(self.pos());
        return Ok(Term::NEqual(span, lterm.into(), rterm.into()));
      }
      &Token::LQueryEq => {
        let start = lterm.span();
        self.maybe_term_spaces(ctx_indent)?;
        let rterm = self.term(this_ctx)?;
        let span = start.hull(self.pos());
        return Ok(Term::QEqual(span, lterm.into(), rterm.into()));
      }
      &Token::LWalrus => {
        let mut this_ctx = this_ctx;
        this_ctx.bp -= 1;
        let start = lterm.span();
        self.maybe_term_spaces(ctx_indent)?;
        let mut rterm = self.term(this_ctx)?;
        match &mut rterm {
          &mut Term::Apply(_, ref mut tup) => {
            let tup = replace(tup, Vec::new());
            let span = start.hull(self.pos());
            return Ok(Term::ApplyBindL(span, lterm.into(), tup));
          }
          _ => {}
        }
        let span = start.hull(self.pos());
        return Ok(Term::BindL(span, lterm.into(), rterm.into()));
      }
      &Token::RWalrus => {
        let start = lterm.span();
        self.maybe_term_spaces(ctx_indent)?;
        let rterm = self.term(this_ctx)?;
        match &mut lterm {
          &mut Term::Apply(_, ref mut tup) => {
            let tup = replace(tup, Vec::new());
            let span = start.hull(self.pos());
            return Ok(Term::ApplyBindR(span, tup, rterm.into()));
          }
          _ => {}
        }
        let span = start.hull(self.pos());
        return Ok(Term::BindR(span, lterm.into(), rterm.into()));
      }
      &Token::Comma => {
        let start = lterm.span();
        self.maybe_term_spaces(ctx_indent)?;
        let rterm = self.term(this_ctx)?;
        match &mut lterm {
          &mut Term::Bunch(ref mut span, ref mut tup) => {
            tup.push(rterm.into());
            *span = start.hull(self.pos());
            return Ok(lterm);
          }
          _ => {}
        }
        let span = start.hull(self.pos());
        return Ok(Term::Bunch(span, vec![lterm.into(), rterm.into()]));
      }
      &Token::DotIdent(_) => {
        let start = lterm.span();
        self.maybe_term_spaces(ctx_indent)?;
        self.next();
        let cur = self.cur();
        match &cur.tok {
          &Token::LParen => {
            let mut this_ctx = this_ctx;
            this_ctx.bp = 0;
            // FIXME: rtup needs the dotident (above) as head.
            let mut rtup = Vec::new();
            let mut rterm = self.term(this_ctx)?;
            match &mut rterm {
              &mut Term::Bunch(_, ref mut tup_) => {
                let tup = replace(tup_, Vec::new());
                rtup.extend(tup);
                self.maybe_term_spaces(ctx_indent)?;
                self.next();
                let cur = self.cur();
                match &cur.tok {
                  &Token::RParen => {}
                  _ => {
                    // FIXME: err.
                  }
                }
                let span = start.hull(self.pos());
                return Ok(Term::Effect(span, lterm.into(), rtup));
              }
              _ => {}
            }
            rtup.push(rterm.into());
            self.maybe_term_spaces(ctx_indent)?;
            self.next();
            let cur = self.cur();
            match &cur.tok {
              &Token::RParen => {}
              _ => {
                // FIXME: err.
              }
            }
            let span = start.hull(self.pos());
            return Ok(Term::Effect(span, lterm.into(), rtup));
          }
          _ => {}
        }
        unimplemented!();
      }
      &Token::LParen => {
        let mut this_ctx = this_ctx;
        this_ctx.bp = 0;
        let start = lterm.span();
        let mut tup: Vec<TermRef> = vec![lterm.into()];
        loop {
          self.maybe_term_spaces(ctx_indent)?;
          self.next();
          let cur = self.cur();
          match &cur.tok {
            &Token::RParen => {
              let span = start.hull(self.pos());
              if tup.len() == 2 {
                match *tup.pop().unwrap() {
                  Term::Bunch(_, tup_) => {
                    tup.extend(tup_);
                    return Ok(Term::Apply(span, tup));
                  }
                  t => {
                    tup.push(Box::new(t));
                  }
                }
              }
              return Ok(Term::Apply(span, tup));
            }
            _ => {}
          }
          self.restore(&cur.span);
          let rterm = self.term(this_ctx)?;
          tup.push(rterm.into());
          self.maybe_term_spaces(ctx_indent)?;
          self.next();
          let cur = self.cur();
          match &cur.tok {
            &Token::Comma => {}
            &Token::RParen => {
              let span = start.hull(self.pos());
              if tup.len() == 2 {
                match *tup.pop().unwrap() {
                  Term::Bunch(_, tup_) => {
                    tup.extend(tup_);
                    return Ok(Term::Apply(span, tup));
                  }
                  t => {
                    tup.push(Box::new(t));
                  }
                }
              }
              return Ok(Term::Apply(span, tup));
            }
            _ => {
              return Err((cur.span, ParseError::Unexpected(cur.tok)).into());
            }
          }
        }
        #[allow(unreachable_code)]
        { unreachable!(); }
        /*let span = start.hull(self.pos());
        if tup.len() == 2 {
          match *tup.pop().unwrap() {
            Term::Bunch(_, tup_) => {
              tup.extend(tup_);
              return Ok(Term::Apply(span, tup));
            }
            t => {
              tup.push(Box::new(t));
            }
          }
        }
        return Ok(Term::Apply(span, tup));*/
      }
      &Token::LColonParen => {
        let mut this_ctx = this_ctx;
        this_ctx.bp = 0;
        let start = lterm.span();
        self.maybe_term_spaces(ctx_indent)?;
        let rterm = self.term(this_ctx)?;
        self.maybe_term_spaces(ctx_indent)?;
        match &cur.tok {
          &Token::RParen => {}
          _ => {
            return Err((cur.span, ParseError::_Bot).into());
          }
        }
        let span = start.hull(self.pos());
        return Ok(Term::Subst(span, lterm.into(), rterm.into()));
      }
      _ => unimplemented!()
    }
    #[allow(unreachable_code)]
    { unreachable!(); }
  }
}

//#[derive(Default)]
pub struct Printer<S> {
  buf:  S,
  indent: RawIndent,
}

impl<S> Printer<S> {
  pub fn new(buf: S) -> Printer<S> {
    Printer{buf, indent: 0}
  }
}

impl<S: AsRef<str>> Printer<S> {
  pub fn _snippet(&self, span: &Span) -> &str {
    self.buf.as_ref().get(span.clone()).unwrap()
  }

  pub fn pretty_print(&self, mod_: &Mod) {
    for stm in mod_.body.iter() {
      self._pretty_print_stm(stm, 0);
    }
  }

  pub fn _pretty_print_stm(&self, stm: &Stm, level: RawIndent) {
    let indent = match self.indent {
      0 => 4,
      x => x
    };
    for _ in 0 .. level * indent {
      print!(" ");
    }
    match stm {
      &Stm::Just(_, ref term) => {
        self._pretty_print_term(term, level);
        println!();
      }
      &Stm::Pass(_) => {
        println!("pass");
      }
      &Stm::Global(_, ref ident) => {
        println!("global {}", ident);
      }
      &Stm::Nonlocal(_, _, ref ident) => {
        // TODO: static scope (debruijn level/index).
        println!("nonlocal {}", ident);
      }
      &Stm::With(_, ref head, ref body) => {
        print!("with ");
        self._pretty_print_term(head, level);
        println!(":");
        for stm in body.iter() {
          self._pretty_print_stm(stm, level + 1);
        }
      }
      &Stm::Def(_, .., ref body) => {
        println!("def _:");
        for stm in body.iter() {
          self._pretty_print_stm(stm, level + 1);
        }
      }
      &Stm::Defmatch(_, prefix, ref head, ref params, ref body) => {
        match prefix {
          None => {}
          Some(DefPrefix::Rule) => {
            print!("rule ");
          }
        }
        print!("defmatch {head}(");
        for (idx, param) in params.iter().enumerate() {
          if param.is_some() {
            print!("{}", param.as_ref().unwrap());
          } else {
            print!("_");
          }
          if idx + 1 < params.len() {
            print!(", ");
          }
        }
        println!("):");
        for stm in body.iter() {
          self._pretty_print_stm(stm, level + 1);
        }
      }
      &Stm::Defproc(_, prefix, ref head, ref params, ref body) => {
        match prefix {
          None => {}
          Some(DefPrefix::Rule) => {
            print!("rule ");
          }
        }
        print!("defproc {head}(");
        for (idx, param) in params.iter().enumerate() {
          if param.is_some() {
            print!("{}", param.as_ref().unwrap());
          } else {
            print!("_");
          }
          if idx + 1 < params.len() {
            print!(", ");
          }
        }
        println!("):");
        for stm in body.iter() {
          self._pretty_print_stm(stm, level + 1);
        }
      }
      &Stm::Quote(_, .., ref body) => {
        println!("```quote");
        for stm in body.iter() {
          self._pretty_print_stm(stm, level);
        }
        println!("```");
      }
      _ => {
        println!("# <stm>");
      }
    }
  }

  pub fn _pretty_print_term(&self, term: &Term, level: RawIndent) {
    match term {
      &Term::Ident(ref span, ..) => {
        print!("{}~[{:?}]", self._snippet(span), span);
      }
      &Term::QualIdent(ref span, ..) => {
        print!(":{}~[{:?}]", self._snippet(span), span);
      }
      &Term::AtomLit(ref span, ..) => {
        print!("{}~[{:?}]", self._snippet(span), span);
      }
      &Term::NoneLit(ref span, ..) => {
        print!("{}~[{:?}]", self._snippet(span), span);
      }
      &Term::BoolLit(ref span, ..) => {
        print!("{}~[{:?}]", self._snippet(span), span);
      }
      &Term::IntLit(ref span, ..) => {
        print!("{}~[{:?}]", self._snippet(span), span);
      }
      &Term::FloatLit(_, ..) => {
        print!("_");
      }
      &Term::ListLit(_, ref terms) => {
        print!("[");
        for (i, term) in terms.iter().enumerate() {
          self._pretty_print_term(term, level);
          if i + 1 < terms.len() {
            print!(",");
          }
        }
        print!("]");
      }
      &Term::Group(_, ref term) => {
        print!("(");
        self._pretty_print_term(term, level);
        print!(")");
      }
      &Term::Bunch(_, ref terms) => {
        for (i, term) in terms.iter().enumerate() {
          self._pretty_print_term(term, level);
          if i + 1 < terms.len() {
            print!(",");
          }
        }
      }
      &Term::Neg(_, ref term) => {
        print!("-/");
        self._pretty_print_term(term, level);
      }
      &Term::Query(_, ref term) => {
        self._pretty_print_term(term, level);
        print!("?");
      }
      &Term::Equal(_, ref lterm, ref rterm) => {
        self._pretty_print_term(lterm, level);
        print!(" = ");
        self._pretty_print_term(rterm, level);
      }
      &Term::NEqual(_, ref lterm, ref rterm) => {
        self._pretty_print_term(lterm, level);
        print!(" /= ");
        self._pretty_print_term(rterm, level);
      }
      &Term::QEqual(_, ref lterm, ref rterm) => {
        self._pretty_print_term(lterm, level);
        print!(" ?= ");
        self._pretty_print_term(rterm, level);
      }
      &Term::BindL(_, ref lterm, ref rterm) => {
        self._pretty_print_term(lterm, level);
        print!(" := ");
        self._pretty_print_term(rterm, level);
      }
      &Term::BindR(_, ref lterm, ref rterm) => {
        self._pretty_print_term(lterm, level);
        print!(" =: ");
        self._pretty_print_term(rterm, level);
      }
      &Term::Subst(_, ref lterm, ref rterm) => {
        self._pretty_print_term(lterm, level);
        print!(" .( ");
        self._pretty_print_term(rterm, level);
        print!(" )");
      }
      &Term::RebindL(_, ref lterm, ref rterm) => {
        self._pretty_print_term(lterm, level);
        print!(" .= ");
        self._pretty_print_term(rterm, level);
      }
      &Term::RebindR(_, ref lterm, ref rterm) => {
        self._pretty_print_term(lterm, level);
        print!(" =. ");
        self._pretty_print_term(rterm, level);
      }
      &Term::Apply(_, ref tup) => {
        self._pretty_print_term(&tup[0], level);
        print!("(");
        for (i, term) in tup[1 .. ].iter().enumerate() {
          self._pretty_print_term(term, level);
          if i + 2 < tup.len() {
            print!(",");
          }
        }
        print!(")");
      }
      &Term::ApplyBindL(_, ref lterm, ref tup) => {
        self._pretty_print_term(lterm, level);
        print!(" := ");
        self._pretty_print_term(&tup[0], level);
        print!("(");
        for (i, term) in tup[1 .. ].iter().enumerate() {
          self._pretty_print_term(term, level);
          if i + 2 < tup.len() {
            print!(",");
          }
        }
        print!(")");
      }
      &Term::ApplyBindR(_, ref tup, ref rterm) => {
        self._pretty_print_term(&tup[0], level);
        print!("(");
        for (i, term) in tup[1 .. ].iter().enumerate() {
          self._pretty_print_term(term, level);
          if i + 2 < tup.len() {
            print!(",");
          }
        }
        print!(")");
        print!(" =: ");
        self._pretty_print_term(rterm, level);
      }
      &Term::Effect(_, ref lterm, ref rterms) => {
        self._pretty_print_term(lterm, level);
        print!(".__(");
        for (i, term) in rterms.iter().enumerate() {
          self._pretty_print_term(term, level);
          if i + 1 < rterms.len() {
            print!(",");
          }
        }
        print!(")");
      }
      /*_ => {
        print!("_");
      }*/
    }
  }
}
