use crate::algo::{
  BTreeMap, BTreeSet, FxHashMap,
  OptionExt,
  SmolStr,
};
use crate::algo::cell::{RefCell};
use crate::algo::rc::{Arc, Rc};
use crate::algo::str::{SafeStr, safe_ascii};
use crate::panick::{Loc, loc, _debugln, _traceln};
use crate::parse::{
  Printer as DebugPrinter,
  FastParser,
  Span as RawSpan_,
  Mod as RawMod_,
  Stm as RawStm_,
  Term as RawTerm_,
  Ident as RawIdent_,
  Lit as RawLit_,
  DefPrefix as RawDefPrefix_,
};

use paste::{paste};
use serde::{Serialize};
use serde::ser::{Serializer, SerializeStruct};

use std::any::{Any, type_name};
use std::cell::{Cell};
use std::cmp::{Ordering, max, min};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::hash::{Hash, Hasher};
use std::mem::{replace};
use std::panic::{Location};
use std::path::{PathBuf};
use std::str::{FromStr};

pub mod prelude;

pub type RawSNum = u32;
pub type RawLClk = i64;
pub type RawMClk = u64;
pub type RawMAddr = Box<[u64]>;

// FIXME: might be very useful for SNum to be a "tagged pointer":
// - a few low bits signify the "type" or "sort"
// - the high bits are uniquely allocated from the counter

pub const SNUM_TAG_BITS: RawSNum = 8;
pub const SNUM_TAG_MASK: RawSNum = 0xff;

pub const SNUM_UNSORT:      RawSNum = 0;
pub const SNUM_SPAN_SORT:   RawSNum = 1;
pub const SNUM_CODE_SORT:   RawSNum = 2;
pub const SNUM_IDENT_SORT:  RawSNum = 3;
pub const SNUM_CELL_SORT:   RawSNum = 7;
pub const SNUM_LITSTR_SORT: RawSNum = 8;
pub const SNUM_TERM_SORT:   RawSNum = 9;
pub const SNUM_VAL_SORT:    RawSNum = 10;
pub const _SNUM_MAX_SORT:   RawSNum = 10;

#[derive(Clone, Copy, Hash)]
#[repr(transparent)]
pub struct SNum(RawSNum);

impl SNum {
  #[inline]
  pub fn _key(&self) -> RawSNum {
    let key = self.0 >> SNUM_TAG_BITS;
    key
  }

  #[inline]
  pub fn _tag(&self) -> RawSNum {
    let tag = self.0 & SNUM_TAG_MASK;
    tag
  }
}

impl Serialize for SNum {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    let mut state = serializer.serialize_struct("SNum", 2)?;
    state.serialize_field("SNum_key", &self._key())?;
    state.serialize_field("SNum_tag", &self._tag())?;
    state.end()
  }
}

impl Debug for SNum {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    let key = self.0 >> SNUM_TAG_BITS;
    let tag = self.0 & SNUM_TAG_MASK;
    write!(f, "SNum({}.{})", key, tag)
  }
}

impl PartialEq for SNum {
  fn eq(&self, rhs: &SNum) -> bool {
    let lkey = self.0 >> SNUM_TAG_BITS;
    let rkey = rhs.0 >> SNUM_TAG_BITS;
    if lkey != rkey {
      return false;
    }
    let ltag = self.0 & SNUM_TAG_MASK;
    let rtag = rhs.0 & SNUM_TAG_MASK;
    let lrcmp = ltag ^ rtag;
    if lrcmp == 0 {
      return true;
    }
    if lrcmp == ltag || lrcmp == rtag {
      return true;
    }
    panic!("bug: SNum::partial_eq: tag sort mismatch: lkey={} ltag={} rkey={} rtag={}",
        lkey, ltag, rkey, rtag);
  }
}

impl Eq for SNum {}

impl PartialOrd for SNum {
  fn partial_cmp(&self, rhs: &SNum) -> Option<Ordering> {
    Some(self.cmp(rhs))
  }
}

impl Ord for SNum {
  fn cmp(&self, rhs: &SNum) -> Ordering {
    let lkey = self.0 >> SNUM_TAG_BITS;
    let rkey = rhs.0 >> SNUM_TAG_BITS;
    if lkey == rkey {
      if self != rhs {
        panic!("bug");
      } else {
        return Ordering::Equal;
      }
    } else if lkey < rkey {
      return Ordering::Less;
    } else if lkey > rkey {
      return Ordering::Greater;
    } else {
      unreachable!();
    }
  }
}

// Simulation logical (linearizable) clock time.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize)]
#[repr(transparent)]
pub struct LClk(RawLClk);

impl LClk {
  pub fn _into_raw(self) -> RawLClk {
    self.0
  }
}

// "Totally ordered" monotonic (physical) time.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct MClk(RawMClk);

// "Unique" machine address.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct MAddr(RawMAddr);

pub trait Nil {
  fn nil() -> Self where Self: Sized;
  fn is_nil(&self) -> bool;
}

impl Nil for SNum {
  fn nil() -> SNum {
    SNum(0)
  }

  fn is_nil(&self) -> bool {
    (self.0 >> SNUM_TAG_BITS) == 0
  }
}

impl Nil for LClk {
  fn nil() -> LClk {
    LClk(i64::min_value())
  }

  fn is_nil(&self) -> bool {
    self.0 == i64::min_value()
  }
}

impl Nil for MemKntRef {
  fn nil() -> MemKntRef {
    None
  }

  fn is_nil(&self) -> bool {
    self.is_none()
  }
}

pub fn nil<T: Nil>() -> T {
  <T as Nil>::nil()
}

pub trait IntoNil {
  type Nil_;

  fn try_into_nil(self) -> Result<Self::Nil_, ()>;
}

impl IntoNil for Option<SNum> {
  type Nil_ = SNum;

  fn try_into_nil(self) -> Result<SNum, ()> {
    match self {
      None => Ok(nil()),
      Some(x) => if !x.is_nil() {
        Ok(x)
      } else {
        Err(())
      }
    }
  }
}

impl IntoNil for Option<StmCodeCellNum> {
  type Nil_ = StmCodeCellNum;

  fn try_into_nil(self) -> Result<StmCodeCellNum, ()> {
    match self {
      None => Ok(nil()),
      Some(x) => if !x.is_nil() {
        Ok(x)
      } else {
        Err(())
      }
    }
  }
}

pub trait Bot {
  fn bot() -> Self where Self: Sized;
}

#[track_caller]
pub fn bot<T: Bot>() -> T {
  <T as Bot>::bot()
}

pub trait Unimpl {
  fn unimpl() -> Self where Self: Sized;
}

#[track_caller]
pub fn unimpl<T: Unimpl>() -> T {
  <T as Unimpl>::unimpl()
}

pub type SCtr = SNumCtr;

#[derive(Debug)]
pub struct SNumCtr {
  rctr: Cell<RawSNum>,
}

impl Default for SNumCtr {
  fn default() -> SNumCtr {
    SNumCtr{
      rctr: Cell::new(0),
    }
  }
}

impl SNumCtr {
  pub fn _reset(&self, new_x: SNum) {
    let next = new_x._into_raw() >> SNUM_TAG_BITS;
    self.rctr.set(next);
  }

  pub fn _fresh(&self) -> SNum {
    let next = self.rctr.get() + 1;
    self.rctr.set(next);
    // NB: untagged/unsorted SNum.
    let x = (next << SNUM_TAG_BITS);
    SNum(x)
  }

  pub fn _get(&self) -> SNum {
    let cur = self.rctr.get();
    // NB: untagged/unsorted SNum.
    let x = (cur << SNUM_TAG_BITS);
    SNum(x)
  }
}

#[derive(Debug)]
pub struct LClkCtr {
  rctr: Cell<RawLClk>,
}

impl Default for LClkCtr {
  fn default() -> LClkCtr {
    LClkCtr{
      rctr: Cell::new(0),
    }
  }
}

impl LClkCtr {
  pub fn _fresh_clock(&self) -> LClk {
    let next = self.rctr.get() + 1;
    self.rctr.set(next);
    LClk(next)
  }

  pub fn _get_clock(&self) -> LClk {
    LClk(self.rctr.get())
  }

  pub fn _next_clock(&self) -> LClk {
    let next = self.rctr.get() + 1;
    LClk(next)
  }
}

#[derive(Clone, Copy, Debug)]
pub struct LClkRange {
  // This is a half-open interval [lb, ub).
  pub lb: LClk,
  pub ub: LClk,
}

#[derive(Default)]
pub struct LClkInvalidSet {
  inner: BTreeMap<LClk, LClk>,
}

impl LClkInvalidSet {
  pub fn _insert(&mut self, lb: LClk, ub: LClk) -> Result<(), ()> {
    if lb >= ub {
      return Err(());
    }
    match (self._find(lb), self._find(ub)) {
      (Some(lrg), Some(urg)) => {
        if lrg.lb != urg.lb {
          let _ = self.inner.remove(&urg.lb);
          self.inner.insert(lrg.lb, urg.ub);
        }
      }
      (Some(lrg), None) => {
        /*assert!(lrg.ub < ub);*/
        self.inner.insert(lrg.lb, ub);
      }
      (None, Some(urg)) => {
        /*assert!(lb < urg.ub);*/
        let _ = self.inner.remove(&urg.lb);
        self.inner.insert(lb, urg.ub);
      }
      (None, None) => {
        self.inner.insert(lb, ub);
      }
    }
    Ok(())
  }

  pub fn _contains(&self, v: LClk) -> bool {
    self._find(v).is_some()
  }

  pub fn _find(&self, v: LClk) -> Option<LClkRange> {
    match self.inner.range( ..= v).next_back() {
      Some((&lb, &ub)) => {
        if lb <= v && v < ub {
          Some(LClkRange{lb, ub})
        } else {
          None
        }
      }
      None => None
    }
  }
}

impl Default for SNum {
  fn default() -> SNum {
    SNum(0)
  }
}

impl SNum {
  pub fn _into_raw(self) -> RawSNum {
    self.0
  }

  pub fn into_fun(self) -> FunNum {
    // FIXME: tag sort check.
    FunNum(self.0)
  }

  pub fn into_obj_cls(self) -> ObjClsNum {
    // FIXME: tag sort check.
    ObjClsNum(self.0)
  }

  pub fn into_obj_val(self) -> ObjValNum {
    // FIXME: tag sort check.
    ObjValNum(self.0)
  }

  pub fn into_cell(self) -> CellNum {
    // FIXME: tag sort check.
    CellNum(self.0)
  }

  pub fn into_atom(self) -> AtomNum {
    // FIXME: tag sort check.
    AtomNum(self.0)
  }

  pub fn into_ident(self) -> IdentNum {
    // FIXME: tag sort check.
    IdentNum(self.0)
  }

  pub fn into_lit_str(self) -> LitStrNum {
    // FIXME: tag sort check.
    LitStrNum(self.0)
  }

  pub fn into_term(self) -> TermNum {
    // FIXME: tag sort check.
    TermNum(self.0)
  }

  pub fn into_span(self) -> SpanNum {
    // FIXME: tag sort check.
    SpanNum(self.0)
  }

  pub fn into_mod_code(self) -> ModCodeNum {
    // FIXME: tag sort check.
    ModCodeNum(self.0)
  }

  pub fn into_stm_code(self) -> StmCodeNum {
    // FIXME: tag sort check.
    StmCodeNum(self.0)
  }

  pub fn into_term_code(self) -> TermCodeNum {
    // FIXME: tag sort check.
    TermCodeNum(self.0)
  }
}

// [Interp-API]
//
// A tuple-cell, or _cell_, is one element in a linked-list repr of a tuple.
#[derive(Clone, Debug)]
pub struct Cell_ {
  pub dptr: SNum,
  pub next: Cell<CellNum>,
  pub prev: Cell<CellNum>,
}

impl Cell_ {
  pub fn _is_head(&self) -> bool {
    self.prev.is_nil()
  }
}

// [Interp-API]
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct AtomTerm_ {
  raw:  SafeStr,
}

// [Interp-API]
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct IdentTerm_ {
  raw:  SafeStr,
}

// [Interp-API]
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct StrTerm_ {
  // TODO: SafeStr is backed by SmolStr, which implements heap
  // allocated str via Arc; but we might also want a Rc version.
  inner: SafeStr,
}

// [Interp-API]
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct IntTerm_ {
  // TODO: arbitrary precision (gmp?).
  inner: i64,
}

// [Interp-API]
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct BoolTerm_ {
  inner: bool,
}

// [Interp-API]
//
// Literal terms basically implement:
//
// () | true | false | Arc<i64> | Arc<str>
//
// (The Arc<str> is an impl detail due to SmolStr.)

pub enum UnpackedLitTerm_ {
  None,
  True,
  False,
  Int(Arc<i64>),
  Str(Arc<SafeStr>),
}

const LIT_TERM_TAG_BITS: usize = 3;
const LIT_TERM_TAG_MASK: usize = 7;

const LIT_TERM_TAG_NONE: usize = 0;
const LIT_TERM_TAG_TRUE: usize = 1;
const LIT_TERM_TAG_FALSE: usize = 2;
const LIT_TERM_TAG_INT: usize = 3;
const LIT_TERM_TAG_STR: usize = 4;

#[repr(transparent)]
pub struct LitTerm_ {
  raw_tptr: usize,
  //raw_len:  usize,
}

impl Clone for LitTerm_ {
  fn clone(&self) -> LitTerm_ {
    let tag = self.raw_tptr & LIT_TERM_TAG_MASK;
    let raw_ptr = self.raw_tptr ^ tag;
    match tag {
      LIT_TERM_TAG_NONE |
      LIT_TERM_TAG_TRUE |
      LIT_TERM_TAG_FALSE => {
        LitTerm_{raw_tptr: self.raw_tptr}
      }
      LIT_TERM_TAG_INT => {
        let ptr = unsafe {
          let ptr = Arc::from_raw(raw_ptr as *const i64);
          let ptr2 = ptr.clone();
          let _ = Arc::into_raw(ptr);
          ptr2
        };
        let raw_ptr = Arc::into_raw(ptr) as usize;
        assert_eq!(raw_ptr & LIT_TERM_TAG_MASK, 0);
        let raw_tptr = raw_ptr ^ LIT_TERM_TAG_INT;
        LitTerm_{raw_tptr}
      }
      LIT_TERM_TAG_STR => {
        let ptr = unsafe {
          let ptr = Arc::from_raw(raw_ptr as *const SafeStr);
          let ptr2 = ptr.clone();
          let _ = Arc::into_raw(ptr);
          ptr2
        };
        let raw_ptr = Arc::into_raw(ptr) as usize;
        assert_eq!(raw_ptr & LIT_TERM_TAG_MASK, 0);
        let raw_tptr = raw_ptr ^ LIT_TERM_TAG_INT;
        LitTerm_{raw_tptr}
      }
      _ => {
        // TODO: unfortunately we can't return an interp check here!
        panic!("bug");
      }
    }
  }
}

impl PartialEq for LitTerm_ {
  fn eq(&self, rhs: &LitTerm_) -> bool {
    let ltag = self.raw_tptr & LIT_TERM_TAG_MASK;
    let rtag = rhs.raw_tptr & LIT_TERM_TAG_MASK;
    let l_raw_ptr = self.raw_tptr ^ ltag;
    let r_raw_ptr = rhs.raw_tptr ^ rtag;
    match (ltag, rtag) {
      (LIT_TERM_TAG_NONE, LIT_TERM_TAG_NONE) |
      (LIT_TERM_TAG_TRUE, LIT_TERM_TAG_TRUE) |
      (LIT_TERM_TAG_FALSE, LIT_TERM_TAG_FALSE) => {
        true
      }
      (LIT_TERM_TAG_INT, LIT_TERM_TAG_INT) => {
        unsafe {
          let l_ptr = Arc::from_raw(l_raw_ptr as *const i64);
          let r_ptr = Arc::from_raw(r_raw_ptr as *const i64);
          let res = l_ptr == r_ptr;
          let _ = Arc::into_raw(l_ptr);
          let _ = Arc::into_raw(r_ptr);
          res
        }
      }
      (LIT_TERM_TAG_STR, LIT_TERM_TAG_STR) => {
        unsafe {
          let l_ptr = Arc::from_raw(l_raw_ptr as *const SafeStr);
          let r_ptr = Arc::from_raw(r_raw_ptr as *const SafeStr);
          let res = l_ptr == r_ptr;
          let _ = Arc::into_raw(l_ptr);
          let _ = Arc::into_raw(r_ptr);
          res
        }
      }
      _ => false
    }
  }
}

impl Eq for LitTerm_ {
}

impl Hash for LitTerm_ {
  fn hash<H: Hasher>(&self, state: &mut H) {
    let tag = self.raw_tptr & LIT_TERM_TAG_MASK;
    let raw_ptr = self.raw_tptr ^ tag;
    match tag {
      LIT_TERM_TAG_NONE => {
        ().hash(state)
      }
      LIT_TERM_TAG_TRUE => {
        true.hash(state)
      }
      LIT_TERM_TAG_FALSE => {
        false.hash(state)
      }
      LIT_TERM_TAG_INT => {
        unsafe {
          let ptr = Arc::from_raw(raw_ptr as *const i64);
          ptr.hash(state);
          let _ = Arc::into_raw(ptr);
        }
      }
      LIT_TERM_TAG_STR => {
        unsafe {
          let ptr = Arc::from_raw(raw_ptr as *const SafeStr);
          ptr.hash(state);
          let _ = Arc::into_raw(ptr);
        }
      }
      _ => {
        panic!("bug");
      }
    }
  }
}

impl Debug for LitTerm_ {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    let tag = self.raw_tptr & LIT_TERM_TAG_MASK;
    let raw_ptr = self.raw_tptr ^ tag;
    match tag {
      LIT_TERM_TAG_NONE => {
        write!(f, "LitTerm_(None)")
      }
      LIT_TERM_TAG_TRUE => {
        write!(f, "LitTerm_(True)")
      }
      LIT_TERM_TAG_FALSE => {
        write!(f, "LitTerm_(False)")
      }
      LIT_TERM_TAG_INT => {
        unsafe {
          let ptr = Arc::from_raw(raw_ptr as *const i64);
          write!(f, "LitTerm_(Int={:?})", ptr)?;
          let _ = Arc::into_raw(ptr);
        }
        Ok(())
      }
      LIT_TERM_TAG_STR => {
        unsafe {
          let ptr = Arc::from_raw(raw_ptr as *const SafeStr);
          // NB: ptr is a boxed SafeStr, so okay to directly display;
          // also, debug print adds an unnecessary quotation level.
          write!(f, "LitTerm_(Str={})", ptr)?;
          let _ = Arc::into_raw(ptr);
        }
        Ok(())
      }
      _ => {
        write!(f, "LitTerm_(raw={:x}.{})", raw_ptr, tag)
      }
    }
  }
}

impl LitTerm_ {
  pub fn new_none() -> LitTerm_ {
    LitTerm_{raw_tptr: LIT_TERM_TAG_NONE}
  }

  pub fn new_true() -> LitTerm_ {
    LitTerm_{raw_tptr: LIT_TERM_TAG_TRUE}
  }

  pub fn new_false() -> LitTerm_ {
    LitTerm_{raw_tptr: LIT_TERM_TAG_FALSE}
  }

  pub fn new_bool(v: bool) -> LitTerm_ {
    match v {
      true => LitTerm_::new_true(),
      false => LitTerm_::new_false(),
    }
  }

  pub fn new_int(v: i64) -> LitTerm_ {
    let ptr = Arc::new(v);
    let raw_ptr = Arc::into_raw(ptr) as usize;
    assert_eq!(raw_ptr & LIT_TERM_TAG_MASK, 0);
    let raw_tptr = raw_ptr ^ LIT_TERM_TAG_INT;
    LitTerm_{raw_tptr}
  }

  pub fn new_str(v: SafeStr) -> LitTerm_ {
    // FIXME: directly convert SafeStr, which is SmolStr-backed, into Arc<str>.
    let ptr = Arc::new(v);
    let raw_ptr = Arc::into_raw(ptr) as usize;
    assert_eq!(raw_ptr & LIT_TERM_TAG_MASK, 0);
    let raw_tptr = raw_ptr ^ LIT_TERM_TAG_STR;
    LitTerm_{raw_tptr}
  }

  #[inline]
  pub fn _tag(&self) -> usize {
    let tag = self.raw_tptr & LIT_TERM_TAG_MASK;
    tag
  }

  pub fn _unpack(&self) -> UnpackedLitTerm_ {
    let tag = self.raw_tptr & LIT_TERM_TAG_MASK;
    let raw_ptr = self.raw_tptr ^ tag;
    match tag {
      LIT_TERM_TAG_NONE => {
        UnpackedLitTerm_::None
      }
      LIT_TERM_TAG_TRUE => {
        UnpackedLitTerm_::True
      }
      LIT_TERM_TAG_FALSE => {
        UnpackedLitTerm_::False
      }
      LIT_TERM_TAG_INT => {
        let ptr = unsafe {
          let ptr = Arc::from_raw(raw_ptr as *const i64);
          let ptr2 = ptr.clone();
          let _ = Arc::into_raw(ptr);
          ptr2
        };
        UnpackedLitTerm_::Int(ptr)
      }
      LIT_TERM_TAG_STR => {
        let ptr = unsafe {
          let ptr = Arc::from_raw(raw_ptr as *const SafeStr);
          let ptr2 = ptr.clone();
          let _ = Arc::into_raw(ptr);
          ptr2
        };
        UnpackedLitTerm_::Str(ptr)
      }
      _ => {
        // TODO: unfortunately we can't return an interp check here!
        panic!("bug");
      }
    }
  }
}

// [Interp-API]
#[derive(Clone, Serialize, Debug)]
#[repr(transparent)]
pub struct NEqualTerm_ {
  buf:  [ENum; 2],
}

// [Interp-API]
#[derive(Clone, Serialize, Debug)]
#[repr(transparent)]
pub struct TupleTerm_ {
  buf:  Box<[ENum]>,
}

// [Interp-API]
#[derive(Clone, Debug)]
pub struct MsgTerm_ {
  recv: ENum,
  buf:  Box<[ENum]>,
}

pub struct NoneObj_ {}
pub struct BoolObj_ { inner: bool }
pub struct IntObj_ { inner: i64 }
pub struct StrObj_ { inner: SafeStr }
pub struct ListObj_ { buf: Vec<ENum> }

// [Interp-API]
#[derive(Clone, Debug)]
pub enum LitVal_ {
  None,
  Bool(bool),
  Int(i64),
  Atom(SafeStr),
  Box{buf: Option<SNum>},
  List{buf: Vec<SNum>},
  //Dict{key: Vec<SNum>, map: FxHashMap<SNum, SNum>},
}

macro_rules! impl_snum_subtype {
  ($T:tt) => {
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct $T(RawSNum);

    impl From<$T> for SNum {
      fn from(x: $T) -> SNum {
        SNum(x.0)
      }
    }

    impl Nil for $T {
      fn nil() -> $T {
        $T(0)
      }

      fn is_nil(&self) -> bool {
        (self.0 >> SNUM_TAG_BITS) == 0
      }
    }

    impl Debug for $T {
      fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let key = self.0 >> SNUM_TAG_BITS;
        let tag = self.0 & SNUM_TAG_MASK;
        write!(f, concat!(stringify!($T), "({}.{})"), key, tag)
      }
    }
  };
}

macro_rules! impl_cell_num_subtype {
  ($T:tt) => {
    impl From<$T> for CellNum {
      fn from(x: $T) -> CellNum {
        CellNum(x.0)
      }
    }

    impl_snum_subtype!($T);
  };
}

// [Interp-API]
//
// Builtin function.
impl_snum_subtype!(FunNum);

// [Interp-API]
//
// Builtin object class.
impl_snum_subtype!(ObjClsNum);

// [Interp-API]
//
// Builtin object value.
impl_snum_subtype!(ObjValNum);

// [Interp-API]
//
// A pointer to a tuple-cell.
impl_snum_subtype!(CellNum);

impl Nil for Cell<CellNum> {
  fn nil() -> Cell<CellNum> {
    Cell::new(nil())
  }

  fn is_nil(&self) -> bool {
    self.get().is_nil()
  }
}

impl CellNum {
  pub fn into_stm_code(self) -> StmCodeCellNum {
    // FIXME: tag sort check.
    StmCodeCellNum(self.0)
  }

  pub fn into_term_code(self) -> TermCodeCellNum {
    // FIXME: tag sort check.
    TermCodeCellNum(self.0)
  }
}

// [Interp-API]
impl_snum_subtype!(AtomNum);

// [Interp-API]
impl_snum_subtype!(IdentNum);

// [To-Deprecate/Interp-API]
impl_snum_subtype!(LitNum);

// [Interp-API]
impl_snum_subtype!(LitStrNum);

// [To-Deprecate]
//impl_snum_subtype!(StrNum);
//impl_snum_subtype!(IntNum);

impl_snum_subtype!(TermNum);
//impl_snum_subtype!(ValNum);

impl_snum_subtype!(SpanNum);
impl_snum_subtype!(ModCodeNum);
impl_snum_subtype!(StmCodeNum);
impl_snum_subtype!(TermCodeNum);

impl_cell_num_subtype!(StmCodeCellNum);
impl_cell_num_subtype!(TermCodeCellNum);

#[derive(Clone, Copy, Debug)]
pub struct ModCode_ {
  span: SpanNum,
  stmp: StmCodeCellNum,
}

#[derive(Clone, Debug)]
pub enum StmCode_ {
  Just{span: SpanNum, term: TermCodeNum},
  Pass{span: SpanNum},
  If{span: SpanNum, cases: Vec<(TermCodeNum, StmCodeCellNum)>, final_case: StmCodeCellNum},
  With{span: SpanNum, ctx: TermCodeNum, stmp: StmCodeCellNum},
  // TODO
  Defproc{span: SpanNum, body_stmp: StmCodeCellNum},
  //Defproc{span: SpanNum, ..., body_stmp: StmCodeCellNum},
  Defmatch{span: SpanNum, body_stmp: StmCodeCellNum},
  //Defmatch{span: SpanNum, ..., body_stmp: StmCodeCellNum},
  // FIXME: stm only b/c of parsing hack.
  Quote{span: SpanNum},
}

#[derive(Clone, Debug)]
pub enum TermCode_ {
  Ident{span: SpanNum, id: IdentNum},
  QualIdent{span: SpanNum, term: TermCodeNum, id: IdentNum},
  AtomLit{span: SpanNum, lit_str: LitStrNum},
  IntLit{span: SpanNum, lit_str: LitStrNum},
  BoolLit{span: SpanNum, lit_str: LitStrNum},
  NoneLit{span: SpanNum, lit_str: LitStrNum},
  ListCon{span: SpanNum, tup: TermCodeCellNum},
  Neg{span: SpanNum, term: TermCodeNum},
  Group{span: SpanNum, term: TermCodeNum},
  Bunch{span: SpanNum, tup: TermCodeCellNum},
  Query{span: SpanNum, term: TermCodeNum},
  Equal{span: SpanNum, lterm: TermCodeNum, rterm: TermCodeNum},
  NEqual{span: SpanNum, lterm: TermCodeNum, rterm: TermCodeNum},
  QEqual{span: SpanNum, lterm: TermCodeNum, rterm: TermCodeNum},
  BindL{span: SpanNum, lterm: TermCodeNum, rterm: TermCodeNum},
  BindR{span: SpanNum, lterm: TermCodeNum, rterm: TermCodeNum},
  Subst{span: SpanNum, lterm: TermCodeNum, rterm: TermCodeNum},
  RebindL{span: SpanNum, lterm: TermCodeNum, rterm: TermCodeNum},
  RebindR{span: SpanNum, lterm: TermCodeNum, rterm: TermCodeNum},
  Apply{span: SpanNum, tup: TermCodeCellNum},
  ApplyBindL{span: SpanNum, lterm: TermCodeNum, tup: TermCodeCellNum},
  ApplyBindR{span: SpanNum, tup: TermCodeCellNum, rterm: TermCodeNum},
  ApplyQuery{span: SpanNum, tup: TermCodeCellNum},
  Effect{span: SpanNum, lterm: TermCodeNum, rtup: TermCodeCellNum},
}

impl TermCode_ {
  pub fn _span(&self) -> Result<SpanNum, ()> {
    Ok(match self {
      &TermCode_::Apply{span, ..} => span,
      _ => return Err(())
    })
  }
}

// [Interp-API]
//
// A "pointer" to an in-memory continuation.
pub type MemKntRef = Option<Box<MemKnt>>;

// [Interp-API]
//
// In-memory continuation.
#[derive(Clone, Debug)]
pub struct MemKnt {
  clk:  LClk,
  prev: MemKntRef,
  cur:  MemKnt_,
}

// [Interp-API]
impl MemKnt {
  #[inline]
  pub fn into_ref(self) -> MemKntRef {
    Some(self.into())
  }
}

pub struct BorrowedMemKnt<'prev> {
  pub clk:  LClk,
  pub prev: &'prev MemKntRef,
  pub cur:  MemKnt_,
}

// [Interp-API]
//
// Continuation ADT variants.
//
// The continuation state is initially `Uninit`. Code-specific states are
// associated with micro-states (e.g. `ModCodeInterpState_`).
//
// Quiescent control state corresponds to a nil continuation "pointer".
#[derive(Clone, Default, Debug)]
pub enum MemKnt_ {
  #[default]
  Uninit,
  // TODO
  InterpMod(ModCodeNum, ModCodeInterpState_),
  InterpStmp(StmCodeCellNum, StmCodeCellInterpState_),
  InterpStm(StmCodeNum, StmCodeInterpState_),
  InterpIfStm(StmCodeNum, IfStmCodeInterpState_),
  InterpTerm(TermCodeNum, TermCodeInterpState_),
  InterpQualIdentTerm(TermCodeNum, QualIdentTermCodeInterpState_),
  InterpBunchTerm(TermCodeNum, BunchTermCodeInterpState_),
  InterpEqualTerm(TermCodeNum, EqualTermCodeInterpState_),
  InterpNEqualTerm(TermCodeNum, NEqualTermCodeInterpState_),
  InterpQEqualTerm(TermCodeNum, QEqualTermCodeInterpState_),
  InterpApplyTerm(TermCodeNum, ApplyTermCodeInterpState_),
  InterpApplyBindLTerm(TermCodeNum, ApplyBindLTermCodeInterpState_),
  InterpApplyBindRTerm(TermCodeNum, ApplyBindRTermCodeInterpState_),
  InterpBindLTerm(TermCodeNum, BindLTermCodeInterpState_),
  InterpBindRTerm(TermCodeNum, BindRTermCodeInterpState_),
  InterpEffectTerm(TermCodeNum, EffectTermCodeInterpState_),
}

// [Interp-API]
//
// This controls the context of term interpretation.
#[derive(Clone, Copy, Default, Debug)]
#[repr(u8)]
pub enum TermContext_ {
  // NB: easier to default to Just.
  #[default]
  /*Uninit,*/
  Unify,
  Match,
  //ProcMatch,
}

#[derive(Clone, Debug)]
pub struct ModCodeInterpState_ {
  stmp: StmCodeCellNum,
}

impl ModCodeInterpState_ {
  pub fn fresh() -> ModCodeInterpState_ {
    ModCodeInterpState_{
      stmp: nil(),
    }
  }
}

#[derive(Clone, Debug)]
pub struct StmCodeCellInterpState_ {
  stmp: StmCodeCellNum,
}

impl StmCodeCellInterpState_ {
  pub fn fresh(stmp: StmCodeCellNum) -> StmCodeCellInterpState_ {
    StmCodeCellInterpState_{
      stmp,
    }
  }
}

#[derive(Clone, Debug)]
pub struct StmCodeInterpState_ {
}

impl StmCodeInterpState_ {
  pub fn fresh() -> StmCodeInterpState_ {
    StmCodeInterpState_{
    }
  }
}

#[derive(Clone, Debug)]
pub enum IfStmCodeInterpCursor_ {
  Cond(TermCodeNum, StmCodeCellNum, usize, Vec<(TermCodeNum, StmCodeCellNum)>, StmCodeCellNum),
  Body(StmCodeCellNum, usize, Vec<(TermCodeNum, StmCodeCellNum)>, StmCodeCellNum),
  FinalBody(StmCodeCellNum),
  Fin,
}

#[derive(Clone, Debug)]
pub struct IfStmCodeInterpState_ {
  cur:  IfStmCodeInterpCursor_,
  save_tctx: Option<TermContext_>,
}

impl IfStmCodeInterpState_ {
  pub fn fresh(cases: Vec<(TermCodeNum, StmCodeCellNum)>, final_case: StmCodeCellNum) -> IfStmCodeInterpState_ {
    IfStmCodeInterpState_{
      cur:  IfStmCodeInterpCursor_::Cond(cases[0].0, cases[0].1, 1, cases, final_case),
      save_tctx: None,
    }
  }
}

#[derive(Clone, Debug)]
pub struct TermCodeInterpState_ {
}

impl TermCodeInterpState_ {
  pub fn fresh() -> TermCodeInterpState_ {
    TermCodeInterpState_{
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum QualIdentTermCodeInterpCursor_ {
  Term(TermCodeNum, IdentNum),
  Ident(IdentNum),
  Fin,
}

#[derive(Clone, Debug)]
pub struct QualIdentTermCodeInterpState_ {
  term:     Option<(TermCodeNum, SNum)>,
  ident:    Option<(IdentNum, SNum)>,
  cur:      QualIdentTermCodeInterpCursor_,
}

impl QualIdentTermCodeInterpState_ {
  pub fn fresh(term: TermCodeNum, id: IdentNum) -> QualIdentTermCodeInterpState_ {
    QualIdentTermCodeInterpState_{
      term: None,
      ident: None,
      cur:  QualIdentTermCodeInterpCursor_::Term(term, id),
    }
  }
}

#[derive(Clone, Debug)]
pub struct BunchTermCodeInterpState_ {
  tup:  Vec<(TermCodeNum, SNum)>,
  cur:  TermCodeCellNum,
}

impl BunchTermCodeInterpState_ {
  pub fn fresh(init_cur: TermCodeCellNum) -> BunchTermCodeInterpState_ {
    BunchTermCodeInterpState_{
      tup:  Vec::new(),
      cur:  init_cur,
    }
  }
}

macro_rules! impl_binop_term_code_interp_state {
  ($T:ident) => { paste! {
    #[derive(Clone, Copy, Debug)]
    pub enum [<$T TermCodeInterpCursor_>] {
      LTerm(TermCodeNum, TermCodeNum),
      RTerm(TermCodeNum),
      Fin,
    }

    #[derive(Clone, Debug)]
    pub struct [<$T TermCodeInterpState_>] {
      lterm: Option<(TermCodeNum, SNum)>,
      rterm: Option<(TermCodeNum, SNum)>,
      cur:  [<$T TermCodeInterpCursor_>],
    }

    impl [<$T TermCodeInterpState_>] {
      pub fn fresh(lterm: TermCodeNum, rterm: TermCodeNum) -> [<$T TermCodeInterpState_>] {
        [<$T TermCodeInterpState_>]{
          lterm: None,
          rterm: None,
          cur:  [<$T TermCodeInterpCursor_>]::LTerm(lterm, rterm),
        }
      }
    }
  } };
}

impl_binop_term_code_interp_state!(Equal);
impl_binop_term_code_interp_state!(NEqual);
impl_binop_term_code_interp_state!(QEqual);

#[derive(Clone, Copy, Debug)]
pub enum BindLTermCodeInterpCursor_ {
  LBind(TermCodeNum, TermCodeNum),
  RTerm(TermCodeNum),
  Fin,
}

#[derive(Clone, Debug)]
pub struct BindLTermCodeInterpState_ {
  lbind:    Option<(TermCodeNum, SNum)>,
  rterm:    Option<(TermCodeNum, SNum)>,
  cur:      BindLTermCodeInterpCursor_,
}

impl BindLTermCodeInterpState_ {
  pub fn fresh(lbind: TermCodeNum, rterm: TermCodeNum) -> BindLTermCodeInterpState_ {
    BindLTermCodeInterpState_{
      lbind: None,
      rterm: None,
      cur:  BindLTermCodeInterpCursor_::LBind(lbind, rterm),
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum BindRTermCodeInterpCursor_ {
  LTerm(TermCodeNum, TermCodeNum),
  RBind(TermCodeNum),
  Fin,
}

#[derive(Clone, Debug)]
pub struct BindRTermCodeInterpState_ {
  lterm:    Option<(TermCodeNum, SNum)>,
  rbind:    Option<(TermCodeNum, SNum)>,
  cur:      BindRTermCodeInterpCursor_,
}

impl BindRTermCodeInterpState_ {
  pub fn fresh(lterm: TermCodeNum, rbind: TermCodeNum) -> BindRTermCodeInterpState_ {
    BindRTermCodeInterpState_{
      lterm: None,
      rbind: None,
      cur:  BindRTermCodeInterpCursor_::LTerm(lterm, rbind),
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum SubstTermCodeInterpCursor_ {
  LTerm(TermCodeNum, TermCodeNum),
  RTerm(TermCodeNum),
  Fin,
}

#[derive(Clone, Debug)]
pub struct SubstTermCodeInterpState_ {
  lterm:    Option<(TermCodeNum, SNum)>,
  rterm:    Option<(TermCodeNum, SNum)>,
  cur:      SubstTermCodeInterpCursor_,
}

impl SubstTermCodeInterpState_ {
  pub fn fresh(lterm: TermCodeNum, rterm: TermCodeNum) -> SubstTermCodeInterpState_ {
    SubstTermCodeInterpState_{
      lterm: None,
      rterm: None,
      cur:  SubstTermCodeInterpCursor_::LTerm(lterm, rterm),
    }
  }
}

#[derive(Clone, Debug)]
pub struct ApplyTermCodeInterpState_ {
  tup:  Vec<(TermCodeNum, SNum)>,
  cur:  TermCodeCellNum,
}

impl ApplyTermCodeInterpState_ {
  pub fn fresh(init_cur: TermCodeCellNum) -> ApplyTermCodeInterpState_ {
    ApplyTermCodeInterpState_{
      tup:  Vec::new(),
      cur:  init_cur,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum ApplyBindLTermCodeInterpCursor_ {
  Bind(TermCodeNum, TermCodeCellNum),
  Tup(TermCodeCellNum),
  Fin,
}

#[derive(Clone, Debug)]
pub struct ApplyBindLTermCodeInterpState_ {
  bind:     Option<(TermCodeNum, SNum)>,
  tup:      Vec<(TermCodeNum, SNum)>,
  cur:      ApplyBindLTermCodeInterpCursor_,
}

impl ApplyBindLTermCodeInterpState_ {
  pub fn fresh(bind_cur: TermCodeNum, tup_cur: TermCodeCellNum) -> ApplyBindLTermCodeInterpState_ {
    ApplyBindLTermCodeInterpState_{
      bind: None,
      tup:  Vec::new(),
      cur:  ApplyBindLTermCodeInterpCursor_::Bind(bind_cur, tup_cur),
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum ApplyBindRTermCodeInterpCursor_ {
  Tup(TermCodeCellNum, TermCodeNum),
  Bind(TermCodeNum),
  Fin,
}

#[derive(Clone, Debug)]
pub struct ApplyBindRTermCodeInterpState_ {
  tup:      Vec<(TermCodeNum, SNum)>,
  bind:     Option<(TermCodeNum, SNum)>,
  cur:      ApplyBindRTermCodeInterpCursor_,
}

impl ApplyBindRTermCodeInterpState_ {
  pub fn fresh(tup_cur: TermCodeCellNum, bind_cur: TermCodeNum) -> ApplyBindRTermCodeInterpState_ {
    ApplyBindRTermCodeInterpState_{
      tup:  Vec::new(),
      bind: None,
      cur:  ApplyBindRTermCodeInterpCursor_::Tup(tup_cur, bind_cur),
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum EffectTermCodeInterpCursor_ {
  LTerm(TermCodeNum, TermCodeCellNum),
  RTup(TermCodeCellNum),
  Fin,
}

#[derive(Clone, Debug)]
pub struct EffectTermCodeInterpState_ {
  lterm:    Option<(TermCodeNum, SNum)>,
  rtup:     Vec<(TermCodeNum, SNum)>,
  cur:      EffectTermCodeInterpCursor_,
}

impl EffectTermCodeInterpState_ {
  pub fn fresh(lterm_cur: TermCodeNum, rtup_cur: TermCodeCellNum) -> EffectTermCodeInterpState_ {
    EffectTermCodeInterpState_{
      lterm: None,
      rtup: Vec::new(),
      cur:  EffectTermCodeInterpCursor_::LTerm(lterm_cur, rtup_cur),
    }
  }
}

// [Interp-API]
#[derive(Clone, Debug)]
pub struct Error_ {
  pub loc:  Loc,
  pub msg:  SmolStr,
}

// [Interp-API]
#[derive(Clone, Default, Debug)]
pub struct Except_ {
  // FIXME
  //_reg: Option<SNum>,
  _err: Option<Error_>,
}

impl<'a> From<&'a str> for Except_ {
  #[track_caller]
  fn from(msg: &'a str) -> Except_ {
    let loc = loc();
    Except_{_err: Some(Error_{loc, msg: msg.into()})}
  }
}

impl From<String> for Except_ {
  #[track_caller]
  fn from(msg: String) -> Except_ {
    let loc = loc();
    Except_{_err: Some(Error_{loc, msg: msg.into()})}
  }
}

impl Except_ {
  pub fn is_some(&self) -> bool {
    self._err.is_some()
  }
}

// [Interp-API]
#[derive(Clone, Copy, Default, Debug)]
pub enum ResReg_ {
  #[default]
  Emp,
  Key(SNum),
  Mat(bool),
}

// [Interp-API]
#[derive(Clone, Default, Debug)]
pub struct Result_ {
  reg:  ResReg_,
  // FIXME: debugging.
  _log: Vec<ResReg_>,
}

impl Result_ {
  pub fn reset(&mut self) -> ResReg_ {
    let x = replace(&mut self.reg, ResReg_::Emp);
    self._log.push(x);
    x
  }

  pub fn put(&mut self, x: SNum) -> ResReg_ {
    replace(&mut self.reg, ResReg_::Key(x))
  }

  pub fn get(&mut self) -> ResReg_ {
    replace(&mut self.reg, ResReg_::Emp)
  }

  pub fn peek(&self) -> ResReg_ {
    self.reg
  }
}

// [Interp-API]
pub trait Tabled: Any + Debug {
  fn as_any(&self) -> &dyn Any;
}

// [Interp-API]
pub trait HashTabled: Tabled + Eq + Hash {
}

macro_rules! impl_tabled {
  ($T:tt) => {
    impl Tabled for $T {
      fn as_any(&self) -> &dyn Any { self }
    }
  };
}

macro_rules! impl_hash_tabled {
  ($T:tt) => {
    impl HashTabled for $T {
    }

    impl_tabled!($T);
  };
}

impl_tabled!(Cell_);

impl_tabled!(AtomTerm_);
impl_tabled!(IdentTerm_);
impl_tabled!(LitTerm_);

impl_tabled!(NEqualTerm_);
impl_tabled!(TupleTerm_);

impl_tabled!(LitVal_);

impl_tabled!(SafeStr);

impl_tabled!(RawSpan_);
impl_tabled!(ModCode_);
impl_tabled!(StmCode_);
impl_tabled!(TermCode_);

// [Interp-API]
//
// An eclass-enode/eid pair.
#[derive(Clone, Copy, Default)]
pub struct ENum {
  cls:  SNum,
  inst: SNum,
}

// FIXME: SNum into ENum should go through a unifier.
impl From<SNum> for ENum {
  fn from(x: SNum) -> ENum {
    ENum{
      cls:  x,
      inst: x,
    }
  }
}

impl Serialize for ENum {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    let mut state = serializer.serialize_struct("ENum", 2)?;
    state.serialize_field("ENum_cls", &self._cls())?;
    state.serialize_field("ENum_inst", &self._inst())?;
    state.end()
  }
}

impl Debug for ENum {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "ENum({}.{}:={}.{})",
        self.cls._key(), self.cls._tag(),
        self.inst._key(), self.inst._tag(),
    )
  }
}

impl PartialEq for ENum {
  fn eq(&self, rhs: &ENum) -> bool {
    // FIXME: ENum comparison should go through a unifier.
    self.cls == rhs.cls
  }
}

impl Eq for ENum {
}

impl PartialOrd for ENum {
  fn partial_cmp(&self, rhs: &ENum) -> Option<Ordering> {
    // FIXME: ENum comparison should go through a unifier.
    self.cls.partial_cmp(&rhs.cls)
  }
}

impl Ord for ENum {
  fn cmp(&self, rhs: &ENum) -> Ordering {
    // FIXME: ENum comparison should go through a unifier.
    self.cls.cmp(&rhs.cls)
  }
}

impl Hash for ENum {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.cls.hash(state)
  }
}

impl ENum {
  #[inline]
  pub fn _cls(&self) -> SNum {
    self.cls
  }

  #[inline]
  pub fn _inst(&self) -> SNum {
    self.inst
  }
}

impl From<()> for InterpCheck {
  #[track_caller]
  fn from(_: ()) -> InterpCheck {
    let loc = loc();
    let msg = "";
    InterpCheck{_err: Error_{loc, msg: msg.into()}}
  }
}

#[derive(Debug)]
pub enum UnifierCheck {
  _Bot,
}

impl From<UnifierCheck> for InterpCheck {
  #[track_caller]
  fn from(check: UnifierCheck) -> InterpCheck {
    let loc = loc();
    // FIXME: debug message.
    let msg = format!("{}: {:?}", type_name::<UnifierCheck>(), check);
    InterpCheck{_err: Error_{loc, msg: msg.into()}}
  }
}

impl IntoInterpCheckExt for UnifierCheck {}

// [Interp-API]
#[derive(Default)]
pub struct FastUnifier_ {
  root: BTreeSet<SNum>,
  next: FxHashMap<SNum, SNum>,
  prev: FxHashMap<SNum, SNum>,
  tree: FxHashMap<SNum, (LClk, SNum)>,
  cache: RefCell<FxHashMap<SNum, (LClk, SNum)>>,
}

impl FastUnifier_ {
  // [Interp-API]
  pub fn _next(&self, query: SNum) -> SNum {
    match self.next.get(&query) {
      Some(&next) => {
        next
      }
      None => {
        query
      }
    }
  }

  // [Interp-API]
  pub fn _prev(&self, query: SNum) -> SNum {
    match self.prev.get(&query) {
      Some(&prev) => {
        prev
      }
      None => {
        query
      }
    }
  }

  // [Interp-API]
  pub fn _link(&mut self, lquery: SNum, rquery: SNum) {
    self.next.insert(lquery, rquery);
    self.prev.insert(rquery, lquery);
  }

  // [Interp-API]
  pub fn _findall(&self, clkinval: &LClkInvalidSet, clk: LClk, query: SNum) -> Result<Vec<ENum>, UnifierCheck> {
    let mut buf = Vec::new();
    self._findall_into(clkinval, clk, query, &mut buf)?;
    Ok(buf)
  }

  // [Interp-API]
  pub fn _findall_into(&self, clkinval: &LClkInvalidSet, clk: LClk, query: SNum, buf: &mut Vec<ENum>) -> Result<(), UnifierCheck> {
    let root = self._find(clkinval, clk, query)?;
    let stop = root.inst;
    let mut cursor = stop;
    loop {
      buf.push(ENum{cls: root.cls, inst: cursor});
      match self.next.get(&cursor) {
        Some(&next) => {
          cursor = next;
        }
        None => {
          if cursor != stop {
            return Err(UnifierCheck::_Bot);
          }
          break;
        }
      }
      if cursor == stop {
        break;
      }
    }
    Ok(())
  }

  // [Interp-API]
  pub fn _find(&self, clkinval: &LClkInvalidSet, clk: LClk, query: SNum) -> Result<ENum, UnifierCheck> {
    if self.root.contains(&query) {
      return Ok(ENum{cls: query, inst: query});
    }
    let mut cache = self.cache.borrow_mut();
    let mut prev_up_clk = clk;
    let mut prev_cursor = query;
    let mut cursor = query;
    loop {
      match cache.get(&cursor) {
        Some(&(up_clk, up)) => {
          if clkinval._contains(up_clk) {
            // FIXME: potential fast path where cache entries are preserved
            // if the entry itself is still valid, though the cache entry
            // timestamp would still need to be updated.
            cache.remove(&cursor);
          } else {
            if cursor == up {
              return Err(UnifierCheck::_Bot);
            }
            if prev_cursor != cursor {
              // FIXME: check that this max() is sensible; maybe it can be
              // weakened to just up_clk.
              cache.insert(prev_cursor, (max(prev_up_clk, up_clk), up));
              prev_up_clk = up_clk;
              prev_cursor = cursor;
            }
            cursor = up;
            continue;
          }
        }
        None => {}
      }
      match self.tree.get(&cursor) {
        Some(&(up_clk, up)) => {
          if cursor == up {
            return Err(UnifierCheck::_Bot);
          }
          prev_up_clk = up_clk;
          prev_cursor = cursor;
          cursor = up;
        }
        None => {
          return Ok(ENum{cls: cursor, inst: query});
        }
      }
    }
  }

  // [Interp-API]
  pub fn _unify(&mut self, log: &mut FastLog_, clkinval: &LClkInvalidSet, clk: LClk, lquery: SNum, rquery: SNum, ) -> Result<SNum, UnifierCheck> {
    if lquery == rquery {
      let root = self._find(clkinval, clk, rquery)?;
      return Ok(root.cls);
    }
    if lquery > rquery {
      return self._unify(log, clkinval, clk, rquery, lquery);
    }
    let l_root = self._find(clkinval, clk, lquery)?;
    let r_root = self._find(clkinval, clk, rquery)?;
    if l_root.cls == r_root.cls {
      return Ok(r_root.cls);
    }
    // NB: invariant unification (lexicographic ordered roots).
    let (oroot, nroot) = if l_root.cls > r_root.cls {
      (l_root.cls, r_root.cls)
    } else {
      (r_root.cls, l_root.cls)
    };
    let onext = self._next(oroot);
    let nprev = self._prev(nroot);
    self._link(oroot, nroot);
    self._link(nprev, onext);
    self.root.remove(&oroot);
    self.root.insert(nroot);
    let otree = self.tree.insert(oroot, (clk, nroot));
    let mut cache = self.cache.borrow_mut();
    cache.insert(oroot, (clk, nroot));
    let state = UnifyUndoState_{
      oroot,
      nroot,
      onext,
      nprev,
      otree,
    };
    log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::Unify(state.into()).into()));
    Ok(nroot)
  }
}

// [Interp-API]
#[derive(Clone, Copy, Debug)]
pub struct FastReg_ {
  // `xlb` is saved at the beginning of a step (in resume_).
  xlb:      SNum,

  // `rst_clk` is set during yield fail, and is then used and unset
  // by the backtracked-to choice().
  rst_clk:  LClk,

  tctx:     TermContext_,
}

impl Default for FastReg_ {
  fn default() -> FastReg_ {
    FastReg_{
      xlb:      nil(),
      rst_clk:  nil(),
      tctx:     TermContext_::default(),
    }
  }
}

// [Interp-API]
#[derive(Clone, Default, Debug)]
pub struct FastCtlReg_ {
  exc_: Except_,
  res_: Result_,
  port: Port_,
}

#[derive(Clone, Debug)]
pub struct LogEntry_ {
  clk:  LClk,
  val:  LogEntryRef_,
}

// [Interp-API]
#[derive(Default, Debug)]
pub struct FastLog_ {
  buf:  Vec<LogEntry_>,
}

impl FastLog_ {
  // [Interp-API]
  pub fn _append(&mut self, clk: LClk, val: LogEntryRef_) {
    self.buf.push(LogEntry_{clk, val});
  }
}

// TODO: the choice point counter type is u16 for historical reasons,
// but consider bumping it up.
pub type RawChoiceRank = u16;

#[derive(Debug)]
pub struct TraceEntry_ {
  xctr: RawChoiceRank,
  xlim: RawChoiceRank,

  // The fresh linear timestamp at the current step, during which the choice
  // function is invoked.
  //
  // Actually observe two timestamps:
  // - "root" timestamp upon the initial _push
  // - any later timestamp, which is updated upon _push
  last_clk: Cell<LClk>,
  root_clk: LClk,

  // `xlb` saved at the very beginning of the same current step.
  xlb:  SNum,

  reg:  FastReg_,
  ctl_: FastCtlReg_,
  knt_: MemKntRef,
}

// [Interp-API]
//
// The choice trace (todo)
#[derive(Default, Debug)]
pub struct FastTrace_ {
  buf:  Vec<TraceEntry_>,
  clk_pos:  BTreeMap<LClk, u32>,
}

impl FastTrace_ {
  // [Interp-API]
  pub fn _maybe_get(&self, clk: LClk) -> Option<&TraceEntry_> {
    match self.clk_pos.get(&clk) {
      None => {
        None
      }
      Some(&pos) => {
        Some(&self.buf[pos as usize])
      }
    }
  }

  // [Interp-API]
  pub fn _push(&mut self, clk: LClk, choice_ub: RawChoiceRank, xlb: SNum, reg: FastReg_, ctl_: FastCtlReg_, knt_: MemKntRef) -> Result<(), ()> {
    let pos: u32 = self.buf.len().try_into().unwrap();
    self.buf.push(TraceEntry_{
      xctr: 0,
      xlim: choice_ub,
      last_clk: Cell::new(clk),
      root_clk: clk,
      xlb,
      reg,
      ctl_,
      knt_,
    });
    match self.clk_pos.insert(clk, pos) {
      Some(_) => {
        // NB: this case should never happen.
        return Err(());
      }
      None => {}
    }
    Ok(())
  }

  // [Interp-API]
  pub fn _pop(&mut self) -> Result<(), ()> {
    unimplemented!();
  }

  // [Interp-API]
  pub fn _pop_pos(&mut self, pos: u32) -> Result<(), ()> {
    if (pos + 1) as usize != self.buf.len() {
      return Err(());
    }
    let te = self.buf.pop().unwrap();
    match self.clk_pos.remove(&te.root_clk) {
      None => {
        return Err(());
      }
      Some(opos) => if opos != pos {
        return Err(());
      }
    }
    Ok(())
  }
}

// [Interp-API]
pub enum TransparentBox<V: ?Sized> {
  Ptr(Box<V>),
  Blk,
}

impl<V: ?Sized> From<Box<V>> for TransparentBox<V> {
  fn from(inner: Box<V>) -> TransparentBox<V> {
    TransparentBox::Ptr(inner)
  }
}

impl<V: ?Sized + Debug> Debug for TransparentBox<V> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    match self {
      TransparentBox::Ptr(ref inner) => {
        write!(f, "TransparentBox<{}>(Ptr={:?})", type_name::<V>(), inner)
      }
      _ => {
        write!(f, "TransparentBox<{}>(Blk)", type_name::<V>())
      }
    }
  }
}

impl<V: ?Sized> TransparentBox<V> {
  pub fn _borrow(&mut self) -> TransparentBox<V> {
    replace(self, TransparentBox::Blk)
  }

  pub fn _swap<T: Into<TransparentBox<V>>>(&mut self, t: T) -> TransparentBox<V> {
    replace(self, t.into())
  }
}

// [Interp-API]
pub trait Function: Any + Debug {
  fn as_any(&self) -> &dyn Any;
  fn __apply__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<Option<Yield_>, InterpCheck>;
}

// [Interp-API]
pub trait ObjectCls: Any + Debug {
  fn as_any(&self) -> &dyn Any;
  fn __create__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<(), InterpCheck>;
}

// [Interp-API]
pub trait ObjectVal: Any + Debug {
  fn as_any(&self) -> &dyn Any;
  fn __init__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<(), InterpCheck>;
  fn __destroy__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<(), InterpCheck>;
  fn __request__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<(), InterpCheck>;
}

// [Interp-API]
pub struct Namespace {
  // TODO: something that can be addressed via qual ident.
  // in particular, one such thing is `__builtins__`.
}

// [Interp-API]
#[derive(Default)]
pub struct FastEnv_ {
  // NB: below, `SNum` in "key"-like position should be interpreted
  // as the "original instance" of an `ENum`.

  // FIXME: may want to fold this in as a sort in the tableau below.
  fun_name:     FxHashMap<IdentNum, SNum>,
  fun_full:     FxHashMap<SNum, TransparentBox<dyn Function>>,

  // FIXME: may want to fold this in as a sort in the tableau below.
  obj_cls_name: FxHashMap<IdentNum, SNum>,
  obj_cls_full: FxHashMap<SNum, TransparentBox<dyn ObjectCls>>,

  // TODO: tabled term storage should likely store tuples of _ENum_
  // instead of _SNum_.
  table_full:   Vec<FxHashMap<SNum, Box<dyn Tabled>>>,
  // TODO: seminaive tables.
  //table_prev:   Vec<FxHashMap<SNum, Box<dyn Tabled>>>,
  //table_new:    Vec<FxHashMap<SNum, Box<dyn Tabled>>>,

  raw_span_index: FxHashMap<RawSpan_, SpanNum>,
  raw_id_index: FxHashMap<RawIdent_, IdentNum>,
  raw_id_bind:  FxHashMap<RawIdent_, SNum>,

  // TODO: a "qual id" is a pair of a term-like Num and an ident.
  qual_id_index: FxHashMap<(SNum, RawIdent_), IdentNum>,

  // TODO: literal syntax allows multiple different literal strings
  // to map to one literal term.
  raw_lit_index: FxHashMap<RawLit_, LitStrNum>,
  raw_lit_cache: FxHashMap<RawLit_, LitTerm_>,
  lit_term_bind: FxHashMap<LitTerm_, SNum>,

  unifier:  FastUnifier_,

  rule_index:   FxHashMap<StmCodeNum, ()>,
}

impl FastEnv_ {
  // [Interp-API]
  pub fn _pre_init(&mut self, ctr: &SNumCtr) {
    // TODO: tableau lookup order.
    while self.table_full.len() <= _SNUM_MAX_SORT as _ {
      self.table_full.push(Default::default());
    }
  }
}

// [Interp-API]
//
// Interpreter control state transitions are associated w/ "ports" (following
// the "4-port model" of Prolog).
//
// The port is conventionally set at the end of a control state transition.
// Control transitions themselves are switched on the pair of (1) port and
// (2) continuation.
#[derive(Clone, Copy, Default, Debug)]
pub enum Port_ {
  #[default]
  Quiescent,
  Enter,
  Return,
}

// [Interp-API]
#[derive(PartialEq, Eq, Debug)]
pub enum Yield_ {
  // FIXME: these generally need labels (?).
  Quiescent,
  Halt,
  Interrupt,
  Break,
  Raise,
  Fail,
  Eval,
}

pub type MaybeLogEntryRef_ = Option<LogEntryRef_>;

#[derive(Clone, Debug)]
#[repr(u8)]
pub enum LogEntryRef_ {
  Undo(UndoLogEntryRef),
}

pub type UndoLogEntryRef = Rc<UndoLogEntry_>;
pub type UndoLogEntry_ = UndoLogEntryInner_;

#[derive(Clone, Debug)]
pub struct UnifyUndoState_ {
  pub oroot:  SNum,
  pub nroot:  SNum,
  pub onext:  SNum,
  pub nprev:  SNum,
  pub otree:  Option<(LClk, SNum)>,
}

#[derive(Clone, Debug)]
pub enum UndoLogEntryInner_ {
  Unify(Box<UnifyUndoState_>),
  AllocCell(CellNum),
  LinkCells(CellNum, CellNum, CellNum, CellNum),
  LoadFunction(FunNum),
  LoadObjectCls(ObjClsNum),
  LoadObjectVal(ObjValNum),
  LoadRawSpan(SpanNum),
  LoadRawMod(ModCodeNum),
  LoadRawStm(StmCodeNum),
  LoadRawTerm(TermCodeNum),
  LoadRawIdent(IdentNum),
  LoadRawLit(LitNum),
  LoadRawLitStr(LitStrNum),
  BindIdent(IdentNum, SNum),
  BindLitStr(LitStrNum, Option<SNum>),
  RebindIdent(IdentNum, SNum, SNum),
  PutTerm(SNum),
  PutVal(SNum),
}

#[derive(Default)]
pub struct PVCache {
  leaf:     MaybeLogEntryRef_,
}

#[derive(Default)]
pub struct SlowPVCache_ {
  tree: BTreeMap<LClk, Vec<TraceEntry_>>,
}

pub type RawPVNodeId = u32;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct PVNodeId(RawPVNodeId);

pub struct PVNode_ {
}

#[derive(Default)]
pub struct FastPVCache_ {
  node: BTreeMap<PVNodeId, PVNode_>,
  next: BTreeMap<PVNodeId, PVNodeId>,
  prev: BTreeMap<PVNodeId, PVNodeId>,
}

// [Interp-API-Pub]
//
// An unhandled interpreter internal error.
#[derive(Clone, Debug)]
pub struct InterpCheck {
  _err: Error_,
}

impl<'a> From<&'a str> for InterpCheck {
  #[track_caller]
  fn from(msg: &'a str) -> InterpCheck {
    let loc = loc();
    InterpCheck{_err: Error_{loc, msg: msg.into()}}
  }
}

impl From<String> for InterpCheck {
  #[track_caller]
  fn from(msg: String) -> InterpCheck {
    let loc = loc();
    InterpCheck{_err: Error_{loc, msg: msg.into()}}
  }
}

impl Bot for InterpCheck {
  #[track_caller]
  fn bot() -> InterpCheck {
    let loc = loc();
    InterpCheck{_err: Error_{loc, msg: Default::default()}}
  }
}

impl Unimpl for InterpCheck {
  #[track_caller]
  fn unimpl() -> InterpCheck {
    let loc = loc();
    InterpCheck{_err: Error_{loc, msg: "unimpl".into()}}
  }
}

pub trait IntoInterpCheckExt: Into<InterpCheck> {
  #[track_caller]
  fn into_check(self) -> InterpCheck {
    self.into()
  }
}

impl<'a> IntoInterpCheckExt for &'a str {}
impl IntoInterpCheckExt for String {}

#[derive(Serialize, Debug)]
pub enum FlatTabled_ {
  _Top,
  // TODO: "sorted" flat cells.
  Cell,
  ModCode,
  StmCode,
  TermCode,
  IdentTerm{raw: SafeStr},
  //AtomTerm,
  NoneLitTerm,
  TrueLitTerm,
  FalseLitTerm,
  IntLitTerm(i64),
  StrLitTerm(SafeStr),
  LitTerm,
  TupleTerm{buf: Box<[ENum]>},
}

#[derive(Serialize, Debug)]
pub struct FlatSpan {
  pub prim_key: SNum,
  pub flat_val: RawSpan_,
}

#[derive(Serialize, Debug)]
pub struct FlatCode {
  pub prim_key: SNum,
  pub flat_val: FlatTabled_,
}

#[derive(Serialize, Debug)]
pub struct FlatIdent {
  pub prim_key: SNum,
  pub flat_val: RawIdent_,
}

#[derive(Serialize, Debug)]
pub struct FlatTerm {
  pub prim_key: SNum,
  pub flat_val: FlatTabled_,
}

#[derive(Serialize, Debug, Default)]
pub struct FlatEnv {
  span: Vec<FlatSpan>,
  code: Vec<FlatCode>,
  ident: Vec<FlatIdent>,
  term: Vec<FlatTerm>,
}

#[derive(Serialize, Debug)]
pub struct FlatInterp {
  clk:  LClk,
  env:  FlatEnv,
}

impl FlatInterp {
  pub fn vectorize(&self) -> String {
    serde_json::to_string_pretty(self).unwrap()
  }

  pub fn sync_testv(&self, dst: &PathBuf) {
    // TODO: logic:
    // - if there is no test vector at the destination path, then serialize
    //   ourselves w/ attestation count 0
    // - if there is a pre-existing test vector at the destination, then
    //   need to _invalidate_ it; but, might want to keep a copy of the
    //   pre-existing vector if it had nonzero attestation
  }
}

#[derive(Default)]
pub struct FastInterp {
  clkctr:   LClkCtr,
  ctr:      SNumCtr,

  env:      FastEnv_,

  reg:      FastReg_,
  exc_:     Except_,
  res_:     Result_,
  port:     Port_,
  knt_:     MemKntRef,

  log:      FastLog_,
  trace:    FastTrace_,

  clkinval: LClkInvalidSet,

  // TODO: a pv can come from multiple sources:
  // - backtrack only (w/ heuristics, branch/bound, etc.)
  // - "oracles"
  pv_cache: SlowPVCache_,

  verbose:  i8,
  parser_v: i8,
}

impl FastInterp {
  // [Interp-API-Pub]
  pub fn set_verbose(&mut self, v: i8) {
    self.verbose = v;
  }

  // [Interp-API-Pub]
  pub fn set_debug(&mut self) {
    self.set_verbose(3);
  }

  // [Interp-API-Pub]
  pub fn set_trace(&mut self) {
    self.set_verbose(7);
  }

  // [Interp-API-Pub]
  pub fn set_parser_debug(&mut self) {
    self.parser_v = 3;
  }

  // [Interp-API]
  pub fn _fresh(&self) -> SNum {
    self.ctr._fresh()
  }

  // [Interp-API]
  pub fn _peek(&self) -> SNum {
    self.ctr._get()
  }

  // [Interp-API]
  pub fn _load_raw_mod(&mut self, raw_mod: &RawMod_) -> Result<ModCodeNum, InterpCheck> {
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_mod_code();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LoadRawMod(x).into()));
    let span = self._load_raw_span(&raw_mod.span)?;
    let mut stmp: CellNum = nil();
    let mut cur_stmp: CellNum = nil();
    for raw_stm in raw_mod.body.iter() {
      let stm = self._load_raw_stm(raw_stm)?;
      let next_stmp = self._alloc_cell(stm.into());
      self._link_cells(cur_stmp, next_stmp)?;
      if stmp.is_nil() {
        stmp = next_stmp;
      }
      cur_stmp = next_stmp;
    }
    let code = ModCode_{span, stmp: stmp.into_stm_code()};
    _traceln!(self, "DEBUG: FastInterp::_load_raw_mod: x={:?} code={:?}", x, code);
    self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
    Ok(x)
  }

  // [Interp-API]
  pub fn _load_raw_stm(&mut self, raw_stm: &RawStm_) -> Result<StmCodeNum, InterpCheck> {
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_stm_code();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LoadRawStm(x).into()));
    match raw_stm {
      &RawStm_::Just(ref raw_span, ref raw_term) => {
        let span = self._load_raw_span(raw_span)?;
        let term = self._load_raw_term(raw_term)?;
        let code = StmCode_::Just{span, term};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_stm: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawStm_::Pass(ref raw_span) => {
        let span = self._load_raw_span(raw_span)?;
        let code = StmCode_::Pass{span};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_stm: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawStm_::If(ref raw_span, ref raw_cases, ref raw_final_case) => {
        let span = self._load_raw_span(raw_span)?;
        let mut cases = Vec::new();
        for &(ref raw_cond, ref raw_body) in raw_cases.iter() {
          let cond = self._load_raw_term(raw_cond)?;
          let mut body: CellNum = nil();
          let mut cur_body: CellNum = nil();
          for raw_body_stm in raw_body.iter() {
            let stm = self._load_raw_stm(raw_body_stm)?;
            let next_body = self._alloc_cell(stm.into());
            self._link_cells(cur_body, next_body)?;
            cur_body = next_body;
            if body.is_nil() {
              body = next_body;
            }
          }
          let body_stmp = body.into_stm_code();
          cases.push((cond, body_stmp));
        }
        let mut final_case = None;
        if let Some(raw_body) = raw_final_case.as_ref() {
          let mut body: CellNum = nil();
          let mut cur_body: CellNum = nil();
          for raw_body_stm in raw_body.iter() {
            let stm = self._load_raw_stm(raw_body_stm)?;
            let next_body = self._alloc_cell(stm.into());
            self._link_cells(cur_body, next_body)?;
            cur_body = next_body;
            if body.is_nil() {
              body = next_body;
            }
          }
          let body_stmp = body.into_stm_code();
          final_case = Some(body_stmp);
        }
        let final_case = final_case.try_into_nil()?;
        let code = StmCode_::If{span, cases, final_case};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_stm: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawStm_::Defproc(ref raw_span, prefix, .., ref raw_body) => {
        //_debugln!(self, "DEBUG: FastInterp::_load_raw_stm: raw span={:?} Defproc: prefix={:?}", raw_span, prefix);
        let span = self._load_raw_span(raw_span)?;
        let mut body: CellNum = nil();
        let mut cur_body: CellNum = nil();
        for raw_body_stm in raw_body.iter() {
          let stm = self._load_raw_stm(raw_body_stm)?;
          let next_body = self._alloc_cell(stm.into());
          self._link_cells(cur_body, next_body)?;
          cur_body = next_body;
          if body.is_nil() {
            body = next_body;
          }
        }
        let body_stmp = body.into_stm_code();
        let code = StmCode_::Defproc{span, body_stmp};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_stm: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        match prefix {
          None => {}
          Some(RawDefPrefix_::Rule) => {
            self.env.rule_index.insert(x.into(), ());
          }
        }
        return Ok(x);
      }
      &RawStm_::Defmatch(ref raw_span, prefix, .., ref raw_body) => {
        //_debugln!(self, "DEBUG: FastInterp::_load_raw_stm: raw span={:?} Defproc: prefix={:?}", raw_span, prefix);
        let span = self._load_raw_span(raw_span)?;
        let mut body: CellNum = nil();
        let mut cur_body: CellNum = nil();
        for raw_body_stm in raw_body.iter() {
          let stm = self._load_raw_stm(raw_body_stm)?;
          let next_body = self._alloc_cell(stm.into());
          self._link_cells(cur_body, next_body)?;
          cur_body = next_body;
          if body.is_nil() {
            body = next_body;
          }
        }
        let body_stmp = body.into_stm_code();
        let code = StmCode_::Defmatch{span, body_stmp};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_stm: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        match prefix {
          None => {}
          Some(RawDefPrefix_::Rule) => {
            self.env.rule_index.insert(x.into(), ());
          }
        }
        return Ok(x);
      }
      // FIXME: this quote _statement_ was a parsing hack.
      &RawStm_::Quote(ref raw_span, ..) => {
        let span = self._load_raw_span(raw_span)?;
        let code = StmCode_::Quote{span};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_stm: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      _ => {}
    }
    // TODO
    return Err(format!("bug: FastInterp::_load_raw_stm: unimpl: raw stm={:?}", raw_stm).into());
  }

  // [Interp-API]
  pub fn _load_raw_term(&mut self, raw_term: &RawTerm_) -> Result<TermCodeNum, InterpCheck> {
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_term_code();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LoadRawTerm(x).into()));
    match raw_term {
      &RawTerm_::Ident(ref raw_span, ref raw_id) => {
        let span = self._load_raw_span(raw_span)?;
        let id = self._load_raw_ident(raw_id)?;
        let code = TermCode_::Ident{span, id};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::QualIdent(ref raw_span, ref raw_term, ref raw_id) => {
        let span = self._load_raw_span(raw_span)?;
        let term = self._load_raw_term(raw_term)?;
        let id = self._load_raw_ident(raw_id)?;
        let code = TermCode_::QualIdent{span, term, id};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::AtomLit(ref raw_span, ref raw_lit) => {
        let span = self._load_raw_span(raw_span)?;
        let lit_str = self._load_raw_lit_str(raw_lit)?;
        let code = TermCode_::AtomLit{span, lit_str};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::IntLit(ref raw_span, ref raw_lit) => {
        let span = self._load_raw_span(raw_span)?;
        let lit_str = self._load_raw_lit_str(raw_lit)?;
        let code = TermCode_::IntLit{span, lit_str};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::BoolLit(ref raw_span, ref raw_lit) => {
        let span = self._load_raw_span(raw_span)?;
        let lit_str = self._load_raw_lit_str(raw_lit)?;
        let code = TermCode_::BoolLit{span, lit_str};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::NoneLit(ref raw_span, ref raw_lit) => {
        let span = self._load_raw_span(raw_span)?;
        let lit_str = self._load_raw_lit_str(raw_lit)?;
        let code = TermCode_::NoneLit{span, lit_str};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::ListLit(ref raw_span, ref raw_tup) => {
        // FIXME: a list should be an obj val, not just rebranded
        // tuple cells.
        let span = self._load_raw_span(raw_span)?;
        let mut tup: CellNum = nil();
        let mut cur_tup: CellNum = nil();
        for raw_tup_term in raw_tup.iter() {
          let term = self._load_raw_term(raw_tup_term)?;
          let next_tup = self._alloc_cell(term.into());
          self._link_cells(cur_tup, next_tup)?;
          cur_tup = next_tup;
          if tup.is_nil() {
            tup = next_tup;
          }
        }
        let tup = tup.into_term_code();
        let code = TermCode_::ListCon{span, tup};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::Bunch(ref raw_span, ref raw_tup) => {
        let span = self._load_raw_span(raw_span)?;
        let mut tup: CellNum = nil();
        let mut cur_tup: CellNum = nil();
        for raw_tup_term in raw_tup.iter() {
          let term = self._load_raw_term(raw_tup_term)?;
          let next_tup = self._alloc_cell(term.into());
          self._link_cells(cur_tup, next_tup)?;
          cur_tup = next_tup;
          if tup.is_nil() {
            tup = next_tup;
          }
        }
        let tup = tup.into_term_code();
        let code = TermCode_::Bunch{span, tup};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::Equal(ref raw_span, ref raw_lterm, ref raw_rterm) => {
        let span = self._load_raw_span(raw_span)?;
        let lterm = self._load_raw_term(raw_lterm)?;
        let rterm = self._load_raw_term(raw_rterm)?;
        let code = TermCode_::Equal{span, lterm, rterm};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::NEqual(ref raw_span, ref raw_lterm, ref raw_rterm) => {
        let span = self._load_raw_span(raw_span)?;
        let lterm = self._load_raw_term(raw_lterm)?;
        let rterm = self._load_raw_term(raw_rterm)?;
        let code = TermCode_::NEqual{span, lterm, rterm};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::QEqual(ref raw_span, ref raw_lterm, ref raw_rterm) => {
        let span = self._load_raw_span(raw_span)?;
        let lterm = self._load_raw_term(raw_lterm)?;
        let rterm = self._load_raw_term(raw_rterm)?;
        let code = TermCode_::QEqual{span, lterm, rterm};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::BindL(ref raw_span, ref raw_lterm, ref raw_rterm) => {
        let span = self._load_raw_span(raw_span)?;
        let lterm = self._load_raw_term(raw_lterm)?;
        let rterm = self._load_raw_term(raw_rterm)?;
        let code = TermCode_::BindL{span, lterm, rterm};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::Apply(ref raw_span, ref raw_tup) => {
        let span = self._load_raw_span(raw_span)?;
        let mut tup: CellNum = nil();
        let mut cur_tup: CellNum = nil();
        for raw_tup_term in raw_tup.iter() {
          let term = self._load_raw_term(raw_tup_term)?;
          let next_tup = self._alloc_cell(term.into());
          self._link_cells(cur_tup, next_tup)?;
          cur_tup = next_tup;
          if tup.is_nil() {
            tup = next_tup;
          }
        }
        let tup = tup.into_term_code();
        let code = TermCode_::Apply{span, tup};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::ApplyBindL(ref raw_span, ref raw_lterm, ref raw_tup) => {
        let span = self._load_raw_span(raw_span)?;
        let lterm = self._load_raw_term(raw_lterm)?;
        let mut tup: CellNum = nil();
        let mut cur_tup: CellNum = nil();
        for raw_tup_term in raw_tup.iter() {
          let term = self._load_raw_term(raw_tup_term)?;
          let next_tup = self._alloc_cell(term.into());
          self._link_cells(cur_tup, next_tup)?;
          cur_tup = next_tup;
          if tup.is_nil() {
            tup = next_tup;
          }
        }
        let tup = tup.into_term_code();
        let code = TermCode_::ApplyBindL{span, lterm, tup};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::ApplyBindR(ref raw_span, ref raw_tup, ref raw_rterm) => {
        let span = self._load_raw_span(raw_span)?;
        let mut tup: CellNum = nil();
        let mut cur_tup: CellNum = nil();
        for raw_tup_term in raw_tup.iter() {
          let term = self._load_raw_term(raw_tup_term)?;
          let next_tup = self._alloc_cell(term.into());
          self._link_cells(cur_tup, next_tup)?;
          cur_tup = next_tup;
          if tup.is_nil() {
            tup = next_tup;
          }
        }
        let tup = tup.into_term_code();
        let rterm = self._load_raw_term(raw_rterm)?;
        let code = TermCode_::ApplyBindR{span, tup, rterm};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      &RawTerm_::Effect(ref raw_span, ref raw_lterm, ref raw_rtup) => {
        let span = self._load_raw_span(raw_span)?;
        let lterm = self._load_raw_term(raw_lterm)?;
        let mut rtup: CellNum = nil();
        let mut rtup_cur: CellNum = nil();
        for raw_tup_term in raw_rtup.iter() {
          let term = self._load_raw_term(raw_tup_term)?;
          let next_tup = self._alloc_cell(term.into());
          self._link_cells(rtup_cur, next_tup)?;
          rtup_cur = next_tup;
          if rtup.is_nil() {
            rtup = next_tup;
          }
        }
        let rtup = rtup.into_term_code();
        let code = TermCode_::Effect{span, lterm, rtup};
        _traceln!(self, "DEBUG: FastInterp::_load_raw_term: x={:?} code={:?}", x, code);
        self.env.table_full[SNUM_CODE_SORT as usize].insert(x.into(), Box::new(code));
        return Ok(x);
      }
      _ => {}
    }
    // TODO
    return Err(format!("bug: FastInterp::_load_raw_term: unimpl: raw term={:?}", raw_term).into());
  }

  // [Interp-API]
  pub fn _load_raw_ident(&mut self, raw_id: &RawIdent_) -> Result<IdentNum, InterpCheck> {
    match self.env.raw_id_index.get(raw_id) {
      None => {}
      Some(&x) => {
        return Ok(x);
      }
    }
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_ident();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LoadRawIdent(x).into()));
    _traceln!(self, "DEBUG: FastInterp::_load_raw_ident: x={:?} raw ident={:?}", x, raw_id);
    self.env.table_full[SNUM_IDENT_SORT as usize].insert(x.into(), Box::new(raw_id.clone()));
    self.env.raw_id_index.insert(raw_id.clone(), x.into());
    Ok(x)
  }

  // [Interp-API]
  pub fn _load_raw_lit_str(&mut self, raw_lit_str: &RawLit_) -> Result<LitStrNum, InterpCheck> {
    match self.env.raw_lit_index.get(raw_lit_str) {
      None => {}
      Some(&x) => {
        return Ok(x);
      }
    }
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_lit_str();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LoadRawLitStr(x).into()));
    _traceln!(self, "DEBUG: FastInterp::_load_raw_lit_str: x={:?} raw lit str={:?}", x, raw_lit_str);
    self.env.table_full[SNUM_LITSTR_SORT as usize].insert(x.into(), Box::new(raw_lit_str.clone()));
    self.env.raw_lit_index.insert(raw_lit_str.clone(), x.into());
    Ok(x)
  }

  // [Interp-API]
  pub fn _load_raw_span(&mut self, raw_span: &RawSpan_) -> Result<SpanNum, InterpCheck> {
    match self.env.raw_span_index.get(raw_span) {
      None => {}
      Some(&x) => {
        return Ok(x);
      }
    }
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_span();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LoadRawSpan(x).into()));
    _traceln!(self, "DEBUG: FastInterp::_load_raw_span: x={:?} raw span={:?}", x, raw_span);
    self.env.table_full[SNUM_SPAN_SORT as usize].insert(x.into(), Box::new(raw_span.clone()));
    self.env.raw_span_index.insert(raw_span.clone(), x.into());
    Ok(x)
  }

  // [Interp-API]
  pub fn _load_function<V: Function>(&mut self, fun: V) -> Result<FunNum, InterpCheck> {
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_fun();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LoadFunction(x).into()));
    _traceln!(self, "DEBUG: FastInterp::_load_function: x={:?} fun={:?}", x, fun);
    self.env.fun_full.insert(x.into(), (Box::new(fun) as Box<dyn Function>).into());
    Ok(x)
  }

  // [Interp-API]
  pub fn _register_builtin_function<RawId: Into<RawIdent_>, V: Function>(&mut self, raw_id: RawId, cls: V) -> Result<FunNum, InterpCheck> {
    let clk = self.clkctr._get_clock();
    // FIXME: should the id belong in the "builtin" namespace?
    let id = self._load_raw_ident(&raw_id.into())?;
    let fun = self._load_function(cls)?;
    self.env.fun_name.insert(id.into(), fun.into());
    self.unify(clk, id, fun)?;
    Ok(fun)
  }

  // [Interp-API]
  pub fn _load_obj_cls<V: ObjectCls>(&mut self, cls: V) -> Result<ObjClsNum, InterpCheck> {
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_obj_cls();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LoadObjectCls(x).into()));
    _traceln!(self, "DEBUG: FastInterp::_load_object: x={:?} obj cls={:?}", x, cls);
    self.env.obj_cls_full.insert(x.into(), (Box::new(cls) as Box<dyn ObjectCls>).into());
    Ok(x)
  }

  // [Interp-API]
  pub fn _register_builtin_obj_cls<RawId: Into<RawIdent_>, V: ObjectCls>(&mut self, raw_id: RawId, cls: V) -> Result<ObjClsNum, InterpCheck> {
    let clk = self.clkctr._get_clock();
    let id = self._load_raw_ident(&raw_id.into())?;
    // FIXME: should the id belong in the "builtin" namespace?
    let obj_cls = self._load_obj_cls(cls)?;
    self.env.obj_cls_name.insert(id.into(), obj_cls.into());
    self.unify(clk, id, obj_cls)?;
    Ok(obj_cls)
  }

  // [Interp-API]
  pub fn maybe_borrow_fun(&mut self, x: FunNum) -> Result<Option<Box<dyn Function>>, InterpCheck> {
    match self.env.fun_full.get_mut(&x.into()) {
      None => {
        Ok(None)
      }
      Some(obj) => {
        match obj._borrow() {
          TransparentBox::Blk => {
            Err(format!("Function is already borrowed: x = {x:?}").into())
          }
          TransparentBox::Ptr(val) => {
            Ok(val.into())
          }
        }
      }
    }
  }

  // [Interp-API]
  pub fn borrow_fun(&mut self, x: FunNum) -> Result<Box<dyn Function>, InterpCheck> {
    match self.env.fun_full.get_mut(&x.into()) {
      None => {
        Err(format!("failed to lookup Function: x = {x:?}").into())
      }
      Some(obj) => {
        match obj._borrow() {
          TransparentBox::Blk => {
            Err(format!("Function is already borrowed: x = {x:?}").into())
          }
          TransparentBox::Ptr(val) => {
            Ok(val)
          }
        }
      }
    }
  }

  // [Interp-API]
  pub fn unborrow_fun(&mut self, x: FunNum, val: Box<dyn Function>) -> Result<(), InterpCheck> {
    match self.env.fun_full.get_mut(&x.into()) {
      None => {
        Err(format!("failed to lookup Function: x = {x:?}").into())
      }
      Some(obj) => {
        match obj._swap(val) {
          TransparentBox::Ptr(_) => {
            Err(format!("Function was not already borrowed: x = {x:?}").into())
          }
          TransparentBox::Blk => {
            Ok(())
          }
        }
      }
    }
  }

  // [Interp-API]
  pub fn _alloc_cell(&mut self, dptr: SNum) -> CellNum {
    let clk = self.clkctr._get_clock();
    let x = self._fresh().into_cell();
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::AllocCell(x).into()));
    let cel = Cell_{
      dptr,
      next: nil(),
      prev: nil(),
    };
    self.env.table_full[SNUM_CELL_SORT as usize].insert(x.into(), Box::new(cel));
    x
  }

  // [Interp-API]
  pub fn _link_cells(&mut self, lcel: CellNum, rcel: CellNum) -> Result<(), InterpCheck> {
    // NB: shortcut case for the common pattern of building a tuple-cell
    // from left-to-right.
    if lcel.is_nil() {
      return Ok(());
    }
    match self.env.table_full[SNUM_CELL_SORT as usize].get(&lcel.into())
      .and_then(|y| y.as_any().downcast_ref::<Cell_>())
    {
      None => {
        return Err("bug".into());
      }
      Some(lcel_) => {
        match self.env.table_full[SNUM_CELL_SORT as usize].get(&rcel.into())
          .and_then(|y| y.as_any().downcast_ref::<Cell_>())
        {
          None => {
            return Err("bug".into());
          }
          Some(rcel_) => {
            let clk = self.clkctr._get_clock();
            let olnext = lcel_.next.get();
            let orprev = rcel_.prev.get();
            self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::LinkCells(lcel, olnext, rcel, orprev).into()));
            lcel_.next.set(rcel);
            rcel_.prev.set(lcel);
            return Ok(());
          }
        }
      }
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn lookup_mod_code(&self, x: ModCodeNum) -> Result<ModCode_, InterpCheck> {
    match self.env.table_full[SNUM_CODE_SORT as usize].get(&x.into()) {
      None => {
        Err(format!("failed to lookup ModCode_: x = {x:?}").into())
      }
      Some(t) => {
        match t.as_any().downcast_ref::<ModCode_>() {
          None => {
            Err(format!("lookup is not a ModCode_: x = {x:?}").into())
          }
          Some(code) => {
            Ok(code.clone())
          }
        }
      }
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn lookup_stm_code(&self, x: StmCodeNum) -> Result<StmCode_, InterpCheck> {
    match self.env.table_full[SNUM_CODE_SORT as usize].get(&x.into()) {
      None => {
        Err(format!("failed to lookup StmCode_: x = {x:?}").into())
      }
      Some(t) => {
        match t.as_any().downcast_ref::<StmCode_>() {
          None => {
            Err(format!("lookup is not a StmCode_: x = {x:?}").into())
          }
          Some(code) => {
            Ok(code.clone())
          }
        }
      }
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn lookup_stm_code_cell(&self, x: StmCodeCellNum) -> Result<Cell_, InterpCheck> {
    match self.env.table_full[SNUM_CELL_SORT as usize].get(&x.into()) {
      None => {
        Err(format!("failed to lookup cell: x = {x:?}").into())
      }
      Some(t) => {
        match t.as_any().downcast_ref::<Cell_>() {
          None => {
            Err(format!("lookup is not a cell: x = {x:?}").into())
          }
          Some(cel_) => {
            // FIXME: the deref of this cell must be a stm code.
            /*if cel_.dptr ... {
              return Err(format!("lookup is not a stm code cell: x = {x:?}").into());
            }*/
            Ok(cel_.clone())
          }
        }
      }
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn lookup_term_code(&self, x: TermCodeNum) -> Result<TermCode_, InterpCheck> {
    match self.env.table_full[SNUM_CODE_SORT as usize].get(&x.into()) {
      None => {
        Err(format!("failed to lookup TermCode_: x = {x:?}").into())
      }
      Some(t) => {
        match t.as_any().downcast_ref::<TermCode_>() {
          None => {
            Err(format!("lookup is not a TermCode_: x = {x:?}").into())
          }
          Some(code) => {
            Ok(code.clone())
          }
        }
      }
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn lookup_term_code_cell(&self, x: TermCodeCellNum) -> Result<Cell_, InterpCheck> {
    match self.env.table_full[SNUM_CELL_SORT as usize].get(&x.into()) {
      None => {
        Err(format!("failed to lookup cell: x = {x:?}").into())
      }
      Some(t) => {
        match t.as_any().downcast_ref::<Cell_>() {
          None => {
            Err(format!("lookup is not a cell: x = {x:?}").into())
          }
          Some(cel_) => {
            // FIXME: the deref of this cell must be a term code.
            /*if cel_.dptr ... {
              return Err(format!("lookup is not a term code cell: x = {x:?}").into());
            }*/
            Ok(cel_.clone())
          }
        }
      }
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn lookup_raw_ident(&self, id: IdentNum) -> Result<&RawIdent_, InterpCheck> {
    match self.env.table_full[SNUM_IDENT_SORT as usize].get(&id.into()) {
      None => {
        Err(format!("failed to lookup raw ident: id = {id:?}").into())
      }
      Some(t) => {
        match t.as_any().downcast_ref::<RawIdent_>() {
          None => {
            Err(format!("lookup is not a raw ident: id = {id:?}").into())
          }
          Some(raw_id) => {
            Ok(raw_id)
          }
        }
      }
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn lookup_raw_lit_str(&self, lit_str: LitStrNum) -> Result<&RawLit_, InterpCheck> {
    match self.env.table_full[SNUM_LITSTR_SORT as usize].get(&lit_str.into()) {
      None => {
        Err(format!("failed to lookup raw literal: lit str = {lit_str:?}").into())
      }
      Some(t) => {
        match t.as_any().downcast_ref::<RawLit_>() {
          None => {
            Err(format!("lookup is not a raw literal: lit str = {lit_str:?}").into())
          }
          Some(lit_) => {
            Ok(lit_)
          }
        }
      }
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn find<K: Into<SNum>>(&self, clk: LClk, query: K) -> Result<ENum, InterpCheck> {
    self.env.unifier._find(&self.clkinval, clk, query.into()).map_err(|e| e.into())
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn unify<LK: Into<SNum>, RK: Into<SNum>>(&mut self, clk: LClk, lquery: LK, rquery: RK) -> Result<SNum, InterpCheck> {
    self.env.unifier._unify(&mut self.log, &self.clkinval, clk, lquery.into(), rquery.into()).map_err(|e| e.into())
  }

  // [Interp-API]: This is part of the interpreter private API.
  pub fn get_vals(&self, clk: LClk, query: ENum) -> Result<Vec<(ENum, LitVal_)>, InterpCheck> {
    let keys = self.env.unifier._findall(&self.clkinval, clk, query.inst).map_err(|e| e.into_check())?;
    _debugln!(self, "DEBUG: FastInterp::get_vals: query={:?} keys={:?}", query, keys);
    let mut vals = Vec::new();
    for &key in keys.iter() {
      if key.cls != query.cls {
        _debugln!(self, "DEBUG: FastInterp::get_vals: cls mismatch: query={:?} key={:?}", query, key);
        return Err(bot());
      }
      // NB: invariant: table lookups go through the ENum _instance_.
      match self.env.table_full[SNUM_VAL_SORT as usize].get(&key.inst) {
        None => {
          _debugln!(self, "DEBUG: FastInterp::get_vals: not a val: query={:?} key={:?}", query, key);
        }
        Some(entry) => {
          if let Some(val) = entry.as_any().downcast_ref::<LitVal_>() {
            vals.push((key, val.clone()));
          } else {
            _debugln!(self, "DEBUG: FastInterp::get_vals: not an LitVal_: query={:?} key={:?}", query, key);
          }
        }
      }
    }
    Ok(vals)
  }

  // [Interp-API]: This is part of the interpreter private API.
  pub fn put_term<K: Into<SNum>, V: Tabled>(&mut self, clk: LClk, key: K, term: V) -> Result<(), InterpCheck> {
    let x = key.into();
    _traceln!(self, "DEBUG: FastInterp::put_term: clk={:?} x={:?} term={:?}", clk, x, term);
    self.env.table_full[SNUM_TERM_SORT as usize].insert(x, Box::new(term));
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::PutTerm(x).into()));
    Ok(())
  }

  // [Interp-API]: This is part of the interpreter private API.
  pub fn put_val<K: Into<SNum>, V: Tabled>(&mut self, clk: LClk, key: K, val: V) -> Result<(), InterpCheck> {
    let x = key.into();
    self.env.table_full[SNUM_VAL_SORT as usize].insert(x, Box::new(val));
    self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::PutVal(x).into()));
    Ok(())
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn reset_res(&mut self) -> Result<(), InterpCheck> {
    self.res_.reset();
    Ok(())
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn put_res<K: Into<SNum>>(&mut self, key: K) -> Result<(), InterpCheck> {
    let x = key.into();
    match self.res_.put(x) {
      ResReg_::Emp => Ok(()),
      ResReg_::Key(y) => {
        Err(format!("already filled result register: x = {:?} y = {:?}", x, y).into())
      }
      _ => unimplemented!()
    }
  }

  // [Interp-API]: This is part of the interpreter private API.
  #[track_caller]
  pub fn get_res(&mut self) -> Result<SNum, InterpCheck> {
    match self.res_.get() {
      ResReg_::Emp => {
        Err(format!("expected result register").into())
      }
      ResReg_::Key(x) => Ok(x),
      _ => unimplemented!()
    }
  }

  pub fn _undo(&mut self, clk: LClk, entry: UndoLogEntryRef) -> Result<(), InterpCheck> {
    match &*entry {
      &UndoLogEntry_::Unify(ref state) => {
        self.env.unifier._link(state.oroot, state.onext);
        self.env.unifier._link(state.nprev, state.nroot);
        self.env.unifier.root.remove(&state.nroot);
        self.env.unifier.root.insert(state.oroot);
        if let Some(otree) = state.otree {
          self.env.unifier.tree.insert(state.oroot, otree);
        } else {
          self.env.unifier.tree.remove(&state.oroot);
        }
      }
      &UndoLogEntry_::BindIdent(id, prev_x) => {
        let raw_id = self.lookup_raw_ident(id)?.clone();
        if prev_x.is_nil() {
          self.env.raw_id_bind.remove(&raw_id);
        } else {
          self.env.raw_id_bind.insert(raw_id, prev_x.into());
        }
      }
      &UndoLogEntry_::PutTerm(x) => {
        if self.env.table_full[SNUM_TERM_SORT as usize].remove(&x).is_none() {
          _debugln!(self, "DEBUG: FastInterp::_undo: PutTerm x={:?} nonexist", x);
          return Err(bot());
        }
      }
      &UndoLogEntry_::PutVal(x) => {
        if self.env.table_full[SNUM_VAL_SORT as usize].remove(&x).is_none() {
          _debugln!(self, "DEBUG: FastInterp::_undo: PutVal x={:?} nonexist", x);
          return Err(bot());
        }
      }
      e => return Err(format!("_undo: unimpl: clk={:?} e={:?}", clk, e).into())
    }
    Ok(())
  }

  // [Interp-API-Pub]
  pub fn flatten_(&self) -> FlatInterp {
    let clk = self.clkctr._get_clock();
    let mut interp = FlatInterp{
      clk,
      env:  FlatEnv::default(),
    };
    // FIXME: sort flattened env sections (table_full is FxHashMap).
    for (&prim_key, entry) in self.env.table_full[SNUM_CODE_SORT as usize].iter() {
      let flat_val = if let Some(ref code) = entry.as_any().downcast_ref::<ModCode_>() {
        FlatTabled_::ModCode
      } else if let Some(ref code) = entry.as_any().downcast_ref::<StmCode_>() {
        FlatTabled_::StmCode
      } else if let Some(ref code) = entry.as_any().downcast_ref::<TermCode_>() {
        FlatTabled_::TermCode
      } else {
        panic!("bug")
      };
      interp.env.code.push(FlatCode{prim_key, flat_val});
    }
    interp.env.code.sort_by_key(|e| e.prim_key);
    for (&prim_key, entry) in self.env.table_full[SNUM_IDENT_SORT as usize].iter() {
      let flat_val = if let Some(val) = entry.as_any().downcast_ref::<RawIdent_>() {
        //FlatTabled_::RawIdent(val.clone())
        val.clone()
      } else {
        panic!("bug")
      };
      interp.env.ident.push(FlatIdent{prim_key, flat_val});
    }
    for (&prim_key, entry) in self.env.table_full[SNUM_TERM_SORT as usize].iter() {
      let flat_val = if let Some(_) = entry.as_any().downcast_ref::<Cell_>() {
        FlatTabled_::Cell
      } else if let Some(ref t) = entry.as_any().downcast_ref::<IdentTerm_>() {
        FlatTabled_::IdentTerm{raw: t.raw.clone()}
      } else if let Some(ref t) = entry.as_any().downcast_ref::<LitTerm_>() {
        match t._unpack() {
          UnpackedLitTerm_::None => {
            FlatTabled_::NoneLitTerm
          }
          UnpackedLitTerm_::True => {
            FlatTabled_::TrueLitTerm
          }
          UnpackedLitTerm_::False => {
            FlatTabled_::FalseLitTerm
          }
          UnpackedLitTerm_::Int(x) => {
            FlatTabled_::IntLitTerm(*x)
          }
          UnpackedLitTerm_::Str(x) => {
            FlatTabled_::StrLitTerm((*x).clone())
          }
        }
      } else if let Some(ref t) = entry.as_any().downcast_ref::<TupleTerm_>() {
        FlatTabled_::TupleTerm{buf: t.buf.clone()}
      } else if let Some(_) = entry.as_any().downcast_ref::<ModCode_>() {
        FlatTabled_::ModCode
      } else if let Some(_) = entry.as_any().downcast_ref::<StmCode_>() {
        FlatTabled_::StmCode
      } else if let Some(_) = entry.as_any().downcast_ref::<TermCode_>() {
        FlatTabled_::TermCode
      } else {
        FlatTabled_::_Top
      };
      interp.env.term.push(FlatTerm{prim_key, flat_val});
    }
    // FIXME: flatten is missing lots of state.
    interp
  }

  // [Interp-API-Pub]
  //
  // This pre-initializes the interpreter with the builtin prelude.
  pub fn pre_init(&mut self) -> Result<(), InterpCheck> {
    _debugln!(self, "DEBUG: FastInterp::pre_init: ...");
    self.env._pre_init(&self.ctr);
    self._register_builtin_function("choice",   self::prelude::ChoiceFun::default())?;
    self._register_builtin_function("failure",  self::prelude::FailureFun::default())?;
    self._register_builtin_function("eval",     self::prelude::EvalFun::default())?;
    self._register_builtin_function("print",    self::prelude::PrintFun::default())?;
    self._register_builtin_obj_cls("TokenTrie", self::prelude::TokenTrieCls::default())?;
    _debugln!(self, "DEBUG: FastInterp::pre_init: done");
    Ok(())
  }

  // [Interp-API-Pub]
  //
  // This loads the given source code into the interpreter.
  pub fn load_(&mut self, src: &str) -> Result<(), InterpCheck> {
    _debugln!(self, "DEBUG: FastInterp::load_: ...");
    if _debugln!(self, "DEBUG: FastInterp::load_: parse...") {
    }
    let mut parser = FastParser::new(src);
    if self.parser_v > 0 {
      parser.set_verbose(self.parser_v);
    }
    let y = parser.mod_().map_err(|e| format!("parse error: {:?}", e))?;
    if _debugln!(self, "DEBUG: FastInterp::load_: pretty print...") {
      let printer = DebugPrinter::new(src);
      printer.pretty_print(&y);
    }
    _debugln!(self, "DEBUG: FastInterp::load_: load...");
    let x = self._load_raw_mod(&y)?;
    drop(parser);
    let clk = self.clkctr._get_clock();
    self.knt_ = MemKnt{
      clk,
      cur:  MemKnt_::InterpMod(x, ModCodeInterpState_::fresh()),
      prev: nil(),
    }.into_ref();
    self.port = Port_::Enter;
    _debugln!(self, "DEBUG: FastInterp::load_: done");
    Ok(())
  }

  // [Interp-API-Pub]
  //
  // This re-initializes the interpreter with the new source code.
  pub fn reload_(&mut self, src: &str) -> Result<(), InterpCheck> {
    _debugln!(self, "DEBUG: FastInterp::reload_: ...");
    unimplemented!();
  }

  // [Interp-API-Pub]
  pub fn resync_(&mut self, ) -> Result<(), InterpCheck> {
    _debugln!(self, "DEBUG: FastInterp::resync_: ...");
    unimplemented!();
  }

  pub fn _debug_print_interp(&self) -> Result<(), InterpCheck> {
    unimplemented!();
  }

  // [Interp-API-Pub]
  pub fn interp_(&mut self) -> Result<Yield_, InterpCheck> {
    //let mut ictr = 0;
    'resume: loop {
      //ictr += 1;
      _debugln!(self, "DEBUG: FastInterp::interp_: clk={:?} resume", self.clkctr._get_clock());
      /*if ictr >= 100 {
        _debugln!(self, "DEBUG: FastInterp::interp_:   breakpoint (timeout)");
        return Ok(Yield_::Break);
      }*/
      let yield_ = self.resume_()?;

      // NB: at this point, ^ has yielded for some reason.
      match yield_ {
        Yield_::Quiescent |
        Yield_::Halt |
        Yield_::Interrupt |
        Yield_::Break |
        Yield_::Raise => {
          let clk = self.clkctr._get_clock();
          match yield_ {
            Yield_::Quiescent => {
              _debugln!(self, "DEBUG: FastInterp::interp_: clk={:?} quiescent", clk);
            }
            Yield_::Halt => {
              _debugln!(self, "DEBUG: FastInterp::interp_: clk={:?} halt", clk);
            }
            Yield_::Interrupt => {
              _debugln!(self, "DEBUG: FastInterp::interp_: clk={:?} interrupt", clk);
            }
            Yield_::Break => {
              _debugln!(self, "DEBUG: FastInterp::interp_: clk={:?} break", clk);
            }
            Yield_::Raise => {
              _debugln!(self, "DEBUG: FastInterp::interp_: clk={:?} raise: except={:?}", clk, self.exc_);
            }
            _ => {}
          }
          _debugln!(self, "DEBUG: FastInterp::interp_: env:  id   tab={:?}",
              &self.env.table_full[SNUM_IDENT_SORT as usize]);
          _debugln!(self, "DEBUG: FastInterp::interp_:       id   idx={:?}",
              &self.env.raw_id_index);
          _debugln!(self, "DEBUG: FastInterp::interp_:       fun  nom={:?}",
              &self.env.fun_name);
          _debugln!(self, "DEBUG: FastInterp::interp_:       fun  tab={:?}",
              &self.env.fun_full);
          _debugln!(self, "DEBUG: FastInterp::interp_:       ocls nom={:?}",
              &self.env.obj_cls_name);
          _debugln!(self, "DEBUG: FastInterp::interp_:       ocls tab={:?}",
              &self.env.obj_cls_full);
          _debugln!(self, "DEBUG: FastInterp::interp_:       term tab={:?}",
              &self.env.table_full[SNUM_TERM_SORT as usize]);
          _debugln!(self, "DEBUG: FastInterp::interp_:       val  tab={:?}",
              &self.env.table_full[SNUM_VAL_SORT as usize]);
          for &x in self.env.unifier.root.iter() {
            _debugln!(self, "DEBUG: FastInterp::interp_:       unifier[{:?}]={:?}", x, self.env.unifier._findall(&self.clkinval, clk, x)?);
          }
          _debugln!(self, "DEBUG: FastInterp::interp_:       rule idx={:?}",
              &self.env.rule_index);
          _debugln!(self, "DEBUG: FastInterp::interp_: res:  log={:?} reg={:?}",
              &self.res_._log[min(1, self.res_._log.len()) .. ],
              self.res_.reg);
          for p in 0 .. self.log.buf.len() {
            _debugln!(self, "DEBUG: FastInterp::interp_: log:  p={} e={:?}",
                p, &self.log.buf[p]);
          }
          return Ok(yield_);
        }
        Yield_::Fail => {
          let clk = self.clkctr._get_clock();
          _debugln!(self, "DEBUG: FastInterp::interp_: yield: clk={:?} failure", clk);
          _debugln!(self, "DEBUG: FastInterp::interp_: yield:   trace.buf.len={}", self.trace.buf.len());
          for p in (0 .. self.trace.buf.len()).rev() {
            self.trace.buf[p].xctr += 1;
            // NB: in this case, still undo, but also continue to backtrack.
            let stop = self.trace.buf[p].xctr < self.trace.buf[p].xlim;
            let rst_clk = self.trace.buf[p].root_clk;
            let rst_xlb = self.trace.buf[p].xlb;
            _debugln!(self, "DEBUG: FastInterp::interp_: yield:   trace.buf[{}]: choice ctr={} ub={} rst clk={:?} xlb={:?}", p, self.trace.buf[p].xctr, self.trace.buf[p].xlim, rst_clk, rst_xlb);
            for logp in (0 .. self.log.buf.len()).rev() {
              if self.log.buf[logp].clk < rst_clk {
                _debugln!(self, "DEBUG: FastInterp::interp_: yield:   trace.buf[{}]: undo[{}]: clk={:?} stop", p, logp, self.log.buf[logp].clk);
                break;
              }
              match &self.log.buf[logp].val {
                &LogEntryRef_::Undo(ref e) => {
                  _debugln!(self, "DEBUG: FastInterp::interp_: yield:   trace.buf[{}]: undo[{}]: clk={:?} entry={:?}", p, logp, self.log.buf[logp].clk, e);
                  self._undo(self.log.buf[logp].clk, e.clone())?;
                }
                _ => return Err(bot())
              }
              let _ = self.log.buf.pop().unwrap();
            }
            self.ctr._reset(rst_xlb);
            // NB: restoring a choice point _should not_ reset linear time!
            // instead, allocate a fresh timestamp next step.
            // (this is the whole point of a _linear_ timestamp.)
            /*self.clkctr._reset_clock(rst_clk);*/
            self.reg.rst_clk = rst_clk;
            self.exc_ = self.trace.buf[p].ctl_.exc_.clone();
            self.res_ = self.trace.buf[p].ctl_.res_.clone();
            self.port = self.trace.buf[p].ctl_.port.clone();
            // FIXME: would prefer to take instead of clone; do we really
            // need to store full trace entries (in the PV cache)?
            let knt = self.trace.buf[p].knt_.clone();
            _debugln!(self, "DEBUG: FastInterp::interp_: yield:   knt={:?}", knt);
            if knt.is_some() {
              let knt = knt.unwrap();
              _traceln!(self, "DEBUG: FastInterp::interp_: yield:     kcur ={:?} {:?}", knt.clk, &knt.cur);
              let mut kprev = knt.prev.as_ref();
              loop {
                if kprev.is_none() {
                  _traceln!(self, "DEBUG: FastInterp::interp_: yield:     kprev=nil");
                  break;
                }
                let knt = kprev.unwrap();
                _traceln!(self, "DEBUG: FastInterp::interp_: yield:     kprev={:?} {:?}", knt.clk, &knt.cur);
                kprev = knt.prev.as_ref();
              }
            }
            self.knt_ = self.trace.buf[p].knt_.clone();
            // FIXME: only update the pv cache if this actually a new best pv.
            /*self.pv_cache.tree.insert(clk, self.trace.buf.clone());*/
            if stop {
              _debugln!(self, "DEBUG: FastInterp::interp_: yield:   stop: p={}", p);
              continue 'resume;
            }
            _debugln!(self, "DEBUG: FastInterp::interp_: yield:   pop: p={}", p);
            self.trace._pop_pos(p as _)?;
          }
          _debugln!(self, "DEBUG: FastInterp::interp_: yield:   halt");
          _debugln!(self, "DEBUG: FastInterp::interp_: env:  id   tab={:?}",
              &self.env.table_full[SNUM_IDENT_SORT as usize]);
          _debugln!(self, "DEBUG: FastInterp::interp_:       id   idx={:?}",
              &self.env.raw_id_index);
          _debugln!(self, "DEBUG: FastInterp::interp_:       fun  nom={:?}",
              &self.env.fun_name);
          _debugln!(self, "DEBUG: FastInterp::interp_:       fun  tab={:?}",
              &self.env.fun_full);
          _debugln!(self, "DEBUG: FastInterp::interp_:       ocls nom={:?}",
              &self.env.obj_cls_name);
          _debugln!(self, "DEBUG: FastInterp::interp_:       ocls tab={:?}",
              &self.env.obj_cls_full);
          _debugln!(self, "DEBUG: FastInterp::interp_:       term tab={:?}",
              &self.env.table_full[SNUM_TERM_SORT as usize]);
          _debugln!(self, "DEBUG: FastInterp::interp_:       val  tab={:?}",
              &self.env.table_full[SNUM_VAL_SORT as usize]);
          for &x in self.env.unifier.root.iter() {
            _debugln!(self, "DEBUG: FastInterp::interp_:       unifier[{:?}]={:?}", x, self.env.unifier._findall(&self.clkinval, clk, x)?);
          }
          _debugln!(self, "DEBUG: FastInterp::interp_:       rule idx={:?}",
              &self.env.rule_index);
          _debugln!(self, "DEBUG: FastInterp::interp_: res:  log={:?} reg={:?}",
              &self.res_._log[min(1, self.res_._log.len()) .. ],
              self.res_.reg);
          for p in 0 .. self.log.buf.len() {
            _debugln!(self, "DEBUG: FastInterp::interp_: log:  p={} e={:?}",
                p, &self.log.buf[p]);
          }
          return Ok(Yield_::Halt);
        }
        Yield_::Eval => {
          let clk = self.clkctr._get_clock();
          _debugln!(self, "DEBUG: FastInterp::interp_: yield: clk={:?} eval", clk);
          _debugln!(self, "DEBUG: FastInterp::interp_: yield:   trace.buf.len={}", self.trace.buf.len());
          return Err(unimpl());
        }
      }
    }
  }

  // [Interp-API]
  pub fn resume_(&mut self) -> Result<Yield_, InterpCheck> {
    _traceln!(self, "DEBUG: FastInterp::resume_: ...");
    loop {
      if self.exc_.is_some() {
        /*self.port = Port_::Except;*/
        return Ok(Yield_::Raise);
      }
      let knt = self.knt_.take();
      if knt.is_none() {
        self.port = Port_::Quiescent;
        return Ok(Yield_::Quiescent);
      }
      let clk = self.clkctr._fresh_clock();
      let xlb = self._peek();
      self.reg.xlb = xlb;
      let mut knt = knt.unwrap();
      _traceln!(self, "DEBUG: FastInterp::resume_: ctl:  clk={:?} xlb={:?} port={:?} res={:?}",
          clk, xlb, self.port, self.res_.peek());
      _traceln!(self, "DEBUG: FastInterp::resume_:       kcur ={:?} {:?}", knt.clk, &knt.cur);
      {
        let mut kprev = knt.prev.as_ref();
        loop {
          if kprev.is_none() {
            _traceln!(self, "DEBUG: FastInterp::resume_:       kprev=nil");
            break;
          }
          let knt = kprev.unwrap();
          _traceln!(self, "DEBUG: FastInterp::resume_:       kprev={:?} {:?}", knt.clk, &knt.cur);
          kprev = knt.prev.as_ref();
        }
      }
      _traceln!(self, "DEBUG: FastInterp::resume_: env:  id   tab={:?}", &self.env.table_full[SNUM_IDENT_SORT as usize]);
      _traceln!(self, "DEBUG: FastInterp::resume_:       id   idx={:?}", &self.env.raw_id_index);
      _traceln!(self, "DEBUG: FastInterp::resume_:       fun  nom={:?}",
          &self.env.fun_name);
      _traceln!(self, "DEBUG: FastInterp::resume_:       fun  tab={:?}",
          &self.env.fun_full);
      _traceln!(self, "DEBUG: FastInterp::resume_:       ocls nom={:?}",
          &self.env.obj_cls_name);
      _traceln!(self, "DEBUG: FastInterp::resume_:       ocls tab={:?}",
          &self.env.obj_cls_full);
      _traceln!(self, "DEBUG: FastInterp::resume_:       term tab={:?}", &self.env.table_full[SNUM_TERM_SORT as usize]);
      _traceln!(self, "DEBUG: FastInterp::resume_:       val  tab={:?}", &self.env.table_full[SNUM_VAL_SORT as usize]);
      for &x in self.env.unifier.root.iter() {
        _traceln!(self, "DEBUG: FastInterp::resume_:       unifier[{:?}]={:?}", x, self.env.unifier._findall(&self.clkinval, clk, x)?);
      }
      match (self.port, &mut knt.cur) {
        (Port_::Enter, &mut MemKnt_::InterpMod(cur_mod_code, ref mut state)) => {
          _traceln!(self, "DEBUG: FastInterp::resume_:   Enter  InterpMod: state.stmp={:?} (prev)", state.stmp);
          state.stmp = self.lookup_mod_code(cur_mod_code)?.stmp;
          _traceln!(self, "DEBUG: FastInterp::resume_:   Enter  InterpMod: state.stmp={:?} (next)", state.stmp);
          if state.stmp.is_nil() {
            self.knt_ = nil();
            self.port = Port_::Quiescent;
            return Ok(Yield_::Quiescent);
          }
          let next_stmp = state.stmp;
          self.knt_ = MemKnt{
            clk,
            prev: knt.into(),
            cur:  MemKnt_::InterpStmp(next_stmp, StmCodeCellInterpState_::fresh(next_stmp)),
          }.into_ref();
          /*self.port = Port_::Enter;*/
        }
        (Port_::Return, &mut MemKnt_::InterpMod(_cur_mod_code, ref mut state)) => {
          self.knt_ = nil();
          self.port = Port_::Quiescent;
          return Ok(Yield_::Quiescent);
        }
        (Port_::Enter, &mut MemKnt_::InterpStmp(_cur_stm_code_ptr, ref mut state)) => {
          if state.stmp.is_nil() {
            self.knt_ = knt.prev;
            self.port = Port_::Return;
          } else {
            let stmp_ = self.lookup_stm_code_cell(state.stmp)?;
            let stm = stmp_.dptr.into_stm_code();
            state.stmp = stmp_.next.get().into_stm_code();
            self.knt_ = MemKnt{
              clk,
              prev: knt.into(),
              cur:  MemKnt_::InterpStm(stm, StmCodeInterpState_::fresh()),
            }.into_ref();
            /*self.port = Port_::Enter;*/
          }
        }
        (Port_::Return, &mut MemKnt_::InterpStmp(_cur_stm_code_ptr, ref mut state)) => {
          self.knt_ = knt.into();
          self.port = Port_::Enter;
        }
        (Port_::Enter, &mut MemKnt_::InterpStm(cur_stm_code, ref mut state)) => {
          let cur_stm_code_ = self.lookup_stm_code(cur_stm_code)?;
          match cur_stm_code_ {
            StmCode_::Just{span, term} => {
              self.reset_res()?;
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpTerm(term, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            // FIXME: this quote _statement_ was a parsing hack.
            StmCode_::Quote{..} => {
              self.port = Port_::Return;
            }
            StmCode_::Pass{..} => {
              self.port = Port_::Return;
            }
            StmCode_::If{span, cases, final_case} => {
              _debugln!(self, "DEBUG: FastInterp::resume_:   Enter  InterpStm: If: ");
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpIfStm(
                    cur_stm_code,
                    IfStmCodeInterpState_::fresh(cases, final_case)
                ),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            StmCode_::Defproc{..} => {
              // FIXME
              self.port = Port_::Return;
            }
            StmCode_::Defmatch{..} => {
              // FIXME
              self.port = Port_::Return;
            }
            cur_stm_code_ => {
              return Err(format!("{:?}", cur_stm_code_).into());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpIfStm(_cur_stm_code, ref mut state)) => {
          match state.cur {
            IfStmCodeInterpCursor_::Cond(cond_term, ..) => {
              let save_tctx = self.reg.tctx;
              self.reg.tctx = TermContext_::Match;
              if state.save_tctx.push(save_tctx).is_some() {
                return Err(bot());
              }
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(cond_term, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            IfStmCodeInterpCursor_::Body(body_stmp, ..) => {
              let save_tctx = state.save_tctx.pop();
              if save_tctx.is_none() {
                return Err(bot());
              }
              self.reg.tctx = save_tctx.unwrap();
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpStmp(body_stmp, StmCodeCellInterpState_::fresh(body_stmp)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            IfStmCodeInterpCursor_::FinalBody(final_body_stmp) => {
              if state.save_tctx.is_some() {
                return Err(bot());
              }
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpStmp(final_body_stmp, StmCodeCellInterpState_::fresh(final_body_stmp)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            IfStmCodeInterpCursor_::Fin => {
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpIfStm(_cur_stm_code, ref mut state)) => {
          match &mut state.cur {
            &mut IfStmCodeInterpCursor_::Cond(cond, body, case_idx, ref cases, final_body) => {
              let cases = cases.clone();
              state.cur = IfStmCodeInterpCursor_::Body(body, case_idx, cases, final_body);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            &mut IfStmCodeInterpCursor_::Body(body, case_idx, ref cases, final_body) => {
              if case_idx > cases.len() {
                return Err(bot());
              } else if case_idx == cases.len() {
                if final_body.is_nil() {
                  state.cur = IfStmCodeInterpCursor_::Fin;
                } else {
                  state.cur = IfStmCodeInterpCursor_::FinalBody(final_body);
                }
              } else {
                let cond = cases[case_idx].0;
                let body = cases[case_idx].1;
                let case_idx = case_idx + 1;
                let cases = cases.clone();
                state.cur = IfStmCodeInterpCursor_::Cond(cond, body, case_idx, cases, final_body);
              }
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            &mut IfStmCodeInterpCursor_::FinalBody(final_body) => {
              state.cur = IfStmCodeInterpCursor_::Fin;
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            &mut IfStmCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpStm(cur_stm, ref mut state)) => {
          self.knt_ = knt.prev;
          /*self.port = Port_::Return;*/
        }
        (Port_::Enter, &mut MemKnt_::InterpTerm(cur_term_code, ref mut state)) => {
          let cur_term_code_ = self.lookup_term_code(cur_term_code)?;
          match cur_term_code_ {
            TermCode_::Ident{span, id} => {
              let raw_id = self.lookup_raw_ident(id)?.clone();
              match self.env.raw_id_bind.get(&raw_id) {
                Some(&x) => {
                  self.put_res(x)?;
                  self.knt_ = knt.prev;
                  self.port = Port_::Return;
                }
                None => {
                  let x = self._fresh().into_term();
                  let term_ = IdentTerm_{raw: raw_id.clone()};
                  self.put_term(clk, x, term_)?;
                  let prev_x = self.env.raw_id_bind.insert(raw_id.clone(), x.into()).try_into_nil()
                      .map_err(|_| format!("nil-bound ident: id = {:?} raw id = {:?}", id, raw_id).into_check())?;
                  self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::BindIdent(id, prev_x).into()));
                  self.put_res(x)?;
                  self.knt_ = knt.prev;
                  self.port = Port_::Return;
                }
              }
            }
            TermCode_::QualIdent{span, term, id} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpQualIdentTerm(cur_term_code, QualIdentTermCodeInterpState_::fresh(term, id)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::AtomLit{span, lit_str} => {
              let raw_lit = self.lookup_raw_lit_str(lit_str)?;
              // FIXME: should unquote raw string literals during parsing;
              // i.e. want to avoid having to parse this literal string twice.
              let inner_val = raw_lit.clone();
              let lit_term_ = LitTerm_::new_str(inner_val.clone());
              let x = match self.env.lit_term_bind.get(&lit_term_) {
                None => {
                  let x = self._fresh();
                  self.put_term(clk, x, lit_term_.clone())?;
                  let y = self._fresh();
                  let val = LitVal_::Atom(inner_val);
                  self.put_val(clk, y, val)?;
                  self.unify(clk, x, y)?;
                  let prev_x = self.env.lit_term_bind.insert(lit_term_, x.into());
                  self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::BindLitStr(lit_str, prev_x).into()));
                  x
                }
                Some(&x) => x
              };
              self.put_res(x)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
            TermCode_::IntLit{span, lit_str} => {
              let raw_lit = self.lookup_raw_lit_str(lit_str)?;
              let inner_val = i64::from_str(raw_lit.as_raw_str()).map_err(|_| format!("not an int: {:?}", raw_lit))?;
              let lit_term_ = LitTerm_::new_int(inner_val);
              let x = match self.env.lit_term_bind.get(&lit_term_) {
                None => {
                  let x = self._fresh();
                  self.put_term(clk, x, lit_term_.clone())?;
                  let y = self._fresh();
                  let val = LitVal_::Int(inner_val);
                  self.put_val(clk, y, val)?;
                  self.unify(clk, x, y)?;
                  let prev_x = self.env.lit_term_bind.insert(lit_term_, x.into());
                  self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::BindLitStr(lit_str, prev_x).into()));
                  x
                }
                Some(&x) => x
              };
              self.put_res(x)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
            TermCode_::BoolLit{span, lit_str} => {
              let raw_lit = self.lookup_raw_lit_str(lit_str)?;
              let inner_val = match raw_lit.as_raw_str() {
                "True" => true,
                "False" => false,
                _ => {
                  return Err(bot());
                }
              };
              let lit_term_ = LitTerm_::new_bool(inner_val);
              let x = match self.env.lit_term_bind.get(&lit_term_) {
                None => {
                  let x = self._fresh();
                  self.put_term(clk, x, lit_term_.clone())?;
                  let y = self._fresh();
                  let val = LitVal_::Bool(inner_val);
                  self.put_val(clk, y, val)?;
                  self.unify(clk, x, y)?;
                  let prev_x = self.env.lit_term_bind.insert(lit_term_, x.into());
                  self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::BindLitStr(lit_str, prev_x).into()));
                  x
                }
                Some(&x) => x
              };
              self.put_res(x)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
            TermCode_::NoneLit{span, lit_str} => {
              let raw_lit = self.lookup_raw_lit_str(lit_str)?;
              match raw_lit.as_raw_str() {
                "None" => {}
                _ => {
                  return Err(bot());
                }
              }
              let lit_term_ = LitTerm_::new_none();
              let x = match self.env.lit_term_bind.get(&lit_term_) {
                None => {
                  let x = self._fresh();
                  self.put_term(clk, x, lit_term_.clone())?;
                  let y = self._fresh();
                  let val = LitVal_::None;
                  self.put_val(clk, y, val)?;
                  self.unify(clk, x, y)?;
                  let prev_x = self.env.lit_term_bind.insert(lit_term_, x.into());
                  self.log._append(clk, LogEntryRef_::Undo(UndoLogEntry_::BindLitStr(lit_str, prev_x).into()));
                  x
                }
                Some(&x) => x
              };
              self.put_res(x)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
            TermCode_::ListCon{span, tup} => {
              let x = self._fresh();
              // FIXME: build actual list obj val from this constructor.
              let obj = LitVal_::List{buf: Vec::new()};
              _traceln!(self, "DEBUG: FastInterp::resume_:   Enter  ListCon: fresh obj val = {:?}", x);
              self.env.table_full[SNUM_VAL_SORT as usize].insert(x.into(), Box::new(obj));
              self.put_res(x)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
            TermCode_::Group{span, term} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpTerm(term, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::Bunch{span, tup} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpBunchTerm(cur_term_code, BunchTermCodeInterpState_::fresh(tup)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::Equal{span, lterm, rterm} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpEqualTerm(cur_term_code, EqualTermCodeInterpState_::fresh(lterm, rterm)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::NEqual{span, lterm, rterm} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpNEqualTerm(cur_term_code, NEqualTermCodeInterpState_::fresh(lterm, rterm)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::QEqual{span, lterm, rterm} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpQEqualTerm(cur_term_code, QEqualTermCodeInterpState_::fresh(lterm, rterm)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::Apply{span, tup} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpApplyTerm(cur_term_code, ApplyTermCodeInterpState_::fresh(tup)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::ApplyBindL{span, lterm, tup} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpApplyBindLTerm(cur_term_code, ApplyBindLTermCodeInterpState_::fresh(lterm, tup)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::ApplyBindR{span, tup, rterm} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpApplyBindRTerm(cur_term_code, ApplyBindRTermCodeInterpState_::fresh(tup, rterm)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            TermCode_::BindL{span, lterm, rterm} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpBindLTerm(cur_term_code, BindLTermCodeInterpState_::fresh(lterm, rterm)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            /*TermCode_::RebindL{span, lterm, rterm} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpRebindLTerm(cur_term_code, RebindLTermCodeInterpState_::fresh(lterm, rterm)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }*/
            TermCode_::Effect{span, lterm, rtup} => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.prev,
                cur:  MemKnt_::InterpEffectTerm(cur_term_code, EffectTermCodeInterpState_::fresh(lterm, rtup)),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            cur_term_code_ => {
              return Err(format!("unimpl: term code = {:?}", cur_term_code_).into());
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpTerm(_cur_term, ref mut _state)) => {
          self.knt_ = knt.prev;
          /*self.port = Port_::Return;*/
        }
        (Port_::Enter, &mut MemKnt_::InterpQualIdentTerm(_cur_term, ref mut state)) => {
          match state.cur {
            QualIdentTermCodeInterpCursor_::Term(term, _ident) => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(term, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            QualIdentTermCodeInterpCursor_::Ident(ident) => {
              return Err(unimpl());
            }
            QualIdentTermCodeInterpCursor_::Fin => {
              return Err(unimpl());
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpQualIdentTerm(_cur_term, ref mut state)) => {
          match state.cur {
            QualIdentTermCodeInterpCursor_::Term(term, ident) => {
              state.term = Some((term, self.get_res()?));
              state.cur = QualIdentTermCodeInterpCursor_::Ident(ident);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            QualIdentTermCodeInterpCursor_::Ident(ident) => {
              state.ident = Some((ident, self.get_res()?));
              state.cur = QualIdentTermCodeInterpCursor_::Fin;
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            QualIdentTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpBunchTerm(_cur_term, ref mut state)) => {
          if state.cur.is_nil() {
            // FIXME: fill result.
            //self.put_res(_)?;
            self.knt_ = knt.prev;
            self.port = Port_::Return;
          } else {
            let cur_cel = self.lookup_term_code_cell(state.cur)?;
            self.knt_ = MemKnt{
              clk,
              prev: knt.into(),
              cur:  MemKnt_::InterpTerm(cur_cel.dptr.into_term_code(), TermCodeInterpState_::fresh()),
            }.into_ref();
            /*self.port = Port_::Enter;*/
          }
        }
        (Port_::Return, &mut MemKnt_::InterpBunchTerm(_cur_term, ref mut state)) => {
          let cur_cel = self.lookup_term_code_cell(state.cur)?;
          state.tup.push((cur_cel.dptr.into_term_code(), self.get_res()?));
          state.cur = cur_cel.next.get().into_term_code();
          self.knt_ = knt.into();
          self.port = Port_::Enter;
        }
        (Port_::Enter, &mut MemKnt_::InterpEqualTerm(_cur_term, ref mut state)) => {
          match state.cur {
            EqualTermCodeInterpCursor_::LTerm(lterm, _) => {
              _traceln!(self, "DEBUG: InterpEqualTerm: Enter:  LTerm: {:?}", lterm);
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(lterm, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            EqualTermCodeInterpCursor_::RTerm(rterm) => {
              _traceln!(self, "DEBUG: InterpEqualTerm: Enter:  RTerm: {:?}", rterm);
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(rterm, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            EqualTermCodeInterpCursor_::Fin => {
              _traceln!(self, "DEBUG: InterpEqualTerm: Enter:  Fin");
              let lterm = state.lterm.unwrap().1;
              let rterm = state.rterm.unwrap().1;
              match self.reg.tctx {
                TermContext_::Unify => {
                  self.unify(clk, lterm, rterm)?;
                }
                TermContext_::Match => {
                  let lroot = self.find(clk, lterm)?;
                  let rroot = self.find(clk, rterm)?;
                  self.res_.reg = ResReg_::Mat(lroot.cls == rroot.cls);
                }
              }
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpEqualTerm(_cur_term, ref mut state)) => {
          match state.cur {
            EqualTermCodeInterpCursor_::LTerm(lterm, rterm) => {
              state.lterm = Some((lterm, self.get_res()?));
              state.cur = EqualTermCodeInterpCursor_::RTerm(rterm);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            EqualTermCodeInterpCursor_::RTerm(rterm) => {
              state.rterm = Some((rterm, self.get_res()?));
              state.cur = EqualTermCodeInterpCursor_::Fin;
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            EqualTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpNEqualTerm(_cur_term, ref mut state)) => {
          match state.cur {
            NEqualTermCodeInterpCursor_::LTerm(lterm, _) => {
              _traceln!(self, "DEBUG: InterpNEqualTerm: Enter:  LTerm: {:?}", lterm);
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(lterm, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            NEqualTermCodeInterpCursor_::RTerm(rterm) => {
              _traceln!(self, "DEBUG: InterpNEqualTerm: Enter:  RTerm: {:?}", rterm);
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(rterm, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            NEqualTermCodeInterpCursor_::Fin => {
              _traceln!(self, "DEBUG: InterpNEqualTerm: Enter:  Fin");
              let lterm = state.lterm.unwrap().1;
              let rterm = state.rterm.unwrap().1;
              let lroot = self.find(clk, lterm)?;
              let rroot = self.find(clk, rterm)?;
              let x = self._fresh().into_term();
              let term_ = NEqualTerm_{buf: [lroot, rroot]};
              self.put_term(clk, x, term_)?;
              self.put_res(x)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpNEqualTerm(_cur_term, ref mut state)) => {
          match state.cur {
            NEqualTermCodeInterpCursor_::LTerm(lterm, rterm) => {
              state.lterm = Some((lterm, self.get_res()?));
              state.cur = NEqualTermCodeInterpCursor_::RTerm(rterm);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            NEqualTermCodeInterpCursor_::RTerm(rterm) => {
              state.rterm = Some((rterm, self.get_res()?));
              state.cur = NEqualTermCodeInterpCursor_::Fin;
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            NEqualTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpQEqualTerm(_cur_term, ref mut state)) => {
          match state.cur {
            QEqualTermCodeInterpCursor_::LTerm(lterm, _) => {
              _traceln!(self, "DEBUG: InterpQEqualTerm: Enter:  LTerm: {:?}", lterm);
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(lterm, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            QEqualTermCodeInterpCursor_::RTerm(rterm) => {
              _traceln!(self, "DEBUG: InterpQEqualTerm: Enter:  RTerm: {:?}", rterm);
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(rterm, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            QEqualTermCodeInterpCursor_::Fin => {
              _traceln!(self, "DEBUG: InterpQEqualTerm: Enter:  Fin");
              let lterm_ = state.lterm.unwrap().1;
              let rterm_ = state.rterm.unwrap().1;
              let x = self._fresh().into_term();
              // FIXME: this is the old value semantics.
              let obj = LitVal_::Bool(lterm_ == rterm_);
              //self.env.table_full[SNUM_TERM_SORT as usize].insert(x.into(), Box::new(_));
              self.env.table_full[SNUM_VAL_SORT as usize].insert(x.into(), Box::new(obj));
              self.put_res(x)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpQEqualTerm(_cur_term, ref mut state)) => {
          match state.cur {
            QEqualTermCodeInterpCursor_::LTerm(lterm, rterm) => {
              state.lterm = Some((lterm, self.get_res()?));
              state.cur = QEqualTermCodeInterpCursor_::RTerm(rterm);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            QEqualTermCodeInterpCursor_::RTerm(rterm) => {
              state.rterm = Some((rterm, self.get_res()?));
              state.cur = QEqualTermCodeInterpCursor_::Fin;
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            QEqualTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpApplyTerm(cur_term_code, ref mut state)) => {
          if state.cur.is_nil() {
            _traceln!(self, "DEBUG: InterpApplyTerm: Enter:  fin");
            let x = self._fresh().into_term();
            let mut tup_buf: Vec<ENum> = Vec::with_capacity(state.tup.len());
            for &(_, t) in state.tup.iter() {
              // FIXME: SNum into ENum should go through unifier.
              tup_buf.push(t.into());
            }
            let term_ = TupleTerm_{buf: tup_buf.into()};
            self.put_term(clk, x, term_)?;
            // FIXME: undo entry for function-based apply.
            //self.log.push(LogEntryRef_::Undo(UndoLogEntry_::ApplyTerm(x).into()));
            let fun_head = state.tup[0].1.into_fun();
            _traceln!(self, "DEBUG: InterpApplyTerm: Enter:  fun head={:?}", fun_head);
            if let Some(fun_head_term) = self.env.table_full[SNUM_TERM_SORT as usize].get(&fun_head.into()) {
              _traceln!(self, "DEBUG: InterpApplyTerm: Enter:  fun head is term = {:?}", fun_head_term);
              if let Some(id_term) = fun_head_term.as_any().downcast_ref::<IdentTerm_>() {
                if let Some(&id) = self.env.raw_id_index.get(&id_term.raw) {
                  if let Some(&fun_head) = self.env.fun_name.get(&id) {
                    _traceln!(self, "DEBUG: InterpApplyTerm: Enter:  fun apply...");
                    let span = self.lookup_term_code(cur_term_code)?._span()?;
                    let fun_head = fun_head.into_fun();
                    let mut fun = self.borrow_fun(fun_head)?;
                    let mut tup = Vec::with_capacity(state.tup.len());
                    for &(_, t) in state.tup.iter() {
                      tup.push(self.find(clk, t)?);
                    }
                    let knt = BorrowedMemKnt{
                      clk:  knt.clk,
                      prev: &knt.prev,
                      cur:  MemKnt_::InterpApplyTerm(cur_term_code, state.clone()),
                    };
                    let result = fun.__apply__(self, span, fun_head.into(), &tup, x.into(), knt)?;
                    self.unborrow_fun(fun_head, fun)?;
                    match result {
                      None => {}
                      Some(yield_) => {
                        _traceln!(self, "DEBUG: InterpApplyTerm: Enter:  fun yield...");
                        return Ok(yield_);
                      }
                    }
                  }
                }
              }
            }
            self.put_res(x)?;
            self.knt_ = knt.prev;
            self.port = Port_::Return;
          } else {
            let cur_cel = self.lookup_term_code_cell(state.cur)?;
            _traceln!(self, "DEBUG: InterpApplyTerm: Enter:  dptr={:?}", cur_cel.dptr);
            self.knt_ = MemKnt{
              clk,
              prev: knt.into(),
              cur:  MemKnt_::InterpTerm(cur_cel.dptr.into_term_code(), TermCodeInterpState_::fresh()),
            }.into_ref();
            /*self.port = Port_::Enter;*/
          }
        }
        (Port_::Return, &mut MemKnt_::InterpApplyTerm(_cur_term, ref mut state)) => {
          let cur_cel_ = self.lookup_term_code_cell(state.cur)?;
          _traceln!(self, "DEBUG: InterpApplyTerm: Return: dptr={:?} next={:?}", cur_cel_.dptr, cur_cel_.next.get());
          state.tup.push((cur_cel_.dptr.into_term_code(), self.get_res()?));
          state.cur = cur_cel_.next.get().into_term_code();
          self.knt_ = knt.into();
          self.port = Port_::Enter;
        }
        (Port_::Enter, &mut MemKnt_::InterpApplyBindLTerm(_cur_term, ref mut state)) => {
          match state.cur {
            ApplyBindLTermCodeInterpCursor_::Bind(bind_cur, _tup_cur) => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(bind_cur, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            ApplyBindLTermCodeInterpCursor_::Tup(tup_cur) => {
              if tup_cur.is_nil() {
                state.cur = ApplyBindLTermCodeInterpCursor_::Fin;
                self.knt_ = knt.into();
                /*self.port = Port_::Enter;*/
              } else {
                let tup_cel_ = self.lookup_term_code_cell(tup_cur)?;
                _traceln!(self, "DEBUG: tup_cel_ = {:?}", tup_cel_);
                self.knt_ = MemKnt{
                  clk,
                  prev: knt.into(),
                  cur:  MemKnt_::InterpTerm(tup_cel_.dptr.into_term_code(), TermCodeInterpState_::fresh()),
                }.into_ref();
                /*self.port = Port_::Enter;*/
              }
            }
            ApplyBindLTermCodeInterpCursor_::Fin => {
              let x = self._fresh().into_term();
              let mut tup_buf: Vec<ENum> = Vec::with_capacity(state.tup.len());
              for &(_, t) in state.tup.iter() {
                tup_buf.push(self.find(clk, t)?);
              }
              let term_ = TupleTerm_{buf: tup_buf.into()};
              self.put_term(clk, x, term_)?;
              let y = match state.bind {
                None => return Err(bot()),
                Some((_, v)) => v
              };
              self.unify(clk, x, y)?;
              self.put_res(y)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpApplyBindLTerm(_cur_term, ref mut state)) => {
          match state.cur {
            ApplyBindLTermCodeInterpCursor_::Bind(bind_cur, tup_cur) => {
              state.bind = Some((bind_cur, self.get_res()?));
              state.cur = ApplyBindLTermCodeInterpCursor_::Tup(tup_cur);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            ApplyBindLTermCodeInterpCursor_::Tup(tup_cur) => {
              if tup_cur.is_nil() {
                state.cur = ApplyBindLTermCodeInterpCursor_::Fin;
              } else {
                let tup_cel_ = self.lookup_term_code_cell(tup_cur)?;
                state.tup.push((tup_cel_.dptr.into_term_code(), self.get_res()?));
                state.cur = ApplyBindLTermCodeInterpCursor_::Tup(tup_cel_.next.get().into_term_code());
              }
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            ApplyBindLTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpApplyBindRTerm(_cur_term, ref mut state)) => {
          match state.cur {
            ApplyBindRTermCodeInterpCursor_::Tup(tup_cur, bind_cur) => {
              if tup_cur.is_nil() {
                state.cur = ApplyBindRTermCodeInterpCursor_::Bind(bind_cur);
                self.knt_ = knt.into();
                /*self.port = Port_::Enter;*/
              } else {
                let tup_cel_ = self.lookup_term_code_cell(tup_cur)?;
                _traceln!(self, "DEBUG: tup_cel_ = {:?}", tup_cel_);
                self.knt_ = MemKnt{
                  clk,
                  prev: knt.into(),
                  cur:  MemKnt_::InterpTerm(tup_cel_.dptr.into_term_code(), TermCodeInterpState_::fresh()),
                }.into_ref();
                /*self.port = Port_::Enter;*/
              }
            }
            ApplyBindRTermCodeInterpCursor_::Bind(bind_cur) => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(bind_cur, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            ApplyBindRTermCodeInterpCursor_::Fin => {
              let x = self._fresh().into_term();
              let mut tup_buf: Vec<ENum> = Vec::with_capacity(state.tup.len());
              for &(_, t) in state.tup.iter() {
                tup_buf.push(self.find(clk, t)?);
              }
              let term_ = TupleTerm_{buf: tup_buf.into()};
              self.put_term(clk, x, term_)?;
              let y = match state.bind {
                None => return Err(bot()),
                Some((_, v)) => v
              };
              self.unify(clk, x, y)?;
              self.put_res(y)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpApplyBindRTerm(_cur_term, ref mut state)) => {
          match state.cur {
            ApplyBindRTermCodeInterpCursor_::Tup(tup_cur, bind_cur) => {
              if tup_cur.is_nil() {
                state.cur = ApplyBindRTermCodeInterpCursor_::Bind(bind_cur);
              } else {
                let cur_cel_ = self.lookup_term_code_cell(tup_cur)?;
                state.tup.push((cur_cel_.dptr.into_term_code(), self.get_res()?));
                state.cur = ApplyBindRTermCodeInterpCursor_::Tup(cur_cel_.next.get().into_term_code(), bind_cur);
              }
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            ApplyBindRTermCodeInterpCursor_::Bind(bind_cur) => {
              state.bind = Some((bind_cur, self.get_res()?));
              state.cur = ApplyBindRTermCodeInterpCursor_::Fin;
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            ApplyBindRTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpBindLTerm(_cur_term, ref mut state)) => {
          match state.cur {
            BindLTermCodeInterpCursor_::LBind(bind_cur, _tup_cur) => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(bind_cur, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            BindLTermCodeInterpCursor_::RTerm(tup_cur) => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(tup_cur, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            BindLTermCodeInterpCursor_::Fin => {
              let x = match state.rterm {
                None => return Err(bot()),
                Some((_, v)) => v
              };
              let y = match state.lbind {
                None => return Err(bot()),
                Some((_, v)) => v
              };
              self.unify(clk, x, y)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpBindLTerm(_cur_term, ref mut state)) => {
          match state.cur {
            BindLTermCodeInterpCursor_::LBind(bind_cur, _tup_cur) => {
              state.lbind = Some((bind_cur, self.get_res()?));
              state.cur = BindLTermCodeInterpCursor_::RTerm(_tup_cur);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            BindLTermCodeInterpCursor_::RTerm(tup_cur) => {
              state.rterm = Some((tup_cur, self.get_res()?));
              state.cur = BindLTermCodeInterpCursor_::Fin;
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            BindLTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpBindRTerm(_cur_term, ref mut state)) => {
          match state.cur {
            BindRTermCodeInterpCursor_::LTerm(tup_cur, _bind_cur) => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(tup_cur, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            BindRTermCodeInterpCursor_::RBind(bind_cur) => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(bind_cur, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            BindRTermCodeInterpCursor_::Fin => {
              let x = match state.lterm {
                None => return Err(bot()),
                Some((_, v)) => v
              };
              let y = match state.rbind {
                None => return Err(bot()),
                Some((_, v)) => v
              };
              self.unify(clk, x, y)?;
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpBindRTerm(_cur_term, ref mut state)) => {
          match state.cur {
            BindRTermCodeInterpCursor_::LTerm(tup_cur, _bind_cur) => {
              state.lterm = Some((tup_cur, self.get_res()?));
              state.cur = BindRTermCodeInterpCursor_::RBind(_bind_cur);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            BindRTermCodeInterpCursor_::RBind(bind_cur) => {
              state.rbind = Some((bind_cur, self.get_res()?));
              state.cur = BindRTermCodeInterpCursor_::Fin;
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            BindRTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        (Port_::Enter, &mut MemKnt_::InterpEffectTerm(_cur_term, ref mut state)) => {
          match state.cur {
            EffectTermCodeInterpCursor_::LTerm(lterm_cur, _) => {
              self.knt_ = MemKnt{
                clk,
                prev: knt.into(),
                cur:  MemKnt_::InterpTerm(lterm_cur, TermCodeInterpState_::fresh()),
              }.into_ref();
              /*self.port = Port_::Enter;*/
            }
            EffectTermCodeInterpCursor_::RTup(tup_cur) => {
              if tup_cur.is_nil() {
                state.cur = EffectTermCodeInterpCursor_::Fin;
                self.knt_ = knt.into();
                /*self.port = Port_::Enter;*/
              } else {
                let tup_cel_ = self.lookup_term_code_cell(tup_cur)?;
                self.knt_ = MemKnt{
                  clk,
                  prev: knt.into(),
                  cur:  MemKnt_::InterpTerm(tup_cel_.dptr.into_term_code(), TermCodeInterpState_::fresh()),
                }.into_ref();
                /*self.port = Port_::Enter;*/
              }
            }
            EffectTermCodeInterpCursor_::Fin => {
              self.knt_ = knt.prev;
              self.port = Port_::Return;
            }
          }
        }
        (Port_::Return, &mut MemKnt_::InterpEffectTerm(_cur_term, ref mut state)) => {
          match state.cur {
            EffectTermCodeInterpCursor_::LTerm(lterm_cur, tup_cur) => {
              state.lterm = Some((lterm_cur, self.get_res()?));
              state.cur = EffectTermCodeInterpCursor_::RTup(tup_cur);
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            EffectTermCodeInterpCursor_::RTup(tup_cur) => {
              if tup_cur.is_nil() {
                //state.cur = EffectTermCodeInterpCursor_::Fin;
                return Err(bot());
              } else {
                let cur_cel_ = self.lookup_term_code_cell(tup_cur)?;
                state.rtup.push((cur_cel_.dptr.into_term_code(), self.get_res()?));
                state.cur = EffectTermCodeInterpCursor_::RTup(cur_cel_.next.get().into_term_code());
              }
              self.knt_ = knt.into();
              self.port = Port_::Enter;
            }
            EffectTermCodeInterpCursor_::Fin => {
              return Err(bot());
            }
          }
        }
        _ => {
          return Err(("unimpl").into());
        }
      }
      let x_post = self._peek();
      _traceln!(self, "DEBUG: FastInterp::resume_: post: clk={:?} xub={:?} port={:?} res={:?}",
          clk, x_post, self.port, self.res_.peek());
      if xlb != x_post {
        _traceln!(self, "DEBUG: FastInterp::resume_:       fresh=({:?} .. {:?}]", xlb, x_post);
      }
    }
  }
}
