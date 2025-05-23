use crate::algo::{BTreeMap};
use crate::algo::blake2s::{Blake2s};
use crate::clock::{Timestamp};
use crate::util::hex::{HexFormat};

use getrandom::{getrandom};
use serde::{Serialize};
use serde::ser::{Serializer, SerializeStruct};

use std::fmt::{Display, Debug, Formatter, Result as FmtResult};
use std::mem::{replace};

pub const HASH_SIZE: usize = 32;
pub const SHORT_HASH_SIZE: usize = 16;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameId {
  inner: Box<[u8]>,
}

impl Debug for FrameId {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "FrameId({})", self.to_string())
  }
}

impl Display for FrameId {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", self.to_string())
  }
}

impl Serialize for FrameId {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&self.to_string())
  }
}

/*impl Default for FrameId {
}*/

impl FrameId {
  pub fn root() -> FrameId {
    let mut inner = Vec::with_capacity(SHORT_HASH_SIZE);
    inner.resize(SHORT_HASH_SIZE, 0);
    let inner = inner.into();
    FrameId{inner}
  }

  pub fn is_root(&self) -> bool {
    for &u in self.inner.iter() {
      if u != 0 {
        return false;
      }
    }
    true
  }

  pub fn to_string(&self) -> String {
    HexFormat::default().lower().rev()
      .to_string(&self.inner)
  }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotHash {
  inner: Box<[u8]>,
}

impl Debug for SnapshotHash {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "SnapshotHash({})", self.to_string())
  }
}

impl Display for SnapshotHash {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", self.to_string())
  }
}

impl Serialize for SnapshotHash {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    // TODO: hash fun spec?
    serializer.serialize_str(&self.to_string())
    //serializer.serialize_str(&format!("blake2s:{}", self.to_string()))
  }
}

/*impl Default for SnapshotHash {
}*/

impl From<[u8; 32]> for SnapshotHash {
  fn from(buf: [u8; 32]) -> SnapshotHash {
    SnapshotHash{
      inner: buf.into(),
    }
  }
}

impl SnapshotHash {
  pub fn root() -> SnapshotHash {
    let mut inner = Vec::with_capacity(HASH_SIZE);
    inner.resize(HASH_SIZE, 0);
    let inner = inner.into();
    SnapshotHash{inner}
  }

  pub fn is_root(&self) -> bool {
    for &u in self.inner.iter() {
      if u != 0 {
        return false;
      }
    }
    true
  }

  pub fn to_string(&self) -> String {
    HexFormat::default().lower()
      .to_string(&self.inner)
  }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentHash {
  inner: Box<[u8]>,
}

impl Debug for ContentHash {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "ContentHash({})", self.to_string())
  }
}

impl Serialize for ContentHash {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    // TODO: hash fun spec?
    serializer.serialize_str(&self.to_string())
    //serializer.serialize_str(&format!("blake2s:{}", self.to_string()))
  }
}

impl From<[u8; 32]> for ContentHash {
  fn from(buf: [u8; 32]) -> ContentHash {
    ContentHash{
      inner: buf.into(),
    }
  }
}

impl ContentHash {
  pub fn empty() -> ContentHash {
    let mut inner = Vec::with_capacity(HASH_SIZE);
    inner.resize(HASH_SIZE, 0);
    let inner = inner.into();
    ContentHash{inner}
  }

  pub fn to_string(&self) -> String {
    HexFormat::default().lower()
      .to_string(&self.inner)
  }
}

#[derive(Debug)]
pub struct FramePointer {
  init: SnapshotHash,
  last: SnapshotHash,
}

impl FramePointer {
  pub fn fresh() -> FramePointer {
    FramePointer{
      init: SnapshotHash::root(),
      last: SnapshotHash::root(),
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum SnapshotMarker {
  FreshFrame,
  Merge,
  MergeConflict,
  Review,
}

// TODO
#[derive(Clone, Debug)]
pub struct SnapshotMetadata {
  frame: FrameId,
  prev: Vec<SnapshotHash>,
  mark: Vec<SnapshotMarker>,
  timestamp: Timestamp,
  // TODO: "author" info.
}

// TODO
#[derive(Clone)]
pub struct Snapshot {
  hash: SnapshotHash,
  rehash: bool,
  //lasthash: Timestamp,
  metadata: SnapshotMetadata,
  //testdata: SnapshotTestData,
  hashdata: SnapshotData,
}

impl Snapshot {
  pub fn rehash(&mut self) {
    if !self.rehash {
      return
    }
    if !self.hashdata.rehash {
      self.rehash = false;
      return;
    }
    self.force_rehash()
  }

  pub fn force_rehash(&mut self) {
    // TODO: other data...
    self.hashdata.rehash();
    let mut merkle_buf = Vec::new();
    merkle_buf.extend(&*self.metadata.frame.inner);
    merkle_buf.extend(&*self.metadata.prev[0].inner);
    merkle_buf.extend(&*self.hashdata.hash.inner);
    let mut h = Blake2s::new_hash();
    h.hash_bytes(&merkle_buf);
    self.hash = SnapshotHash::from(h.finalize());
    self.rehash = false;
  }
}

// TODO
#[derive(Clone)]
pub enum Data {
  // TODO
  Empty,
  String(String),
  //List(Vec<Data>),
  //Tree(Vec<(String, Data)>),
}

impl Data {
  pub fn content_hash(&self) -> ContentHash {
    match self {
      &Data::Empty => {
        ContentHash::empty()
      }
      &Data::String(ref s) => {
        let mut h = Blake2s::new_hash();
        h.hash_bytes(s.as_bytes());
        ContentHash::from(h.finalize())
      }
    }
  }

  pub fn mut_string(&mut self) -> &mut String {
    match self as &_ {
      &Data::Empty => {
        *self = Data::String(String::new());
      }
      &Data::String(_) => {}
    }
    match self {
      &mut Data::String(ref mut s) => s,
      _ => panic!("bug")
    }
  }
}

// TODO
#[derive(Clone)]
pub struct SnapshotTestData {
  // TODO
  hash: ContentHash,
  rehash: bool,
  //lasthash: Timestamp,
  data: Data,
}

// TODO
#[derive(Clone)]
pub struct SnapshotData {
  // TODO
  hash: ContentHash,
  rehash: bool,
  //lasthash: Timestamp,
  data: Data,
  //test_data: _,
  //review_data: _,
  //merge_data: _,
  //issue_data: _,
  //goal_data: _,
}

impl SnapshotData {
  pub fn empty() -> SnapshotData {
    SnapshotData{
      hash: ContentHash::empty(),
      rehash: false,
      data: Data::Empty,
    }
  }

  pub fn rehash(&mut self) {
    if !self.rehash {
      return
    }
    self.hash = self.data.content_hash();
    self.rehash = false;
  }

  pub fn data(&self) -> &Data {
    &self.data
  }

  pub fn mut_data(&mut self) -> &mut Data {
    self.rehash = true;
    &mut self.data
  }
}

pub type Repo = DBase;

pub struct DBase {
  snapshots: BTreeMap<SnapshotHash, Snapshot>,
  frames: BTreeMap<FrameId, FramePointer>,
}

impl DBase {
  pub fn root() -> DBase {
    DBase{
      snapshots: BTreeMap::new(),
      frames: BTreeMap::new(),
    }
  }

  pub fn debug_dump(&self) {
    println!("DEBUG: DBase::debug_dump: snapshot count = {}", self.snapshots.len());
    println!("DEBUG: DBase::debug_dump: frame count    = {}", self.frames.len());
  }

  pub fn fresh_frame(&mut self) -> FrameId {
    let mut inner = Vec::with_capacity(SHORT_HASH_SIZE);
    inner.resize(SHORT_HASH_SIZE, 0);
    match getrandom(&mut inner) {
      Err(_) => panic!("bug"),
      Ok(_) => {}
    }
    let inner = inner.into();
    let frame_id = FrameId{inner};
    match self.frames.insert(frame_id.clone(), FramePointer::fresh()) {
      None => {}
      Some(_) => panic!("bug")
    }
    frame_id
  }

  pub fn get_snapshot(&self, hash: SnapshotHash) -> Option<Snapshot> {
    if hash.is_root() {
      return None;
    }
    match self.snapshots.get(&hash) {
      None => panic!("bug"),
      Some(v) => Some(v.clone())
    }
  }

  pub fn commit_snapshot(&mut self, snapshot: Snapshot) {
    if snapshot.rehash {
      panic!("bug");
    }
    let frame_id = snapshot.metadata.frame.clone();
    if frame_id.is_root() {
      // TODO
      return;
    }
    match self.frames.get_mut(&frame_id) {
      None => panic!("bug"),
      Some(frameptr) => {
        if frameptr.init.is_root() {
          frameptr.init = snapshot.hash.clone();
        }
        frameptr.last = snapshot.hash.clone();
        println!("DEBUG: DBase::commit_snapshot: metadata = {:?}", &snapshot.metadata);
        println!("DEBUG: DBase::commit_snapshot: frameptr = {:?}", &frameptr);
      }
    }
    match self.snapshots.insert(snapshot.hash.clone(), snapshot) {
      None => {}
      Some(_) => panic!("bug")
    }
  }
}

pub struct Frame {
  snapshot: SnapshotHash,
  modified: bool,
  hashdata: SnapshotData,
}

impl Frame {
  pub fn root() -> Frame {
    Frame{
      snapshot: SnapshotHash::root(),
      modified: false,
      hashdata: SnapshotData::empty(),
    }
  }

  pub fn debug_print_status(&self, dbase: &DBase) {
    match dbase.get_snapshot(self.snapshot.clone()) {
      None => {
        println!("DEBUG: Frame::debug_print_status: frame = {} snapshot = {} modified? = {:?}",
            FrameId::root(),
            &self.snapshot,
            self.modified,
        );
      }
      Some(snapshot) => {
        println!("DEBUG: Frame::debug_print_status: frame = {} snapshot = {} modified? = {:?}",
            &snapshot.metadata.frame,
            &self.snapshot,
            self.modified,
        );
      }
    }
  }

  pub fn fresh(&self, dbase: &mut DBase) -> Frame {
    let timestamp = Timestamp::fresh();
    let frame_id = dbase.fresh_frame();
    let metadata = SnapshotMetadata {
      frame: frame_id,
      prev: vec![self.snapshot.clone()],
      timestamp,
      mark: vec![SnapshotMarker::FreshFrame],
    };
    println!("DEBUG: Frame::fresh: metadata = {:?}", metadata);
    let mut hashdata = self.hashdata.clone();
    hashdata.rehash();
    let mut merkle_buf = Vec::new();
    merkle_buf.extend(&*metadata.frame.inner);
    merkle_buf.extend(&*metadata.prev[0].inner);
    merkle_buf.extend(&*hashdata.hash.inner);
    let mut h = Blake2s::new_hash();
    h.hash_bytes(&merkle_buf);
    let hash = SnapshotHash::from(h.finalize());
    let snapshot = Snapshot{
      hash,
      rehash: false,
      metadata,
      //testdata: _,
      hashdata: hashdata.clone(),
    };
    let snapshot_hash = snapshot.hash.clone();
    dbase.commit_snapshot(snapshot);
    Frame{
      snapshot: snapshot_hash,
      modified: false,
      hashdata,
    }
  }

  pub fn commit(&mut self, dbase: &mut DBase) {
    if !self.modified {
      return;
    }
    let old_snapshot = self.snapshot.clone();
    let old_hash = self.hashdata.hash.clone();
    self.hashdata.rehash();
    let new_hash = self.hashdata.hash.clone();
    if old_hash == new_hash {
      self.modified = false;
      return;
    }
    if self.snapshot.is_root() {
      let hashdata = replace(&mut self.hashdata, SnapshotData::empty());
      *self = self.fresh(dbase);
      let _ = replace(&mut self.hashdata, hashdata);
    }
    let mut snapshot = match dbase.get_snapshot(old_snapshot.clone()) {
      None => panic!("bug"),
      Some(snapshot) => snapshot
    };
    snapshot.metadata.prev.clear();
    snapshot.metadata.prev.push(old_snapshot);
    snapshot.metadata.mark.clear();
    snapshot.metadata.timestamp = Timestamp::fresh();
    snapshot.hashdata = self.hashdata.clone();
    snapshot.force_rehash();
    let snapshot_hash = snapshot.hash.clone();
    dbase.commit_snapshot(snapshot);
    self.snapshot = snapshot_hash;
    self.modified = false;
  }

  pub fn view(&self) -> &SnapshotData {
    &self.hashdata
  }

  pub fn modify(&mut self) -> &mut SnapshotData {
    self.modified = true;
    &mut self.hashdata
  }
}
