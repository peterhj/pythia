use crate::algo::{BTreeMap};
use crate::algo::blake2s::{Blake2s};
use crate::algo::hex::{HexFormat};
use crate::algo::json::{JsonFormat};
use crate::clock::{Timestamp};

use getrandom::{getrandom};
use serde::{Serialize};
use serde::ser::{Serializer, SerializeStruct};

use std::fmt::{Display, Debug, Formatter, Result as FmtResult};
use std::fs::{OpenOptions, create_dir};
use std::io::{BufWriter, Write};
use std::mem::{replace};
use std::path::{PathBuf};

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

  pub fn _as_data_bytes(&self) -> &[u8] {
    &self.inner
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Debug)]
pub enum SnapshotMarker {
  FreshFrame,
  Merge,
  MergeConflict,
  Review,
}

// TODO
#[derive(Clone, Serialize, Debug)]
pub struct SnapshotMetadata {
  frame: FrameId,
  prev: Vec<SnapshotHash>,
  mark: Vec<SnapshotMarker>,
  timestamp: Timestamp,
  // TODO: "author" info.
  // TODO: message or comment.
}

// TODO
#[derive(Clone)]
pub struct Snapshot {
  hash: SnapshotHash,
  rehash: bool,
  //lasthash: Timestamp,
  metadata: SnapshotMetadata,
  //dokidata: SnapshotDokiData,
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
pub struct SnapshotDokiData {
  // TODO
  hash: ContentHash,
  rehash: bool,
  //lasthash: Timestamp,
  data: Data,
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
  data: Object,
  //data: Data,
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
      data: Object::empty(),
    }
  }

  pub fn rehash(&mut self) {
    if !self.rehash {
      return
    }
    //self.hash = self.data.content_hash();
    self.data.rehash();
    self.hash = self.data.hash.clone();
    self.rehash = false;
  }

  pub fn data(&self) -> &Object {
    &self.data
  }

  pub fn mut_data(&mut self) -> &mut Object {
    self.rehash = true;
    &mut self.data
  }
}

// TODO
#[derive(Clone)]
pub enum Data {
  // FIXME: should not be inline but stored in the Store...
  Empty,
  String(String),
  //List(Vec<Data>),
  //Tree(Vec<(String, Data)>),
  Tree(BTreeMap<String, Data>),
}

impl Data {
  pub fn empty() -> Data {
    Data::Empty
  }

  pub fn tree() -> Data {
    Data::Tree(BTreeMap::new())
  }

  pub fn tree_insert_str(&mut self, k: &str, v: &str) {
    match self {
      &mut Data::Tree(ref mut kvs) => {
        kvs.insert(k.to_owned(), Data::String(v.to_owned()));
      }
      _ => panic!("bug")
    }
  }

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
      &Data::Tree(ref kvs) => {
        let mut htree = Blake2s::new_hash();
        for (k, v) in kvs.iter() {
          let mut h = Blake2s::new_hash();
          h.hash_bytes(k.as_bytes());
          let hk = ContentHash::from(h.finalize());
          let hv = v.content_hash();
          htree.hash_bytes(hk._as_data_bytes());
          htree.hash_bytes(hv._as_data_bytes());
        }
        ContentHash::from(htree.finalize())
      }
    }
  }

  pub fn mut_string(&mut self) -> &mut String {
    match self as &_ {
      &Data::Empty => {
        *self = Data::String(String::new());
      }
      &Data::String(_) => {}
      _ => unimplemented!()
    }
    match self {
      &mut Data::String(ref mut s) => s,
      _ => panic!("bug")
    }
  }
}

#[derive(Clone)]
pub enum ObjectData {
  Empty,
  String(String),
  Tree(BTreeMap<String, Object>),
}

impl ObjectData {
  pub fn content_hash(&mut self) -> ContentHash {
    match self {
      &mut ObjectData::Empty => {
        ContentHash::empty()
      }
      &mut ObjectData::String(ref s) => {
        let mut h = Blake2s::new_hash();
        h.hash_bytes(s.as_bytes());
        ContentHash::from(h.finalize())
      }
      &mut ObjectData::Tree(ref mut kvs) => {
        let mut htree = Blake2s::new_hash();
        for (k, v) in kvs.iter_mut() {
          let mut h = Blake2s::new_hash();
          h.hash_bytes(k.as_bytes());
          let hk = ContentHash::from(h.finalize());
          v.rehash();
          let hv = v.hash.clone();
          htree.hash_bytes(hk._as_data_bytes());
          htree.hash_bytes(hv._as_data_bytes());
        }
        ContentHash::from(htree.finalize())
      }
    }
  }
}

// TODO: replacement for Data.
#[derive(Clone)]
pub struct Object {
  hash: ContentHash,
  rehash: bool,
  data: ObjectData,
}

impl Object {
  pub fn empty() -> Object {
    Object{
      hash: ContentHash::empty(),
      rehash: false,
      data: ObjectData::Empty,
    }
  }

  pub fn string_from(s: &str) -> Object {
    Object{
      hash: ContentHash::empty(),
      rehash: true,
      data: ObjectData::String(s.to_owned()),
    }
  }

  pub fn tree() -> Object {
    Object{
      hash: ContentHash::empty(),
      rehash: true,
      data: ObjectData::Tree(BTreeMap::new()),
    }
  }

  pub fn tree_insert_str(&mut self, k: &str, v: &str) {
    match &mut self.data {
      &mut ObjectData::Tree(ref mut kvs) => {
        kvs.insert(k.to_owned(), Object::string_from(v));
      }
      _ => panic!("bug")
    }
  }

  pub fn rehash(&mut self, /*store: &mut Store*/) {
    if !self.rehash {
      return
    }
    self.hash = self.data.content_hash();
    self.rehash = false;
  }

  pub fn mut_string(&mut self) -> &mut String {
    self.rehash = true;
    match &self.data {
      &ObjectData::Empty => {
        self.data = ObjectData::String(String::new());
      }
      &ObjectData::String(_) => {}
      _ => unimplemented!()
    }
    match &mut self.data {
      &mut ObjectData::String(ref mut s) => s,
      _ => panic!("bug")
    }
  }
}

pub type Repo = Store;

pub struct Store {
  frames: BTreeMap<FrameId, FramePointer>,
  snapshots: BTreeMap<SnapshotHash, Snapshot>,
  objects: BTreeMap<ContentHash, Object>,
}

impl Store {
  pub fn root() -> Store {
    Store{
      snapshots: BTreeMap::new(),
      frames: BTreeMap::new(),
      objects: BTreeMap::new(),
    }
  }

  pub fn debug_dump(&self) {
    println!("DEBUG: Store::debug_dump: snapshot count = {}", self.snapshots.len());
    println!("DEBUG: Store::debug_dump: frame count    = {}", self.frames.len());
  }

  pub fn _save(&self) {
    let manifest_dir = PathBuf::from(crate::build::MANIFEST_DIR);
    let mut save_dir = manifest_dir.clone();
    save_dir.push(".aikido");
    let _ = create_dir(&save_dir).ok();
    let mut snapshot_path = save_dir.clone();
    snapshot_path.push("_log.jsonl");
    let log_file = OpenOptions::new()
      .create(true)
      .append(true)
      .open(&snapshot_path).unwrap();
    let mut log_file = BufWriter::new(log_file);
    /*#[derive(Default, Serialize, Debug)]
    enum _FrameType {
      #[default]
      #[serde(rename = "frame")]
      Frame,
    }
    #[derive(Serialize, Debug)]
    struct _Frame {
      _type: _FrameType,
      id: FrameId,
      init: SnapshotHash,
      last: SnapshotHash,
    }*/
    #[derive(Default, Serialize, Debug)]
    enum _SnapshotType {
      #[default]
      #[serde(rename = "snapshot")]
      Snapshot,
    }
    #[derive(Serialize, Debug)]
    struct _Snapshot {
      _type: _SnapshotType,
      hash: SnapshotHash,
      metadata: SnapshotMetadata,
      data: ContentHash,
    }
    for (hash, snapshot) in self.snapshots.iter() {
      assert_eq!(hash, &snapshot.hash);
      assert!(!snapshot.rehash);
      assert!(!snapshot.hashdata.rehash);
      let m = _Snapshot{
        _type: Default::default(),
        hash: hash.clone(),
        metadata: snapshot.metadata.clone(),
        data: snapshot.hashdata.hash.clone(),
      };
      let json_fmt = JsonFormat::new()
          .ascii(true)
          .comma(", ").unwrap()
          .colon(": ").unwrap()
      ;
      let s = json_fmt.to_string(&m).unwrap();
      writeln!(&mut log_file, "{}", s).unwrap();
      self._save_object(&snapshot.hashdata.data, &mut log_file, );
    }
  }

  pub fn _save_object<W: Write>(&self, obj: &Object, log_file: &mut W) {
    /*#[derive(Default, Serialize, Debug)]
    enum _StringType {
      #[default]
      #[serde(rename = "object/string")]
      String,
    }
    #[derive(Serialize, Debug)]
    struct _StringObject {
      _type: _EmptyType,
      hash: ContentHash,
      content: String,
    }*/
    #[derive(Default, Serialize, Debug)]
    enum _TreeType {
      #[default]
      //#[serde(rename = "object")]
      #[serde(rename = "object/tree")]
      Tree,
    }
    #[derive(Serialize, Debug)]
    struct _TreeObject {
      _type: _TreeType,
      hash: ContentHash,
      up: ContentHash,
      key: String,
    }
    match &obj.data {
      &ObjectData::Empty => {}
      &ObjectData::Tree(ref kvs) => {
        for (k, v) in kvs.iter() {
          assert!(!v.rehash);
          let o = _TreeObject{
            _type: Default::default(),
            hash: v.hash.clone(),
            up: obj.hash.clone(),
            key: k.clone(),
          };
          let json_fmt = JsonFormat::new()
              .ascii(true)
              .comma(", ").unwrap()
              .colon(": ").unwrap()
          ;
          let s = json_fmt.to_string(&o).unwrap();
          writeln!(log_file, "{}", s).unwrap();
        }
      }
      _ => unimplemented!()
    }
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
        println!("DEBUG: Store::commit_snapshot: metadata = {:?}", &snapshot.metadata);
        println!("DEBUG: Store::commit_snapshot: frameptr = {:?}", &frameptr);
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

  pub fn _new(metadata: SnapshotMetadata, data: SnapshotData, store: &mut Store) -> Frame {
    let mut merkle_buf = Vec::new();
    merkle_buf.extend(&*metadata.frame.inner);
    merkle_buf.extend(&*metadata.prev[0].inner);
    merkle_buf.extend(&*data.hash.inner);
    let mut h = Blake2s::new_hash();
    h.hash_bytes(&merkle_buf);
    let hash = SnapshotHash::from(h.finalize());
    let snapshot = Snapshot{
      hash,
      rehash: false,
      metadata,
      //testdata: _,
      hashdata: data.clone(),
    };
    let snapshot_hash = snapshot.hash.clone();
    store.commit_snapshot(snapshot);
    Frame{
      snapshot: snapshot_hash,
      modified: false,
      hashdata: data,
    }
  }

  pub fn debug_print_status(&self, store: &Store) {
    match store.get_snapshot(self.snapshot.clone()) {
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

  pub fn import(&self, import_data: Object, store: &mut Store) -> Frame {
    let timestamp = Timestamp::fresh();
    let frame_id = store.fresh_frame();
    let metadata = SnapshotMetadata {
      frame: frame_id,
      prev: vec![self.snapshot.clone()],
      timestamp,
      mark: vec![SnapshotMarker::FreshFrame],
    };
    let mut hashdata = self.hashdata.clone();
    let _ = replace(hashdata.mut_data(), import_data);
    hashdata.rehash();
    Frame::_new(metadata, hashdata, store)
  }

  pub fn fresh(&self, store: &mut Store) -> Frame {
    let timestamp = Timestamp::fresh();
    let frame_id = store.fresh_frame();
    let metadata = SnapshotMetadata {
      frame: frame_id,
      prev: vec![self.snapshot.clone()],
      timestamp,
      mark: vec![SnapshotMarker::FreshFrame],
    };
    println!("DEBUG: Frame::fresh: metadata = {:?}", metadata);
    let mut hashdata = self.hashdata.clone();
    hashdata.rehash();
    Frame::_new(metadata, hashdata, store)
  }

  pub fn commit(&mut self, store: &mut Store) {
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
      *self = self.fresh(store);
      let _ = replace(&mut self.hashdata, hashdata);
    }
    let mut snapshot = match store.get_snapshot(old_snapshot.clone()) {
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
    store.commit_snapshot(snapshot);
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
