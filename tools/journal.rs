extern crate pythia;
extern crate uv;

use pythia::journal::{JournalBackend};
use uv::*;
use uv::bindings::*;
use uv::extras::*;

use std::cell::{RefCell};
use std::collections::{HashMap, HashSet};
use std::os::raw::{c_int};
use std::str::{from_utf8};

thread_local! {
  static BACKEND: Backend = Backend::new();
}

const RESPONSE_OK:  &'static [u8] = b"ok";
const RESPONSE_ERR: &'static [u8] = b"err";

pub enum APIAction {
  Put,
  Get,
}

#[derive(Default)]
pub struct Store {
  buf: HashSet<BackingBuf>,
}

pub struct Backend {
  inner: RefCell<JournalBackend>,
  store: RefCell<Store>,
  loop_: UvLoop,
}

impl Backend {
  pub fn new() -> Backend {
    let inner = JournalBackend::cold_start();
    let loop_ = UvLoop::new();
    Backend{
      inner: RefCell::new(inner),
      store: RefCell::new(Store::default()),
      loop_,
    }
  }

  pub fn run(&self) {
    let sig = UvSignal::new(&self.loop_);
    let tcp = UvTcp::new(&self.loop_);
    println!("DEBUG: Backend::run: signal...");
    //sig.start::<Backend>(2);
    sig.start::<Backend>(15);
    println!("DEBUG: Backend::run: bind...");
    tcp.bind(("127.0.0.1", 9000));
    println!("DEBUG: Backend::run: listen...");
    tcp.listen::<Backend>();
    println!("DEBUG: Backend::run: run...");
    self.loop_.run();
  }

  pub fn stop(&self) {
    println!("DEBUG: Backend::stop: ...");
    self.loop_.stop();
  }
}

impl UvAllocCb for Backend {
  fn callback(_handle: UvHandle, suggested_size: usize, buf: &mut UvBuf) {
    println!("DEBUG: Backend: alloc callback: size = {} buf.len = {:?}", suggested_size, buf.as_bytes().map(|b| b.len()));
    BACKEND.with(|backend| {
      let mut store = backend.store.borrow_mut();
      println!("DEBUG: Backend: alloc callback: alloc backing buf...");
      let backing_buf = BackingBuf::new_uninit(suggested_size);
      let _ = buf.replace_raw_parts_unchecked(backing_buf.ptr as _, backing_buf.len);
      if let Some(_) = store.buf.replace(backing_buf) {
        println!("DEBUG: Backend: alloc callback: warning: backing buf was already stored!");
      }
    });
  }
}

impl UvReadCb for Backend {
  fn callback(client: UvStream, nread: isize, buf: &mut UvBuf) {
    println!("DEBUG: Backend: read callback: nread = {} buf.len = {}", nread, buf.len());
    if nread < 0 {
      let errno = nread as c_int;
      if errno == UV_EOF {
        println!("DEBUG: Backend: read callback: eof");
      } else {
        // FIXME
        println!("DEBUG: Backend: read callback: error = {}", errno);
      }
      let (backing_ptr, backing_len) = buf.take_raw_parts();
      if let Some(mut backing_buf) = BackingBuf::maybe_from_raw_parts_unchecked(backing_ptr as _, backing_len) {
        BACKEND.with(|backend| {
          let mut store = backend.store.borrow_mut();
          if !store.buf.remove(&backing_buf) {
            println!("DEBUG: Backend: read callback: warning: backing buf was NOT in store!");
          }
          println!("DEBUG: Backend: read callback: free backing buf...");
          backing_buf.free_unchecked();
        });
      }
      let req = UvShutdown::new();
      client.shutdown::<Backend>(req);
      return;
    }
    let nread = nread as usize;
    if nread >= 4 {
      if let Some(b) = buf.as_bytes() {
        // TODO
        let action = if &b[ .. 4] == b"put\n" {
          APIAction::Put
        } else if &b[ .. 4] == b"get\n" {
          APIAction::Get
        } else {
          let mut backing_buf = BackingBuf::new_uninit(RESPONSE_ERR.len());
          backing_buf.as_mut_bytes().copy_from_slice(RESPONSE_ERR);
          let write_buf = UvBuf::from_raw_parts_unchecked(backing_buf.as_ptr() as _, backing_buf.len());
          BACKEND.with(|backend| {
            let mut store = backend.store.borrow_mut();
            if let Some(_) = store.buf.replace(backing_buf) {
              println!("DEBUG: Backend: read callback: warning: backing buf was already stored!");
            }
          });
          let req = UvWrite::new();
          client.write::<Backend>(req, &write_buf);
          return;
        };
        match action {
          APIAction::Put => {
            if nread <= 4 {
              let mut backing_buf = BackingBuf::new_uninit(RESPONSE_ERR.len());
              backing_buf.as_mut_bytes().copy_from_slice(RESPONSE_ERR);
              let write_buf = UvBuf::from_raw_parts_unchecked(backing_buf.as_ptr() as _, backing_buf.len());
              BACKEND.with(|backend| {
                let mut store = backend.store.borrow_mut();
                if let Some(_) = store.buf.replace(backing_buf) {
                  println!("DEBUG: Backend: read callback: warning: backing buf was already stored!");
                }
              });
              let req = UvWrite::new();
              client.write::<Backend>(req, &write_buf);
              return;
            }
            if b[nread-1] != b'\n' {
              let mut backing_buf = BackingBuf::new_uninit(RESPONSE_ERR.len());
              backing_buf.as_mut_bytes().copy_from_slice(RESPONSE_ERR);
              let write_buf = UvBuf::from_raw_parts_unchecked(backing_buf.as_ptr() as _, backing_buf.len());
              BACKEND.with(|backend| {
                let mut store = backend.store.borrow_mut();
                if let Some(_) = store.buf.replace(backing_buf) {
                  println!("DEBUG: Backend: read callback: warning: backing buf was already stored!");
                }
              });
              let req = UvWrite::new();
              client.write::<Backend>(req, &write_buf);
              return;
            }
            for p in 4 .. (nread - 1) {
              if b[p] == b'\n' {
                let mut backing_buf = BackingBuf::new_uninit(RESPONSE_ERR.len());
                backing_buf.as_mut_bytes().copy_from_slice(RESPONSE_ERR);
                let write_buf = UvBuf::from_raw_parts_unchecked(backing_buf.as_ptr() as _, backing_buf.len());
                BACKEND.with(|backend| {
                  let mut store = backend.store.borrow_mut();
                  if let Some(_) = store.buf.replace(backing_buf) {
                    println!("DEBUG: Backend: read callback: warning: backing buf was already stored!");
                  }
                });
                let req = UvWrite::new();
                client.write::<Backend>(req, &write_buf);
                return;
              }
            }
            match from_utf8(&b[4 .. (nread - 1)]) {
              Err(_) => {
                let mut backing_buf = BackingBuf::new_uninit(RESPONSE_ERR.len());
                backing_buf.as_mut_bytes().copy_from_slice(RESPONSE_ERR);
                let write_buf = UvBuf::from_raw_parts_unchecked(backing_buf.as_ptr() as _, backing_buf.len());
                BACKEND.with(|backend| {
                  let mut store = backend.store.borrow_mut();
                  if let Some(_) = store.buf.replace(backing_buf) {
                    println!("DEBUG: Backend: read callback: warning: backing buf was already stored!");
                  }
                });
                let req = UvWrite::new();
                client.write::<Backend>(req, &write_buf);
                return;
              }
              Ok(s) => {
                BACKEND.with(|backend| {
                  println!("DEBUG: Backend: read callback: journal append");
                  backend.inner.borrow_mut().append(s);
                });
              }
            }
          }
          _ => {
            // TODO
          }
        }
      }
    }
    println!("DEBUG: Backend: read callback: write response...");
    // TODO: always assure buffer lifetime.
    //let res_str = format!("Hello, world! {}\n", nread);
    //let res_buf = res_str.as_bytes();
    let mut backing_buf = BackingBuf::new_uninit(RESPONSE_OK.len());
    backing_buf.as_mut_bytes().copy_from_slice(RESPONSE_OK);
    let write_buf = UvBuf::from_raw_parts_unchecked(backing_buf.as_ptr() as _, backing_buf.len());
    BACKEND.with(|backend| {
      let mut store = backend.store.borrow_mut();
      if let Some(_) = store.buf.replace(backing_buf) {
        println!("DEBUG: Backend: read callback: warning: backing buf was already stored!");
      }
    });
    let (backing_ptr, backing_len) = buf.take_raw_parts();
    if let Some(mut backing_buf) = BackingBuf::maybe_from_raw_parts_unchecked(backing_ptr as _, backing_len) {
      BACKEND.with(|backend| {
        let mut store = backend.store.borrow_mut();
        if !store.buf.remove(&backing_buf) {
          println!("DEBUG: Backend: read callback: warning: backing buf was NOT in store!");
        }
        println!("DEBUG: Backend: read callback: free backing buf...");
        backing_buf.free_unchecked();
      });
    }
    let req = UvWrite::new();
    client.write::<Backend>(req, &write_buf);
    //let req = UvShutdown::new();
    //client.shutdown::<Backend>(req);
  }
}

impl UvWriteCb for Backend {
  fn callback(mut req: UvWrite, status: c_int) {
    println!("DEBUG: Backend: write callback: status = {}", status);
    if let Some(bufs) = req._inner_mut_bufs_unchecked() {
      println!("DEBUG: Backend: write callback: found req bufs: bufs.len = {}", bufs.len());
      for buf in bufs.iter_mut() {
        let (backing_ptr, backing_len) = buf.take_raw_parts();
        if let Some(mut backing_buf) = BackingBuf::maybe_from_raw_parts_unchecked(backing_ptr as _, backing_len) {
          BACKEND.with(|backend| {
            let mut store = backend.store.borrow_mut();
            if !store.buf.remove(&backing_buf) {
              println!("DEBUG: Backend: write callback: warning: backing buf was NOT in store!");
            }
            println!("DEBUG: Backend: write callback: free backing buf...");
            backing_buf.free_unchecked();
          });
        }
      }
    } else {
      println!("DEBUG: Backend: write callback: warning: no req bufs found!");
    }
    req.into_req()._free_unchecked();
  }
}

impl UvShutdownCb for Backend {
  fn callback(req: UvShutdown, status: c_int) {
    println!("DEBUG: Backend: shutdown callback: status = {}", status);
    req.into_req()._free_unchecked();
  }
}

impl UvConnectionCb for Backend {
  fn callback_raw(server: *mut uv_stream_t, status: c_int) {
    println!("DEBUG: Backend: connection callback: hello... status = {}", status);
    BACKEND.with(|backend| {
      let loop_ = &backend.loop_;
      let client = UvTcp::new(loop_);
      let stream = UvStream::from_raw(server);
      match stream.accept(&client) {
        Err(e) => {
          println!("DEBUG: Backend: connection callback: accept: err = {e}");
          let req = UvShutdown::new();
          client.shutdown::<Backend>(req);
        }
        Ok(_) => {
          println!("DEBUG: Backend: connection callback: accept: ok");
          client.read_start::<Backend, Backend>();
        }
      }
    });
  }
}

impl UvCloseCb for Backend {
  fn callback(handle: UvHandle) {
    println!("DEBUG: Backend: close callback");
    handle._free_unchecked();
  }
}

impl UvSignalCb for Backend {
  fn callback(signal: UvSignal, signum: c_int) {
    println!("DEBUG: Backend: signal callback: signum = {}", signum);
    if signum == 2 || signum == 15 {
      BACKEND.with(|backend| {
        backend.stop();
      });
    }
    //signal.stop();
  }
}

fn main() {
  init_once_uv();
  BACKEND.with(|backend| {
    backend.run();
  });
}
