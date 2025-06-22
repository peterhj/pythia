extern crate libuv_ffi;
extern crate pythia;

use pythia::algo::json::{JsonFormat, deserialize_json_value};
use pythia::journal::{
  JournalEntrySort_,
  JournalBackend,
  RootSort_,
  AikidoSort_,
  ApproxOracleSort_,
  TestSort_,
};
use libuv_ffi::*;
use libuv_ffi::bindings::*;
use libuv_ffi::extras::*;

use std::cell::{RefCell};
use std::collections::{HashMap, HashSet};
use std::os::raw::{c_int};
use std::str::{from_utf8};

thread_local! {
  static BACKEND: Backend = Backend::new();
}

//const RESPONSE_OK:  &'static [u8] = b"ok";
//const RESPONSE_ERR: &'static [u8] = b"err";

pub enum APIAction {
  Hi,
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
    tcp.bind(("127.0.0.1", 9001));
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

impl Backend {
  pub fn _cleanup_request(buf: &mut UvBuf) {
    // TODO
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
  }

  pub fn _write_response(client: &UvStream, res_buf: &[u8]) {
    let mut backing_buf = BackingBuf::new_uninit(res_buf.len() + 4);
    (&mut backing_buf.as_mut_bytes()[ .. res_buf.len()]).copy_from_slice(res_buf);
    (&mut backing_buf.as_mut_bytes()[res_buf.len() .. res_buf.len() + 4]).copy_from_slice(&u32::to_le_bytes(0u32));
    let write_buf = UvBuf::from_raw_parts_unchecked(backing_buf.as_ptr() as _, backing_buf.len());
    BACKEND.with(|backend| {
      let mut store = backend.store.borrow_mut();
      if let Some(_) = store.buf.replace(backing_buf) {
        println!("DEBUG: Backend: _write: warning: backing buf was already stored!");
      }
    });
    let req = UvWrite::new();
    client.write::<Backend>(req, &write_buf);
  }

  pub fn _write_response2(client: &UvStream, res_buf: &[u8], res_buf2: &[u8]) {
    let mut backing_buf = BackingBuf::new_uninit(res_buf.len() + 4 + res_buf2.len());
    (&mut backing_buf.as_mut_bytes()[ .. res_buf.len()]).copy_from_slice(res_buf);
    (&mut backing_buf.as_mut_bytes()[res_buf.len() .. res_buf.len() + 4]).copy_from_slice(&u32::to_le_bytes(res_buf2.len() as u32));
    (&mut backing_buf.as_mut_bytes()[res_buf.len() + 4 .. ]).copy_from_slice(res_buf2);
    let write_buf = UvBuf::from_raw_parts_unchecked(backing_buf.as_ptr() as _, backing_buf.len());
    BACKEND.with(|backend| {
      let mut store = backend.store.borrow_mut();
      if let Some(_) = store.buf.replace(backing_buf) {
        println!("DEBUG: Backend: _write: warning: backing buf was already stored!");
      }
    });
    let req = UvWrite::new();
    client.write::<Backend>(req, &write_buf);
  }

  pub fn _handle_request(nread: usize, buf: &[u8]) -> Result<Option<Box<[u8]>>, ()> {
    // TODO
    let action = match &buf[ .. 4] {
      b"hi \n" => APIAction::Hi,
      b"put\n" => APIAction::Put,
      b"get\n" => APIAction::Get,
      _ => {
        return Err(());
      }
    };
    match action {
      APIAction::Hi => {
        return Ok(None);
      }
      APIAction::Put => {
      }
      APIAction::Get => {
      }
      _ => {
        // TODO
        unimplemented!();
      }
    }
    if nread <= 4 {
      return Err(());
    }
    let sort_start = 4;
    let mut sort_end = nread as usize;
    /*if buf[nread-1] != b'\n' {
      Backend::_write_response(&client, RESPONSE_ERR);
      return;
    }*/
    for p in 4 .. nread as usize {
      if buf[p] == b'\n' {
        sort_end = p;
        break;
      }
    }
    let sort: JournalEntrySort_ = match from_utf8(&buf[sort_start .. sort_end]) {
      Err(_) => {
        return Err(());
      }
      Ok(s) => {
        match s.parse() {
          Err(_) => {
            println!("DEBUG: Backend: read callback: warning: invalid sort: s = {:?}", s);
            return Err(());
          }
          Ok(sort) => {
            println!("DEBUG: Backend: read callback: sort = {:?}", sort);
            sort
          }
        }
      }
    };
    let item_start = sort_end + 1;
    let mut item_end = nread as usize;
    for p in item_start .. nread as usize {
      if buf[p] == b'\n' {
        item_end = p;
        break;
      }
    }
    let item_v = match deserialize_json_value(&buf[item_start .. item_end]) {
      Err(_) => {
        println!("DEBUG: Backend: read callback: warning: invalid item");
        return Err(());
      }
      Ok(v) => {
        println!("DEBUG: Backend: read callback: valid item");
        v
      }
    };
    // FIXME
    match sort {
      JournalEntrySort_::_Root => {
        println!("DEBUG: Backend: read callback: sort = {:?}", sort);
        let item = RootSort_::item_from_value(item_v);
      }
      JournalEntrySort_::Aikido => {
        println!("DEBUG: Backend: read callback: sort = {:?}", sort);
        let item = AikidoSort_::item_from_value(item_v);
      }
      JournalEntrySort_::ApproxOracle => {
        println!("DEBUG: Backend: read callback: sort = {:?}", sort);
        match ApproxOracleSort_::item_from_value(item_v.clone()) {
          Err(e) => {
            println!("DEBUG: Backend: read callback: failed to deserialize item from value = {:?} error = {:?}", item_v, e);
          }
          Ok(item) => {
            match action {
              APIAction::Hi => unreachable!(),
              APIAction::Put => {
                println!("DEBUG: Backend: read callback: put: append item = {:?}", item);
                BACKEND.with(|backend| {
                  let mut inner = backend.inner.borrow_mut();
                  inner.append_item(&item);
                });
              }
              APIAction::Get => {
                println!("DEBUG: Backend: read callback: get: lookup item = {:?}", item);
                return BACKEND.with(|backend| {
                  let mut inner = backend.inner.borrow_mut();
                  let key_item = item._to_key_item();
                  match inner._lookup_approx_oracle_item(&key_item) {
                    None => {
                      println!("DEBUG: Backend: read callback: get: lookup result = None");
                      // TODO
                      Ok(None)
                    }
                    Some(item) => {
                      let json_fmt = JsonFormat::new()
                          .ascii(true)
                          .colon(": ").unwrap()
                          .comma(", ").unwrap();
                      let s = json_fmt.to_string(&item).unwrap();
                      println!("DEBUG: Backend: read callback: get: lookup result = {:?}", s);
                      // TODO
                      Ok(Some(s.into_bytes().into()))
                    }
                  }
                });
              }
              _ => {
                // TODO
                unimplemented!();
              }
            }
          }
        }
      }
      JournalEntrySort_::Test => {
        println!("DEBUG: Backend: read callback: sort = {:?}", sort);
        match TestSort_::item_from_value(item_v.clone()) {
          Err(e) => {
            println!("DEBUG: Backend: read callback: failed to deserialize item from value = {:?} error = {:?}", item_v, e);
          }
          Ok(item) => {
            println!("DEBUG: Backend: read callback: append item = {:?}", item);
            /*let json_fmt = JsonFormat::new()
                .ascii(true)
                .colon(": ").unwrap()
                .comma(", ").unwrap();
            let s = json_fmt.to_string(&item).unwrap();*/
            BACKEND.with(|backend| {
              let mut inner = backend.inner.borrow_mut();
              //inner._append(s);
              inner.append_item(&item);
            });
          }
        }
      }
      _ => {
        println!("DEBUG: Backend: read callback: unsupported sort = {:?}", sort);
      }
    }
    Ok(None)
  }
}

impl UvReadCb for Backend {
  fn callback(client: UvStream, nread: isize, buf: &mut UvBuf) {
    println!("DEBUG: Backend: read callback: nread = {} buf.len = {}", nread, buf.len());
    if nread < 0 {
      let errno = nread as c_int;
      match errno {
        UV_EOF => {
          println!("DEBUG: Backend: read callback: eof");
          let req = UvShutdown::new();
          client.shutdown::<Backend>(req);
        }
        UV_ECONNRESET => {
          println!("DEBUG: Backend: read callback: econnreset");
          // TODO: close.
        }
        UV_ENOTCONN => {
          println!("DEBUG: Backend: read callback: enotconn");
          // TODO: close.
        }
        _ => {
          // FIXME
          println!("DEBUG: Backend: read callback: error = {}", errno);
        }
      }
      /*let (backing_ptr, backing_len) = buf.take_raw_parts();
      if let Some(mut backing_buf) = BackingBuf::maybe_from_raw_parts_unchecked(backing_ptr as _, backing_len) {
        BACKEND.with(|backend| {
          let mut store = backend.store.borrow_mut();
          if !store.buf.remove(&backing_buf) {
            println!("DEBUG: Backend: read callback: warning: backing buf was NOT in store!");
          }
          println!("DEBUG: Backend: read callback: free backing buf...");
          backing_buf.free_unchecked();
        });
      }*/
      return Backend::_cleanup_request(buf);
    }
    if nread < 4 {
      println!("DEBUG: Backend: read callback: truncated");
      let req = UvShutdown::new();
      client.shutdown::<Backend>(req);
      return Backend::_cleanup_request(buf);
    }
    let nread = nread as usize;
    match buf.as_bytes() {
      None => {
        println!("DEBUG: Backend: read callback: invalid buffer");
        let req = UvShutdown::new();
        client.shutdown::<Backend>(req);
        return Backend::_cleanup_request(buf);
      }
      Some(buf) => {
        match Backend::_handle_request(nread, buf) {
          Err(_) => {
            println!("DEBUG: Backend: read callback: err: write response...");
            Backend::_write_response(&client, b"err\n");
          }
          Ok(None) => {
            println!("DEBUG: Backend: read callback: ok: write short response...");
            // TODO: always assure buffer lifetime.
            //let res_str = format!("Hello, world! {}\n", nread);
            //let res_buf = res_str.as_bytes();
            Backend::_write_response(&client, b"ok \n");
          }
          Ok(Some(res_payload_buf)) => {
            println!("DEBUG: Backend: read callback: ok: write full response...");
            // TODO: full response.
            Backend::_write_response2(&client, b"ok \n", &res_payload_buf);
          }
        }
      }
    }
    Backend::_cleanup_request(buf);
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
