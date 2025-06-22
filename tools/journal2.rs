extern crate async_std;
extern crate once_cell;
extern crate pythia;

use async_std::prelude::*;
use async_std::net::{TcpStream, TcpListener, ToSocketAddrs};
use once_cell::sync::{Lazy};
use pythia::algo::json::{JsonFormat, deserialize_json_value};
use pythia::journal::{
  JournalEntrySort_,
  JournalBackend,
  RootSort_,
  AikidoSort_,
  ApproxOracleSort_,
  TestSort_,
};

//use std::cell::{RefCell};
use std::io::{ErrorKind as IoErrorKind};
use std::str::{from_utf8};
use std::sync::{Arc, Mutex};

static BACKEND: Lazy<APIBackend> = Lazy::new(|| APIBackend::new());

#[derive(Clone, Copy, Debug)]
enum APIAction {
  Hi,
  Put,
  Get,
}

struct APIBackend {
  inner: Arc<Mutex<JournalBackend>>,
}

impl APIBackend {
  fn new() -> APIBackend {
    let inner = JournalBackend::cold_start();
    APIBackend{
      inner: Arc::new(Mutex::new(inner)),
    }
  }

  fn _handle_request(action: APIAction, buf: &[u8]) -> Result<Option<Box<[u8]>>, ()> {
    // TODO
    /*if buf.len() < 4 {
      return Err(());
    }
    let action = match &buf[ .. 4] {
      b"hi \n" => APIAction::Hi,
      b"put\n" => APIAction::Put,
      b"get\n" => APIAction::Get,
      _ => {
        return Err(());
      }
    };*/
    println!("DEBUG: APIBackend::_handle_request: action = {:?}", action);
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
    //let sort_start = 4;
    let sort_start = 0;
    let mut sort_end = buf.len();
    for p in sort_start .. buf.len() {
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
            println!("DEBUG: APIBackend: read callback: warning: invalid sort: s = {:?}", s);
            return Err(());
          }
          Ok(sort) => {
            println!("DEBUG: APIBackend: read callback: sort = {:?}", sort);
            sort
          }
        }
      }
    };
    // TODO
    let item_start = sort_end + 1;
    let mut item_end = buf.len();
    for p in item_start .. buf.len() {
      if buf[p] == b'\n' {
        item_end = p;
        break;
      }
    }
    let item_v = match deserialize_json_value(&buf[item_start .. item_end]) {
      Err(_) => {
        println!("DEBUG: APIBackend: read callback: warning: invalid item");
        return Err(());
      }
      Ok(v) => {
        println!("DEBUG: APIBackend: read callback: valid item");
        v
      }
    };
    // FIXME
    match sort {
      JournalEntrySort_::_Root => {
        println!("DEBUG: APIBackend: read callback: sort = {:?}", sort);
        let item = RootSort_::item_from_value(item_v);
      }
      JournalEntrySort_::Aikido => {
        println!("DEBUG: APIBackend: read callback: sort = {:?}", sort);
        let item = AikidoSort_::item_from_value(item_v);
      }
      JournalEntrySort_::ApproxOracle => {
        println!("DEBUG: APIBackend: read callback: sort = {:?}", sort);
        match ApproxOracleSort_::item_from_value(item_v.clone()) {
          Err(e) => {
            println!("DEBUG: APIBackend: read callback: failed to deserialize item from value = {:?} error = {:?}", item_v, e);
          }
          Ok(item) => {
            match action {
              APIAction::Hi => unreachable!(),
              APIAction::Put => {
                println!("DEBUG: APIBackend: read callback: put: append item = {:?}", item);
                {
                  let backend = &*BACKEND;
                  let mut inner = backend.inner.lock().unwrap();
                  inner.append_item(&item);
                }
              }
              APIAction::Get => {
                println!("DEBUG: APIBackend: read callback: get: lookup item = {:?}", item);
                return ({
                  let backend = &*BACKEND;
                  let mut inner = backend.inner.lock().unwrap();
                  let key_item = item._to_key_item();
                  match inner._lookup_approx_oracle_item(&key_item) {
                    None => {
                      println!("DEBUG: APIBackend: read callback: get: lookup result = None");
                      // TODO
                      Ok(None)
                    }
                    Some(item) => {
                      let json_fmt = JsonFormat::new()
                          .ascii(true)
                          .colon(": ").unwrap()
                          .comma(", ").unwrap();
                      let s = json_fmt.to_string(&item).unwrap();
                      println!("DEBUG: APIBackend: read callback: get: lookup result = {:?}", s);
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
        println!("DEBUG: APIBackend: read callback: sort = {:?}", sort);
        match TestSort_::item_from_value(item_v.clone()) {
          Err(e) => {
            println!("DEBUG: APIBackend: read callback: failed to deserialize item from value = {:?} error = {:?}", item_v, e);
          }
          Ok(item) => {
            println!("DEBUG: APIBackend: read callback: append item = {:?}", item);
            {
              let backend = &*BACKEND;
              let mut inner = backend.inner.lock().unwrap();
              inner.append_item(&item);
            };
          }
        }
      }
      _ => {
        println!("DEBUG: APIBackend: read callback: unsupported sort = {:?}", sort);
      }
    }
    Ok(None)
  }
}

async fn handle_request(stream: TcpStream) -> Result<(), String> {
  let addr = stream.peer_addr().map_err(|e| format!("{:?}", e))?;
  println!("DEBUG: handle_request: addr = {}", addr);
  let mut reader = stream.clone();
  let mut writer = stream;
  // TODO: below is just the echo server example.
  //async_std::io::copy(&mut reader, &mut writer).await.map_err(|e| format!("{:?}", e))?;
  //writeln!(&mut writer, "Goodbye, world!").await.map_err(|e| format!("{:?}", e))?;
  // TODO: async version of Backend::_handle_request (libuv-based journal).
  println!("DEBUG: handle_request: read...");
  let mut buf = Vec::new();
  let mut enc_action = [0u8; 4];
  let mut enc_len = [0u8; 4];
  loop {
    match reader.read_exact(&mut enc_action).await {
      Err(e) => {
        if e.kind() == IoErrorKind::UnexpectedEof {
          return Ok(());
        }
        return Err(format!("{:?}", e));
      }
      Ok(_) => {}
    }
    reader.read_exact(&mut enc_len).await.map_err(|e| format!("{:?}", e))?;
    buf.resize(u32::from_le_bytes(enc_len) as usize, 0);
    reader.read_exact(&mut buf).await.map_err(|e| format!("{:?}", e))?;
    let action = match &enc_action {
      b"hi \n" => APIAction::Hi,
      b"put\n" => APIAction::Put,
      b"get\n" => APIAction::Get,
      _ => {
        return Err(format!("invalid action encoding"));
      }
    };
    match APIBackend::_handle_request(action, &buf) {
      Err(_) => {
        writer.write_all(b"err\n").await.map_err(|e| format!("{:?}", e))?;
      }
      Ok(wbuf) => {
        writer.write_all(b"ok \n").await.map_err(|e| format!("{:?}", e))?;
        if let Some(wbuf) = wbuf {
          let wbuf_len = wbuf.len();
          let wbuf_len_enc = u32::to_le_bytes(wbuf_len as u32);
          writer.write_all(&wbuf_len_enc).await.map_err(|e| format!("{:?}", e))?;
          writer.write_all(&*wbuf).await.map_err(|e| format!("{:?}", e))?;
        } else {
          writer.write_all(&[0, 0, 0, 0]).await.map_err(|e| format!("{:?}", e))?;
        }
      }
    }
  }
  Ok(())
}

async fn accept_loop(addr: impl ToSocketAddrs) -> Result<(), String> {
  let listener = TcpListener::bind(addr).await.map_err(|e| format!("{:?}", e))?;
  let addr = listener.local_addr().map_err(|e| format!("{:?}", e))?;
  println!("DEBUG: accept_loop: addr = {}", addr);
  let mut incoming = listener.incoming();
  while let Some(stream) = incoming.next().await {
    println!("DEBUG: accept_loop: next stream");
    let stream = stream.map_err(|e| format!("{:?}", e))?;
    async_std::task::spawn(async {
      match handle_request(stream).await {
        Err(e) => {
          println!("DEBUG: accept_loop: stream error = {:?}", e);
        }
        Ok(_) => {}
      }
    });
  }
  Ok(())
}

fn main() -> Result<(), String> {
  async_std::task::block_on(accept_loop("127.0.0.1:9001"))
}
