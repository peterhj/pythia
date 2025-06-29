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

use std::io::{ErrorKind as IoErrorKind};
use std::str::{from_utf8};
use std::sync::{Arc, Mutex};

static BACKEND: Lazy<JournalAPIBackend> = Lazy::new(|| JournalAPIBackend::new());

#[derive(Clone, Copy, Debug)]
enum JournalAPIAction {
  Hi,
  Put,
  Get,
}

struct JournalAPIBackend {
  inner: Arc<Mutex<JournalBackend>>,
}

impl JournalAPIBackend {
  fn new() -> JournalAPIBackend {
    let inner = JournalBackend::cold_start();
    JournalAPIBackend{
      inner: Arc::new(Mutex::new(inner)),
    }
  }

  fn _handle_request(action: JournalAPIAction, buf: &[u8]) -> Result<Option<Box<[u8]>>, ()> {
    // TODO
    println!("DEBUG: JournalAPIBackend::_handle_request: action = {:?}", action);
    match action {
      JournalAPIAction::Hi => {
        return Ok(None);
      }
      JournalAPIAction::Put => {
      }
      JournalAPIAction::Get => {
      }
      _ => {
        // TODO
        unimplemented!();
      }
    }
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
            println!("DEBUG: JournalAPIBackend: read callback: warning: invalid sort: s = {:?}", s);
            return Err(());
          }
          Ok(sort) => {
            println!("DEBUG: JournalAPIBackend: read callback: sort = {:?}", sort);
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
        println!("DEBUG: JournalAPIBackend: read callback: warning: invalid item");
        return Err(());
      }
      Ok(v) => {
        println!("DEBUG: JournalAPIBackend: read callback: valid item");
        v
      }
    };
    // FIXME
    match sort {
      JournalEntrySort_::_Root => {
        println!("DEBUG: JournalAPIBackend: read callback: sort = {:?}", sort);
        let item = RootSort_::item_from_value(item_v);
      }
      JournalEntrySort_::Aikido => {
        println!("DEBUG: JournalAPIBackend: read callback: sort = {:?}", sort);
        let item = AikidoSort_::item_from_value(item_v);
      }
      JournalEntrySort_::ApproxOracle => {
        println!("DEBUG: JournalAPIBackend: read callback: sort = {:?}", sort);
        match ApproxOracleSort_::item_from_value(item_v.clone()) {
          Err(e) => {
            println!("DEBUG: JournalAPIBackend: read callback: failed to deserialize item from value = {:?} error = {:?}", item_v, e);
          }
          Ok(item) => {
            match action {
              JournalAPIAction::Hi => unreachable!(),
              JournalAPIAction::Put => {
                println!("DEBUG: JournalAPIBackend: read callback: put: append item = {:?}", item);
                {
                  let backend = &*BACKEND;
                  let mut inner = backend.inner.lock().unwrap();
                  inner.append_item(&item);
                }
              }
              JournalAPIAction::Get => {
                println!("DEBUG: JournalAPIBackend: read callback: get: lookup item = {:?}", item);
                return ({
                  let backend = &*BACKEND;
                  let mut inner = backend.inner.lock().unwrap();
                  let key_item = item._to_key_item();
                  match inner._lookup_approx_oracle_item(&key_item) {
                    None => {
                      println!("DEBUG: JournalAPIBackend: read callback: get: lookup result = None");
                      // TODO
                      Ok(None)
                    }
                    Some(item) => {
                      let json_fmt = JsonFormat::new()
                          .ascii(true)
                          .colon(": ").unwrap()
                          .comma(", ").unwrap();
                      let s = json_fmt.to_string(&item).unwrap();
                      println!("DEBUG: JournalAPIBackend: read callback: get: lookup result = {:?}", s);
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
        println!("DEBUG: JournalAPIBackend: read callback: sort = {:?}", sort);
        match TestSort_::item_from_value(item_v.clone()) {
          Err(e) => {
            println!("DEBUG: JournalAPIBackend: read callback: failed to deserialize item from value = {:?} error = {:?}", item_v, e);
          }
          Ok(item) => {
            println!("DEBUG: JournalAPIBackend: read callback: append item = {:?}", item);
            {
              let backend = &*BACKEND;
              let mut inner = backend.inner.lock().unwrap();
              inner.append_item(&item);
            };
          }
        }
      }
      _ => {
        println!("DEBUG: JournalAPIBackend: read callback: unsupported sort = {:?}", sort);
      }
    }
    Ok(None)
  }

  async fn handle_stream(stream: TcpStream) -> Result<(), String> {
    let addr = stream.peer_addr().map_err(|e| format!("{:?}", e))?;
    println!("DEBUG: JournalAPIBackend::handle_stream: addr = {}", addr);
    let mut reader = stream.clone();
    let mut writer = stream;
    println!("DEBUG: JournalAPIBackend::handle_stream: read...");
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
        b"hi \n" => JournalAPIAction::Hi,
        b"put\n" => JournalAPIAction::Put,
        b"get\n" => JournalAPIAction::Get,
        _ => {
          return Err(format!("invalid action encoding"));
        }
      };
      match JournalAPIBackend::_handle_request(action, &buf) {
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
    println!("DEBUG: JournalAPIBackend::accept_loop: addr = {}", addr);
    let mut incoming = listener.incoming();
    while let Some(stream) = incoming.next().await {
      println!("DEBUG: JournalAPIBackend::accept_loop: next stream");
      let stream = stream.map_err(|e| format!("{:?}", e))?;
      async_std::task::spawn(async {
        match JournalAPIBackend::handle_stream(stream).await {
          Err(e) => {
            println!("DEBUG: JournalAPIBackend::accept_loop: stream error = {:?}", e);
          }
          Ok(_) => {}
        }
      });
    }
    Ok(())
  }
}

fn main() -> Result<(), String> {
  async_std::task::block_on(JournalAPIBackend::accept_loop("127.0.0.1:9001"))
}
