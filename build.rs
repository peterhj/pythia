//#[cfg(feature = "pyo3")]
//extern crate pyo3_build_config;

use std::fs::{File};
use std::io::{Write, BufRead, Cursor};
use std::path::{PathBuf};
use std::process::{Command};

fn _main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=.git/logs/HEAD");

  let triple = std::env::var("TARGET").unwrap();
  let cwd = std::env::current_dir().unwrap();
  let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
  let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

  {
    let mut file = File::create(out_dir.join("target.txt")).unwrap();
    write!(&mut file, "{}", triple).unwrap();
  }

  {
    let mut file = File::create(out_dir.join("cwd.txt")).unwrap();
    write!(&mut file, "{}", cwd.display()).unwrap();
  }

  let out = Command::new("git")
    .current_dir(&manifest_dir)
    .arg("log")
    .arg("-n").arg("1")
    .arg("--format=%H")
    .output().unwrap();
  if !out.status.success() {
    panic!("`git log` failed with exit status: {:?}", out.status);
  }
  match Cursor::new(out.stdout).lines().next() {
    None => panic!("`git log` did not print the commit hash"),
    Some(line) => {
      let line = line.unwrap();
      let mut file = File::create(out_dir.join("git_commit_hash.txt")).unwrap();
      write!(&mut file, "{}", line).unwrap();
    }
  }

  let mut changed = false;
  let out = Command::new("git")
    .current_dir(&manifest_dir)
    .arg("diff")
    .arg("--stat")
    .output().unwrap();
  if !out.status.success() {
    panic!("`git diff` failed with exit status: {:?}", out.status);
  }
  match Cursor::new(out.stdout).lines().next() {
    None => {}
    Some(_) => {
      changed = true;
    }
  }
  {
    let mut file = File::create(out_dir.join("git_modified.txt")).unwrap();
    write!(&mut file, "{}", changed).unwrap();
  }
}

#[cfg(not(feature = "pyo3"))]
fn main() {
  _main();
}

#[cfg(all(feature = "pyo3", not(target_os = "macos")))]
fn _fixup_rpath_bins() {
}

#[cfg(all(feature = "pyo3", target_os = "macos"))]
fn _fixup_rpath_bins() {
  // NB: see:
  // - https://pyo3.rs/v0.21.0/building-and-distribution.html?highlight=rpath#macos
  // - https://github.com/PyO3/pyo3/issues/1800#issuecomment-906786649
  println!("cargo:rustc-link-arg-bins=-Wl,-rpath,/Applications/Xcode.app/Contents/Developer/Library/Frameworks");
  println!("cargo:rustc-link-arg-bins=-Wl,-rpath,/Library/Developer/CommandLineTools/Library/Frameworks");
}

#[cfg(feature = "pyo3")]
fn main() {
  //pyo3_build_config::add_extension_module_link_args();
  _fixup_rpath_bins();
  _main();
}
