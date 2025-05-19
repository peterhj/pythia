pub const TARGET: &'static str = include_str!(concat!(env!("OUT_DIR"), "/target.txt"));
pub const CWD: &'static str = include_str!(concat!(env!("OUT_DIR"), "/cwd.txt"));
pub const GIT_COMMIT_HASH: &'static str = include_str!(concat!(env!("OUT_DIR"), "/git_commit_hash.txt"));
pub const GIT_COMMIT_MODIFIED_STR: &'static str = include_str!(concat!(env!("OUT_DIR"), "/git_modified.txt"));

#[derive(Clone, Copy, Debug)]
pub struct Triple {
  pub arch: &'static str,
  pub vendor: &'static str,
  pub system: &'static str,
  pub abi: Option<&'static str>,
}

#[inline]
pub fn triple() -> Triple {
  let mut trip = Triple{
    arch: "",
    vendor: "",
    system: "",
    abi: None,
  };
  for (i, s) in TARGET.split("-").enumerate() {
    match i {
      0 => {
        trip.arch = s;
      }
      1 => {
        trip.vendor = s;
      }
      2 => {
        trip.system = s;
      }
      3 => {
        trip.abi = Some(s);
      }
      _ => panic!("bug")
    }
  }
  trip
}

#[inline]
pub fn git_commit_hash() -> &'static str {
  GIT_COMMIT_HASH
}

#[inline]
pub fn git_commit_modified() -> bool {
  match GIT_COMMIT_MODIFIED_STR {
    "false" => false,
    "true"  => true,
    _ => panic!("bug")
  }
}
