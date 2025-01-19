pub const CWD: &'static str = include_str!(concat!(env!("OUT_DIR"), "/cwd.txt"));
pub const GIT_COMMIT_HASH: &'static str = include_str!(concat!(env!("OUT_DIR"), "/git_commit_hash.txt"));
pub const GIT_COMMIT_MODIFIED_STR: &'static str = include_str!(concat!(env!("OUT_DIR"), "/git_modified.txt"));

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
