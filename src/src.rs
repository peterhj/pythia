macro_rules! _src {
  ($name:ident, $path:expr) => {
    pub const $name: &'static str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $path));
  };
}

_src!(SRC, "/src/src.rs");
_src!(_CARGO_TOML, "/Cargo.toml");
_src!(_CARGO_LOCK, "/Cargo.lock");
_src!(_BUILD, "/build.rs");
_src!(_EXTLIB_APPROX_ORACLE_PY, "/_extlib/_approx_oracle.py");
_src!(LIB, "/src/lib.rs");
_src!(BUILD, "/src/build.rs");
_src!(TEST_DATA, "/src/test_data.rs");
_src!(_EXTLIB, "/src/_extlib.rs");
_src!(ALGO, "/src/algo/mod.rs");
_src!(ALGO_CELL, "/src/algo/cell.rs");
_src!(ALGO_RC, "/src/algo/rc.rs");
_src!(ALGO_STR, "/src/algo/str.rs");
_src!(ALGO_TOKEN, "/src/algo/token.rs");
_src!(CLOCK, "/src/clock.rs");
_src!(INTERP, "/src/interp.rs");
_src!(INTERP_PRELUDE, "/src/interp/prelude.rs");
_src!(JOURNAL, "/src/journal.rs");
_src!(ORACLE, "/src/oracle.rs");
_src!(PANICK, "/src/panick.rs");
_src!(PARSE, "/src/parse.rs");
_src!(SMP, "/src/smp/mod.rs");
_src!(SMP_LINUX, "/src/smp/linux.rs");
_src!(SMP_MACOS, "/src/smp/macos.rs");
_src!(SYS, "/src/sys/mod.rs");
_src!(SYS_MMAP, "/src/sys/mmap.rs");
