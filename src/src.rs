macro_rules! _src {
  ($name:ident, $path:expr) => {
    pub const $name: &'static str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $path));
  };
}

_src!(SRC, "/src/src.rs");
_src!(__CARGO_TOML, "/Cargo.toml");
_src!(__CARGO_LOCK, "/Cargo.lock");
_src!(__BUILD, "/build.rs");
_src!(__EXTLIB_APPROX_ORACLE_PY, "/_extlib/_approx_oracle.py");
_src!(LIB, "/src/lib.rs");
_src!(BUILD, "/src/build.rs");
_src!(TEST_DATA, "/src/test_data.rs");
_src!(_EXTLIB, "/src/_extlib.rs");
_src!(AIKIDO, "/src/aikido.rs");
_src!(ALGO, "/src/algo.rs");
_src!(ALGO_BLAKE2S, "/src/algo/blake2s.rs");
_src!(ALGO_CELL, "/src/algo/cell.rs");
_src!(ALGO_EXTRACT, "/src/algo/extract.rs");
_src!(ALGO_RC, "/src/algo/rc.rs");
_src!(ALGO_STR, "/src/algo/str.rs");
_src!(ALGO_TOKEN, "/src/algo/token.rs");
_src!(CLOCK, "/src/clock.rs");
_src!(INTERP, "/src/interp.rs");
_src!(INTERP_PRELUDE, "/src/interp/prelude.rs");
_src!(INTERP_TEST, "/src/interp_test.rs");
_src!(JOURNAL, "/src/journal.rs");
_src!(META, "/src/meta.rs");
_src!(ORACLE, "/src/oracle.rs");
_src!(PANICK, "/src/panick.rs");
_src!(PARSE, "/src/parse.rs");
_src!(SMP, "/src/smp.rs");
_src!(SMP_LINUX, "/src/smp/linux.rs");
_src!(SMP_MACOS, "/src/smp/macos.rs");
_src!(SYS, "/src/sys.rs");
_src!(SYS_MMAP, "/src/sys/mmap.rs");
_src!(TAP, "/src/tap.rs");
_src!(UTIL, "/src/util.rs");
_src!(UTIL_HEX, "/src/util/hex.rs");
