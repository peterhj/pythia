pub const SRC: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/src.rs"
));
pub const LIB: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/lib.rs"
));
pub const BUILD: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/build.rs"
));
pub const TEST_DATA: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/test_data.rs"
));
pub const _EXTLIB: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/_extlib.rs"
));
pub const ALGO: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algo/mod.rs"
));
pub const ALGO_CELL: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algo/cell.rs"
));
pub const ALGO_RC: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algo/rc.rs"
));
pub const ALGO_STR: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algo/str.rs"
));
pub const ALGO_TOKEN: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algo/token.rs"
));
pub const CLOCK: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/clock.rs"
));
pub const INTERP: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/interp.rs"
));
pub const INTERP_PRELUDE: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/interp/prelude.rs"
));
pub const ORACLE: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/oracle.rs"
));
pub const PANICK: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/panick.rs"
));
pub const PARSE: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/parse.rs"
));
pub const SMP: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/smp/mod.rs"
));
pub const SMP_LINUX: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/smp/linux.rs"
));
pub const SMP_MACOS: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/smp/macos.rs"
));
pub const STORE: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/store.rs"
));
pub const SYS: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/sys/mod.rs"
));
pub const SYS_MMAP: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/sys/mmap.rs"
));
