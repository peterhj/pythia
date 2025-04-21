CARGO := cargo --offline
RUSTC := rustc

.PHONY: all
all: release

.PHONY: v version
v: version
version:
	$(RUSTC) --version > rustc.version
	$(CARGO) --version > cargo.version

.PHONY: dev debug
dev: debug
debug:
	$(CARGO) build --lib --bins

.PHONY: rel release
rel: release
release:
	$(CARGO) build --release --lib --bins

.PHONY: clean
clean:
	rm -rf target

.PHONY: o oracle boot-oracle
o: boot-oracle
oracle: boot-oracle
boot-oracle:
	RUST_BACKTRACE=1 ./target/release/boot-oracle

.PHONY: devint dev-int dev-boot-interp dev-boot-interp-test
devint: dev-boot-interp-test
dev-int: dev-boot-interp-test
dev-boot-interp: dev-boot-interp-test
dev-boot-interp-test:
	./target/debug/boot-interp

.PHONY: i in int interp boot-interp boot-interp-test
i: boot-interp-test
in: boot-interp-test
int: boot-interp-test
interp: boot-interp-test
boot-interp: boot-interp-test
boot-interp-test:
	./target/release/boot-interp-test

.PHONY: 1 i1 int1 interp1
1: boot-interp-test-1
i1: boot-interp-test-1
int1: boot-interp-test-1
interp1: boot-interp-test-1
boot-interp-test-1:
	./target/release/boot-interp-test-1

.PHONY: last ilast int-last interp-last
last: boot-interp-test-last
ilast: boot-interp-test-last
int-last: boot-interp-test-last
interp-last: boot-interp-test-last
boot-interp-test-last:
	./target/release/boot-interp-test-last
