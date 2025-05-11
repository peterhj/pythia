extern crate pythia;

use pythia::algo::extract::*;

static TESTVEC: &'static str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/test/extract_0.txt"));

fn main() {
  let markdown = MarkdownCodeBlocks::parse(TESTVEC);
  for block in markdown.blocks {
    println!("Language: {:?}", block.language);
    println!("Code:\n{}", block.content);
  }
}
