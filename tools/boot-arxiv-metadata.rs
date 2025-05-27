extern crate _data;

fn main() {
  let mut data = _data::arxiv::ArxivMetadata::default();
  data._load();
}
