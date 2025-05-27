extern crate _data;

fn main() {
  let mut data = _data::arxiv::ArxivData::default();
  data._load();
}
