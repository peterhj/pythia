use crate::util::encoding::*;
use crate::util::gzip::*;
use crate::util::tar::*;

use walkdir::{WalkDir};

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File};
use std::io::{Read, Seek, Cursor, copy as iocopy};
use std::path::{PathBuf, Path};
use std::str::{from_utf8};

#[derive(Debug)]
pub struct ArxivDatum {
  pub attrs: ArxivDatumAttrs,
  pub paths: Vec<PathBuf>,
  pub value: Vec<u8>,
}

#[derive(Default, Debug)]
pub struct ArxivDatumAttrs {
  pub toplevel: bool,
  pub tar_like: bool,
  pub tar_chk:  Option<bool>,
  pub text_asc: bool,
  pub text_enc: Option<&'static Encoding>,
  //pub text_amb: bool,
}

#[derive(Default)]
pub struct ArxivData {
  datum_map: BTreeMap<usize, ArxivDatum>,
}

impl ArxivData {
  pub fn _load(&mut self) {
    for e in WalkDir::new("data/arxiv/arxiv_data").into_iter() {
      let e = e.unwrap();
      let p = e.path();
      let p_s = p.as_os_str().to_str().unwrap();
      if p_s.ends_with(".tar") {
        println!("{:?}", p);
        let mut attrs = ArxivDatumAttrs::default();
        attrs.toplevel = true;
        let paths = vec![p.to_owned()];
        let file = File::open(p).unwrap();
        let tar = TarFile::new(file);
        self._load_tar(attrs, paths, tar);
      }
    }
  }

  pub fn _load_tar<F: Read + Seek>(&mut self, attrs: ArxivDatumAttrs, paths: Vec<PathBuf>, tar: TarFile<F>) {
    let it = tar.entry_iter();
    for e in it {
      let e = e.unwrap();
      let p = &e.path;
      let p_s = p.as_os_str().to_str().unwrap();
      if attrs.toplevel && p_s.ends_with(".gz") {
        let mut e_attrs = ArxivDatumAttrs::default();
        let mut e_paths = paths.clone();
        e_paths.push(p.to_owned());
        let e_cat = p_s.get(5 .. p_s.len() - 10).unwrap();
        match (attrs.toplevel, e_cat) {
          (true, "cs") |
          (false, _) => {
            let e_file = tar.slice_subreader_at(e.entry_pos, e.entry_sz);
            let mut e_file = GzipReader::new(e_file);
            let mut e_buf = Vec::new();
            iocopy(&mut e_file, &mut e_buf).unwrap();
            if e_buf.len() >= 512 && e_buf.len() % 512 == 0 {
              e_attrs.tar_like = true;
              match check_tar_header_block(&e_buf[ .. 512]) {
                Ok(true) => {
                  e_attrs.tar_chk = Some(true);
                }
                Ok(false) => {
                  e_attrs.tar_chk = Some(false);
                }
                Err(_) => {
                  e_attrs.tar_chk = None;
                }
              }
            }
            if e_attrs.tar_like && e_attrs.tar_chk == Some(true) {
              if e_paths.len() > 2 {
                assert!(!attrs.toplevel);
                println!("DEBUG: ArxivData::_load_tar: warning: overly nested tar");
              }
              let e_buf = Cursor::new(e_buf);
              let e_tar = TarFile::new(e_buf);
              println!("DEBUG: ArxivData::_load_tar: gz tar: {:?}", p);
              self._load_tar(e_attrs, e_paths, e_tar);
            } else {
              let mut enc_det = EncodingDetector::new();
              e_attrs.text_asc = !enc_det.feed(&e_buf, true);
              let (enc, amb) = enc_det.guess_assess(None, true);
              e_attrs.text_enc = Some(enc);
              //e_attrs.text_amb = amb;
              println!("DEBUG: ArxivData::_load_tar: gz text: {:?} {:?}", p, e_attrs.text_enc);
              self._load_text(e_attrs, e_paths, e_buf);
            }
          }
          _ => {}
        }
      } else if !attrs.toplevel && p_s.ends_with(".tex") {
        let mut e_attrs = ArxivDatumAttrs::default();
        let mut e_paths = paths.clone();
        e_paths.push(p.to_owned());
        let mut e_file = tar.slice_subreader_at(e.entry_pos, e.entry_sz);
        let mut e_buf = Vec::new();
        iocopy(&mut e_file, &mut e_buf).unwrap();
        let mut enc_det = EncodingDetector::new();
        e_attrs.text_asc = !enc_det.feed(&e_buf, true);
        let (enc, amb) = enc_det.guess_assess(None, true);
        e_attrs.text_enc = Some(enc);
        //e_attrs.text_amb = amb;
        println!("DEBUG: ArxivData::_load_tar: tex: {:?}", p);
        self._load_text(e_attrs, e_paths, e_buf);
      } else if !attrs.toplevel && p_s.ends_with(".bib") {
        println!("DEBUG: ArxivData::_load_tar: todo: bib: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".bbl") {
        println!("DEBUG: ArxivData::_load_tar: todo: bbl: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".bst") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip bst: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".sty") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip sty: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".cls") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip cls: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".clo") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip clo: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".log") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip log: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".out") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip out: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".eps") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip eps: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".ps") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip ps: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".pslatex") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip pslatex: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".pstex") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip pstex: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".pstex_t") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip pstex_t: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".fig") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip fig: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".pic") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip pic: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".pdf") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip pdf: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".cry") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip cry: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".jpg") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip jpg: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".gnu") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip gnu: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with(".bak") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip bak: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with("Makefile") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip Makefile: {:?}", p);
      } else if !attrs.toplevel && p_s.ends_with("/") {
        //println!("DEBUG: ArxivData::_load_tar: warning: skip dir: {:?}", p);
      } else if !attrs.toplevel {
        println!("DEBUG: ArxivData::_load_tar: warning: unhandled entry: {:?} {:?}", p.extension(), p);
      }
    }
    if attrs.toplevel {
      //println!("DEBUG: ArxivDatum::_load: num datums = {}", self.datum_map.len());
      let mut total_sz = 0;
      for (_, datum) in self.datum_map.iter() {
        total_sz += datum.value.len();
      }
      println!("DEBUG: ArxivDatum::_load: num datums = {} total sz = {}", self.datum_map.len(), total_sz);
    }
  }

  pub fn _load_text(&mut self, attrs: ArxivDatumAttrs, paths: Vec<PathBuf>, buf: Vec<u8>) {
    let idx = self.datum_map.len();
    if idx == 0 && attrs.text_enc == Some(UTF_8) {
      println!("DEBUG: ArxivDatum::_load_text: dump first utf8");
      let mut f = File::create(".tmp.txt").unwrap();
      let mut src = Cursor::new(&buf);
      iocopy(&mut src, &mut f).unwrap();
    }
    self.datum_map.insert(idx, ArxivDatum{
      attrs,
      paths,
      value: buf,
    });
  }
}

#[derive(Debug, PartialEq)]
enum ParserState {
  Empty,
  Id,
  Category,
  //OtherMeta,
}

#[derive(Clone, Default, Debug)]
pub struct ArxivMetadatum {
  pub id: Option<String>,
  pub id_set: Option<(u16, u8)>,
  pub categories: Vec<String>,
  pub sets: Vec<String>,
}

fn parse_arxiv_metadata(input: &str) -> Vec<ArxivMetadatum> {
  let mut articles = Vec::new();
  let mut current_article = ArxivMetadatum::default();
  let mut state = ParserState::Empty;
  let mut current_tag = String::new();

  for line in input.lines() {
    let line = line.trim();

    // Skip empty lines and XML declaration
    if line.is_empty() || line.starts_with("<?xml") {
      continue;
    }

    // Check for record boundaries
    if line.starts_with("<record>") {
      // Start new article
      current_article = ArxivMetadatum::default();
      state = ParserState::Empty;
    } else if line.starts_with("</record>") {
      // Finish current article and add to collection
      if current_article.id.is_some() {
        articles.push(current_article);
      }
      current_article = ArxivMetadatum::default();
      state = ParserState::Empty;
    } else if line.starts_with("<id>") && line.ends_with("</id>") {
      // Extract ID
      let content = line
        .trim_start_matches("<id>")
        .trim_end_matches("</id>")
        .trim()
        .to_string();
      current_article.id = Some(content.clone());
      for part in content.rsplit("/") {
        if part.len() > 0 {
          let c = part.chars().next().unwrap();
          if c >= '0' && c <= '9' {
            for part in part.split(".") {
              let id_year: u16 = part.get( .. 2).unwrap().parse().unwrap();
              let id_mon: u8 = part.get(2 .. 4).unwrap().parse().unwrap();
              let id_year = if id_year >= 90 {
                1900 + id_year
              } else if id_year < 50 {
                2000 + id_year
              } else {
                unimplemented!();
              };
              current_article.id_set = Some((id_year, id_mon));
              break;
            }
            break;
          }
        }
      }
      /*for part in content.split("/") {
        if part.len() > 0 {
          let c = part.chars().next().unwrap();
          if c >= '0' && c <= '9' {
            current_article.id = Some(part.to_owned());
            break;
          }
        }
      }
      assert!(current_article.id.is_some());*/
      state = ParserState::Id;
    } else if line.starts_with("<categories>") && line.ends_with("</categories>") {
      // Extract categories
      let content = line
        .trim_start_matches("<categories>")
        .trim_end_matches("</categories>")
        .trim()
        .to_string();
      let mut sets = BTreeSet::new();
      for category in content.split_whitespace() {
        for part in category.split(".") {
          sets.insert(part.to_owned());
          break;
        }
        current_article.categories.push(category.to_owned());
      }
      for set in sets.into_iter() {
        current_article.sets.push(set);
      }
      state = ParserState::Category;
    }
  }

  // Add the last article if it exists
  if current_article.id.is_some() {
    articles.push(current_article);
  }

  articles
}

#[derive(Default)]
pub struct ArxivMetadata {
  datum_map: BTreeMap<String, ArxivMetadatum>,
  datum_idx: BTreeMap<(u16, u8), BTreeSet<String>>,
}

impl ArxivMetadata {
  pub fn _load(&mut self) {
    let mut metadata = Vec::new();
    for e in WalkDir::new("data/arxiv/arxiv_metadata").into_iter() {
      let e = e.unwrap();
      let p = e.path();
      let p_s = p.as_os_str().to_str().unwrap();
      if p_s.ends_with(".xml") {
        //println!("{:?}", p);
        let mut file = File::open(p).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        let s = from_utf8(&buf).unwrap();
        let m = parse_arxiv_metadata(&s);
        for m in m.into_iter() {
          assert!(m.id.is_some());
          let id = m.id.clone().unwrap();
          if m.id_set.is_none() {
            println!("DEBUG: ArxivMetadata::_load: warning: id = {:?}", id);
          }
          let id_set = m.id_set.unwrap();
          match self.datum_idx.get_mut(&id_set) {
            None => {
              let mut ids = BTreeSet::new();
              ids.insert(id.clone());
              self.datum_idx.insert(id_set, ids);
            }
            Some(ids) => {
              ids.insert(id.clone());
            }
          }
          assert!(self.datum_map.insert(id, m.clone()).is_none());
          metadata.push(m);
        }
      }
    }
    println!("DEBUG: num metadata = {}", metadata.len());
    println!("DEBUG: metadatum[0] = {:?}", metadata[0]);
    println!("DEBUG: metadatum[1] = {:?}", metadata[1]);
    //println!("DEBUG: idx[0] = {:?}", self.datum_idx.iter().next().unwrap().0);
    for (idx, ids) in self.datum_idx.iter() {
      let mut n_cs = 0;
      let mut n_math = 0;
      for id in ids.iter() {
        match self.datum_map.get(id) {
          None => panic!("bug"),
          Some(m) => {
            for set in m.sets.iter() {
              if set == "cs" {
                n_cs += 1;
              } else if set == "math" {
                n_math += 1;
              }
            }
          }
        }
      }
      println!("DEBUG: idx = {:?} num ids = {} num cs = {} num math = {}", idx, ids.len(), n_cs, n_math);
    }
  }
}
