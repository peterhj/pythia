pub struct MarkdownCodeBlock {
  pub language: String,
  pub content:  String,
}

pub struct MarkdownCodeBlocks {
  pub blocks: Vec<MarkdownCodeBlock>,
  pub err: Option<()>,
}

impl MarkdownCodeBlocks {
  pub fn parse(input: &str) -> MarkdownCodeBlocks {
    let mut lines = input.lines();
    let mut blocks = Vec::new();
    let mut block_level = 0;
    let mut current_lang = String::new();
    let mut current_block = String::new();
    while let Some(line) = lines.next() {
      if line.starts_with("```") {
        let lang = line[3 .. ].trim().to_string();
        let empty_lang = lang.is_empty();
        if block_level <= 0 {
          if empty_lang {
            //println!("DEBUG: num blocks  = {}", blocks.len());
            //println!("DEBUG: blocks      = {blocks:?}");
            //println!("DEBUG: block level = {block_level}");
            return MarkdownCodeBlocks{blocks, err: Some(())};
          }
          block_level = 1;
          current_lang = lang.clone();
          current_block.clear();
        } else if !empty_lang {
          block_level += 1;
          current_block.push_str(line);
          current_block.push('\n');
        } else {
          block_level -= 1;
          if block_level > 0 {
            current_block.push_str(line);
            current_block.push('\n');
          } else {
            blocks.push(MarkdownCodeBlock{
              language: current_lang,
              content:  current_block
            });
            current_lang = String::new();
            current_block = String::new();
          }
        }
      } else if block_level > 0 {
        current_block.push_str(line);
        current_block.push('\n');
      }
    }
    if block_level > 0 {
      // TODO: maybe err/warn here.
      blocks.push(MarkdownCodeBlock{
        language: current_lang,
        content:  current_block
      });
    }
    MarkdownCodeBlocks{blocks, err: None}
  }
}
