//! Load all SDG<number>.txt query files in a directory.

use crate::ast::Node;
use crate::parser::Parser;
use crate::tokenizer::tokenize;
use std::fs;
use std::path::Path;

pub struct Query {
    pub sdg: String, // digits as written in the file name, e.g. "01", "17"
    pub blocks: Vec<Node>,
}

fn sdg_number(name: &str) -> Option<String> {
    let up = name.to_ascii_uppercase();
    let idx = up.find("SDG")?;
    let rest = &up[idx + 3..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        Some(digits)
    }
}

pub fn load_queries(dir: &Path) -> Result<Vec<Query>, String> {
    let mut files: Vec<_> = fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| p.file_name().map_or(false, |n| n.to_string_lossy().ends_with(".txt")))
        .collect();
    files.sort(); // lexicographic, like Python's sorted(glob)

    let mut out: Vec<Query> = Vec::new();
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        let Some(sdg) = sdg_number(&name) else { continue };
        let text = fs::read_to_string(&f)
            .map_err(|e| format!("cannot read {}: {e}", f.display()))?;
        let root = Parser::new(tokenize(&text).map_err(|e| format!("{name}: {e}"))?)
            .parse()
            .map_err(|e| format!("{name}: {e}"))?;
        // top-level OR children are the "blocks"
        let blocks = match &root {
            Node::Group { op, children } if op == "OR" => children.clone(),
            _ => vec![root],
        };
        out.push(Query { sdg, blocks });
    }
    out.sort_by(|a, b| {
        let an = a.sdg.parse::<u32>().unwrap_or(0);
        let bn = b.sdg.parse::<u32>().unwrap_or(0);
        an.cmp(&bn)
    });
    Ok(out)
}
