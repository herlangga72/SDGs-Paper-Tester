//! Paper parsing: YAML frontmatter (title/abstract/keywords) + body sections.

use crate::simd::lower_ascii;

pub const F_TITLE: u8 = 0;
pub const F_ABS: u8 = 1;
pub const F_KEY: u8 = 2;
pub const F_AUTHKEY: u8 = 3;
pub const F_ANY: u8 = 255; // unknown field -> check full text
pub const ALL_FIELDS: [u8; 4] = [F_TITLE, F_ABS, F_KEY, F_AUTHKEY];

pub enum YVal {
    S(String),
    L(Vec<String>),
}

fn scalar(v: &str) -> String {
    let t = v.trim();
    if t.len() >= 2 {
        let b = t.as_bytes();
        if (b[0] == b'"' && b[t.len() - 1] == b'"') || (b[0] == b'\'' && b[t.len() - 1] == b'\'') {
            return t[1..t.len() - 1].to_string();
        }
    }
    t.to_string()
}

/// Minimal YAML-subset parser: `key: value`, `[a, b]` lists, `- item` block
/// lists, and `|` block scalars.
fn parse_simple_yaml(lines: &[&str]) -> Vec<(String, YVal)> {
    let mut out = Vec::new();
    let (mut i, n) = (0usize, lines.len());
    while i < n {
        let line = lines[i].trim_end();
        let stripped = line.trim();
        if stripped.is_empty() || stripped.starts_with('#') {
            i += 1;
            continue;
        }
        if let Some(ci) = line.find(':') {
            let key = line[..ci].trim();
            let key_ok = !key.is_empty()
                && key.chars().next().map_or(false, |c| c.is_ascii_alphanumeric() || c == '_')
                && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
            if key_ok {
                let val = line[ci + 1..].trim();
                i += 1;
                if val.is_empty() || val == "|" || val == ">" || val == "|-" || val == ">-" {
                    let mut j = i;
                    while j < n && lines[j].trim().is_empty() {
                        j += 1;
                    }
                    if j < n && lines[j].trim_start().starts_with("- ") {
                        let mut items = Vec::new();
                        while j < n {
                            if let Some(rest) = lines[j].trim_start().strip_prefix("- ") {
                                items.push(scalar(rest));
                                j += 1;
                            } else {
                                break;
                            }
                        }
                        out.push((key.to_string(), YVal::L(items)));
                        i = j;
                    } else if j < n && (lines[j].starts_with(' ') || lines[j].starts_with('\t')) {
                        let mut bl = Vec::new();
                        while j < n
                            && (lines[j].starts_with(' ') || lines[j].starts_with('\t') || lines[j].trim().is_empty())
                        {
                            if !lines[j].trim().is_empty() {
                                bl.push(lines[j].trim().to_string());
                            }
                            j += 1;
                        }
                        out.push((key.to_string(), YVal::S(bl.join("\n"))));
                        i = j;
                    } else {
                        out.push((key.to_string(), YVal::S(String::new())));
                    }
                } else if val.starts_with('[') && val.ends_with(']') {
                    let inner = &val[1..val.len() - 1];
                    out.push((
                        key.to_string(),
                        YVal::L(inner.split(',').map(|x| scalar(x)).filter(|x| !x.is_empty()).collect()),
                    ));
                } else {
                    out.push((key.to_string(), YVal::S(scalar(val))));
                }
                continue;
            }
        }
        i += 1;
    }
    out
}

/// (meta, body) — YAML frontmatter between `---` lines, if present.
pub fn parse_frontmatter(text: &str) -> (Vec<(String, YVal)>, String) {
    if !text.trim_start().starts_with("---") {
        return (Vec::new(), text.to_string());
    }
    let lines: Vec<&str> = text.split('\n').collect();
    let mut end = None;
    for (idx, l) in lines.iter().enumerate().skip(1) {
        if l.trim() == "---" {
            end = Some(idx);
            break;
        }
    }
    let Some(end) = end else { return (Vec::new(), text.to_string()) };
    let meta = parse_simple_yaml(&lines[1..end]);
    let body = lines[end + 1..].join("\n");
    (meta, body)
}

/// Paper metadata (YAML frontmatter or the web form). Used by the web UI
/// to fill the report header and the sample/DOI JSON endpoints.
#[derive(Debug, Default, Clone)]
pub struct Meta {
    pub title: Option<String>,
    pub authors: Vec<String>,
    pub year: Option<String>,
    pub journal: Option<String>,
    pub doi: Option<String>,
    pub keywords: Vec<String>,
    pub abstract_text: Option<String>,
}

impl Meta {
    /// Build Meta from the raw frontmatter key/value pairs (keys are
    /// already lowercased by `parse_simple_yaml`).
    pub fn from_pairs(pairs: &[(String, YVal)]) -> Meta {
        let mut m = Meta::default();
        for (k, v) in pairs {
            match (k.as_str(), v) {
                ("title", YVal::S(s)) => m.title = Some(s.clone()),
                ("abstract" | "summary", YVal::S(s)) => m.abstract_text = Some(s.clone()),
                ("keywords" | "keyword" | "author_keywords", YVal::L(l)) => {
                    m.keywords = l.clone();
                }
                ("keywords" | "keyword" | "author_keywords", YVal::S(s)) => {
                    m.keywords = s
                        .split(',')
                        .map(|x| x.trim().to_string())
                        .filter(|x| !x.is_empty())
                        .collect();
                }
                ("authors" | "author" | "creators", YVal::L(l)) => m.authors = l.clone(),
                ("authors" | "author" | "creators", YVal::S(s)) => {
                    m.authors = s
                        .split(',')
                        .map(|x| x.trim().to_string())
                        .filter(|x| !x.is_empty())
                        .collect();
                }
                ("year", YVal::S(s)) => m.year = Some(s.clone()),
                ("journal", YVal::S(s)) => m.journal = Some(s.clone()),
                ("doi", YVal::S(s)) => m.doi = Some(s.clone()),
                _ => {}
            }
        }
        m
    }
}

pub struct Paper {
    pub title: Option<String>,
    full_text: String,
    lower_sections: [Option<Vec<u8>>; 4],
    lower_full: Vec<u8>,
}

impl Paper {
    pub fn from_text(text: &str) -> Paper {
        let (meta, body) = parse_frontmatter(text);
        let mut sections: [Option<String>; 4] = [None, None, None, None];

        // body marker lines (legacy TITLE:/ABSTRACT:/KEYWORDS: format)
        let mut cur: Option<usize> = None;
        for ln in body.lines() {
            let t = ln.trim_start();
            let mut marker: Option<usize> = None;
            if let Some(rest) = t.strip_prefix("TITLE:") {
                marker = Some(F_TITLE as usize);
                sections[F_TITLE as usize] = Some(rest.trim().to_string());
            } else if let Some(rest) = t.strip_prefix("ABSTRACT:") {
                marker = Some(F_ABS as usize);
                sections[F_ABS as usize] = Some(rest.trim().to_string());
            } else if let Some(rest) = t.strip_prefix("KEYWORDS:") {
                sections[F_KEY as usize] = Some(rest.trim().to_string());
                sections[F_AUTHKEY as usize] = Some(rest.trim().to_string());
                marker = Some(F_KEY as usize);
            } else if let Some(rest) = t.strip_prefix("AUTHKEY:") {
                sections[F_AUTHKEY as usize] = Some(rest.trim().to_string());
                marker = Some(F_AUTHKEY as usize);
            }
            if marker.is_some() {
                cur = marker;
            } else if let Some(c) = cur {
                if !ln.trim().is_empty() {
                    if let Some(s) = sections[c].as_mut() {
                        s.push(' ');
                        s.push_str(ln.trim());
                    }
                }
            }
        }

        // frontmatter wins over body markers
        let mut title = None;
        for (k, v) in meta {
            match (k.as_str(), v) {
                ("title", YVal::S(s)) => {
                    title = Some(s.clone());
                    sections[F_TITLE as usize] = Some(s);
                }
                ("abstract" | "summary", YVal::S(s)) => sections[F_ABS as usize] = Some(s),
                ("keywords" | "keyword" | "author_keywords", YVal::L(l)) => {
                    let j = l.join(", ");
                    sections[F_KEY as usize] = Some(j.clone());
                    sections[F_AUTHKEY as usize] = Some(j);
                }
                ("keywords" | "keyword" | "author_keywords", YVal::S(s)) => {
                    sections[F_KEY as usize] = Some(s.clone());
                    sections[F_AUTHKEY as usize] = Some(s);
                }
                _ => {}
            }
        }

        let has_sections = sections.iter().any(|s| s.as_ref().map_or(false, |x| !x.trim().is_empty()));
        let full = if !body.trim().is_empty() {
            body.to_string()
        } else if has_sections {
            sections
                .iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            text.to_string()
        };

        let lower_full = lower_ascii(full.as_bytes());
        let lower_sections = [
            sections[0].as_ref().map(|s| lower_ascii(s.as_bytes())),
            sections[1].as_ref().map(|s| lower_ascii(s.as_bytes())),
            sections[2].as_ref().map(|s| lower_ascii(s.as_bytes())),
            sections[3].as_ref().map(|s| lower_ascii(s.as_bytes())),
        ];
        Paper { title, lower_full, lower_sections, full_text: full }
    }

    /// Parse text and also return the frontmatter metadata (web server).
    pub fn from_text_with_meta(text: &str) -> (Paper, Meta) {
        let (pairs, _) = parse_frontmatter(text);
        let meta = Meta::from_pairs(&pairs);
        (Paper::from_text(text), meta)
    }

    /// Build a Paper straight from field sections (web form input, no YAML
    /// round-trip), with the same fallback semantics as `from_text`: fields
    /// that are missing fall back to the joined full text.
    pub fn from_sections(sections: [Option<String>; 4]) -> Paper {
        let full = sections.iter().flatten().cloned().collect::<Vec<_>>().join("\n");
        let lower_full = lower_ascii(full.as_bytes());
        let lower_sections = [
            sections[0].as_ref().map(|s| lower_ascii(s.as_bytes())),
            sections[1].as_ref().map(|s| lower_ascii(s.as_bytes())),
            sections[2].as_ref().map(|s| lower_ascii(s.as_bytes())),
            sections[3].as_ref().map(|s| lower_ascii(s.as_bytes())),
        ];
        let title = sections[F_TITLE as usize].clone();
        Paper { title, lower_full, lower_sections, full_text: full }
    }

    /// Original (un-lowercased) full text; byte offsets in it are 1:1 with
    /// `text_lower(F_ANY)` because ASCII lowercasing preserves length.
    pub fn full_text(&self) -> &str {
        &self.full_text
    }

    /// Lowercased text for a field id (falls back to full text).
    pub fn text_lower(&self, f: u8) -> &[u8] {
        if f <= F_AUTHKEY {
            if let Some(s) = &self.lower_sections[f as usize] {
                if !s.is_empty() {
                    return s;
                }
            }
        }
        &self.lower_full
    }
}
