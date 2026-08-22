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
/// The body is a slice of `text` (no join/copy), so an owned input string
/// can be kept whole by the Paper and the searchable body sliced out of it.
pub fn parse_frontmatter(text: &str) -> (Vec<(String, YVal)>, &str) {
    if !text.trim_start().starts_with("---") {
        return (Vec::new(), text);
    }
    let lines: Vec<&str> = text.split('\n').collect();
    let mut end = None;
    let mut end_byte = text.len();
    for (idx, l) in lines.iter().enumerate().skip(1) {
        if l.trim() == "---" {
            end = Some(idx);
            // byte offset just past the closing `---` line: sum of the line
            // lengths plus one '\n' per line (clamped; a trailing `---`
            // without '\n' overshoots by one).
            end_byte = lines[..=idx].iter().map(|s| s.len() + 1).sum::<usize>().min(text.len());
            break;
        }
    }
    let Some(end) = end else { return (Vec::new(), text) };
    let meta = parse_simple_yaml(&lines[1..end]);
    (meta, &text[end_byte..])
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
    /// Owned input text. `full_text()` slices `[body_start..]`, so the
    /// searchable body is never copied out of the caller's string.
    input: String,
    /// Byte offset of the searchable body within `input` (frontmatter is
    /// dropped by slicing, not copying). 0 when the full text is the
    /// sections join or the whole input.
    body_start: usize,
    lower_sections: [Option<Vec<u8>>; 4],
    lower_full: Vec<u8>,
    /// True when `lower_full` is the join of the section buffers (no body
    /// text), so every field search can use the full text alone.
    pub(crate) full_covers_sections: bool,
}


/// Shared construction from an owned input, parsed frontmatter pairs and
/// the body's byte offset within it. Zero-copy: the body is sliced out of
/// `text`; only the frontmatter/section strings (small) are copied.
fn build_paper(text: String, meta: Vec<(String, YVal)>, body_start: usize) -> Paper {
    let body_is_ws = text[body_start..].trim().is_empty();
    let mut sections: [Option<String>; 4] = [None, None, None, None];

    // body marker lines (legacy TITLE:/ABSTRACT:/KEYWORDS: format)
    let mut cur: Option<usize> = None;
    for ln in text[body_start..].lines() {
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
    let (input, body_start) = if !body_is_ws {
        (text, body_start)
    } else if has_sections {
        (
            sections
                .iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
            0,
        )
    } else {
        (text, 0)
    };

    let lower_full = lower_ascii(&input.as_bytes()[body_start..]);
    let lower_sections = [
        sections[0].as_ref().map(|s| lower_ascii(s.as_bytes())),
        sections[1].as_ref().map(|s| lower_ascii(s.as_bytes())),
        sections[2].as_ref().map(|s| lower_ascii(s.as_bytes())),
        sections[3].as_ref().map(|s| lower_ascii(s.as_bytes())),
    ];
    Paper {
        title,
        input,
        body_start,
        lower_full,
        lower_sections,
        full_covers_sections: body_is_ws,
    }
}

impl Paper {
    /// Parse an owned text string. The string is kept as-is (body is a
    /// slice), so no full-text copy happens on the hot path. Callers that
    /// already own a `String` should use this; `from_text` (borrowed) is
    /// for literals and tests.
    pub fn from_owned(text: String) -> Paper {
        let (meta, body) = parse_frontmatter(&text);
        let body_start = text.len() - body.len();
        build_paper(text, meta, body_start)
    }

    pub fn from_text(text: &str) -> Paper {
        Paper::from_owned(text.to_string())
    }


    /// Parse text and also return the frontmatter metadata (web server).
    /// The frontmatter is parsed once for both the `Meta` and the sections.
    pub fn from_owned_with_meta(text: String) -> (Paper, Meta) {
        let (pairs, body) = parse_frontmatter(&text);
        let meta = Meta::from_pairs(&pairs);
        let body_start = text.len() - body.len();
        (build_paper(text, pairs, body_start), meta)
    }

    pub fn from_text_with_meta(text: &str) -> (Paper, Meta) {
        Paper::from_owned_with_meta(text.to_string())
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
        Paper { title, input: full, body_start: 0, lower_full, lower_sections, full_covers_sections: true }
    }

    /// Byte ranges of each present section within `text_lower(F_ANY)`.
    /// Valid only when `full_covers_sections` (full = sections joined with
    /// '\n'); the ranges are the join positions, so per-segment glob
    /// matching keeps per-field semantics.
    pub(crate) fn full_section_ranges(&self) -> [(usize, usize); 4] {
        let mut out = [(0usize, 0usize); 4];
        let mut pos = 0usize;
        let mut first = true;
        for (f, s) in self.lower_sections.iter().enumerate() {
            if let Some(sec) = s {
                if !first {
                    pos += 1; // the join separator byte (newline)
                }
                out[f] = (pos, pos + sec.len());
                pos += sec.len();
                first = false;
            }
        }
        out
    }

    /// Original (un-lowercased) full text; byte offsets in it are 1:1 with
    /// `text_lower(F_ANY)` because ASCII lowercasing preserves length.
    pub fn full_text(&self) -> &str {
        &self.input[self.body_start..]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_paper_keeps_title_and_sections() {
        let p = Paper::from_owned(
            "---\ntitle: \"T\"\nabstract: |\n  A b.\nkeywords: [x, y]\n---\nBody text here.".to_string(),
        );
        assert_eq!(p.title.as_deref(), Some("T"));
        assert_eq!(p.full_text(), "Body text here.");
        assert_eq!(p.text_lower(F_TITLE), b"t");
        assert_eq!(p.text_lower(F_ABS), b"a b.");
        assert_eq!(p.text_lower(F_KEY), b"x, y");
        assert_eq!(p.text_lower(F_AUTHKEY), b"x, y"); // keywords feed both
    }

    #[test]
    fn owned_paper_no_frontmatter() {
        let p = Paper::from_owned("just a plain paper body".to_string());
        assert_eq!(p.title, None);
        assert_eq!(p.full_text(), "just a plain paper body");
        assert_eq!(p.text_lower(F_ANY), b"just a plain paper body");
        // body text present -> full text is the body, not the sections join
        assert!(!p.full_covers_sections);
    }
}
