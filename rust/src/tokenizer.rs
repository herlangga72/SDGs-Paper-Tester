//! Tokenizer: text -> tokens. Handles unquoted keywords (`TITLE-ABS(H2)`),
//! attached wildcards (`cereal*`), proximity operators (`W/4`), quoted phrases
//! and `{...}` exact terms. Quoted/braced content is scanned with SIMD.

use crate::simd::{next_special, skip_ws};
use std::fmt;

#[derive(Debug, Clone)]
pub enum Token {
    LParen,
    RParen,
    Op(&'static str), // "OR" | "AND" | "NOT"
    Prox(String),     // "W/4", "PRE/2", ...
    Field(String),    // "TITLE-ABS-KEY", "AUTHKEY", ...
    Str { value: String, exact: bool },
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub pos: usize,
}

impl ParseError {
    pub fn new(message: impl Into<String>, pos: usize) -> Self {
        Self { message: message.into(), pos }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at position {}", self.message, self.pos)
    }
}

impl std::error::Error for ParseError {}

#[inline]
fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'-'
}

#[inline]
fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn scan_quoted(b: &[u8], start: usize) -> Result<(String, usize), ParseError> {
    // b[start] == '"'
    let mut out: Vec<u8> = Vec::new();
    let n = b.len();
    let mut cur = start + 1;
    loop {
        match next_special(b, cur, b"\"\\") {
            Some(p) => {
                out.extend_from_slice(&b[cur..p]);
                if b[p] == b'"' {
                    return Ok((String::from_utf8_lossy(&out).into_owned(), p + 1));
                }
                // escaped char: keep next byte, skip the backslash
                if p + 1 >= n {
                    return Err(ParseError::new("unterminated quoted string", start));
                }
                out.push(b[p + 1]);
                cur = p + 2;
            }
            None => return Err(ParseError::new("unterminated quoted string", start)),
        }
    }
}

fn scan_braced(b: &[u8], start: usize) -> Result<(String, usize), ParseError> {
    // b[start] == '{', supports nested braces
    let mut out: Vec<u8> = Vec::new();
    let mut cur = start + 1;
    let mut depth = 1usize;
    loop {
        match next_special(b, cur, b"{}") {
            Some(p) => {
                out.extend_from_slice(&b[cur..p]);
                if b[p] == b'{' {
                    depth += 1;
                    out.push(b'{');
                } else {
                    depth -= 1;
                    if depth == 0 {
                        return Ok((String::from_utf8_lossy(&out).into_owned(), p + 1));
                    }
                    out.push(b'}');
                }
                cur = p + 1;
            }
            None => return Err(ParseError::new("unterminated {...} term", start)),
        }
    }
}

pub fn tokenize(text: &str) -> Result<Vec<Token>, ParseError> {
    let b = text.as_bytes();
    let n = b.len();
    let mut toks: Vec<Token> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let c = b[i];
        if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
            i = skip_ws(b, i);
        } else if c == b'(' {
            toks.push(Token::LParen);
            i += 1;
        } else if c == b')' {
            toks.push(Token::RParen);
            i += 1;
        } else if c == b'"' {
            let (val, end) = scan_quoted(b, i)?;
            toks.push(Token::Str { value: val, exact: false });
            i = end;
        } else if c == b'{' {
            let (val, end) = scan_braced(b, i)?;
            toks.push(Token::Str { value: val, exact: true });
            i = end;
        } else if is_ident(c) {
            let start = i;
            while i < n && is_ident(b[i]) {
                i += 1;
            }
            let name = &text[start..i];
            let up = name.to_ascii_uppercase();
            let next = b.get(i).copied();
            // operator words need a word boundary after them (Python `\b` semantics)
            let boundary_ok = match next {
                Some(x) => !is_word(x),
                None => true,
            };
            if boundary_ok && (up == "OR" || up == "AND" || up == "NOT") {
                toks.push(Token::Op(match up.as_str() {
                    "OR" => "OR",
                    "AND" => "AND",
                    _ => "NOT",
                }));
            } else if boundary_ok
                && matches!(up.as_str(), "W" | "PRE" | "POST" | "NEAR" | "ONEAR")
                && next == Some(b'/')
            {
                let mut j = i + 1;
                let mut num = 0usize;
                while j < n && b[j].is_ascii_digit() {
                    num = num * 10 + (b[j] - b'0') as usize;
                    j += 1;
                }
                if num == 0 {
                    return Err(ParseError::new(format!("invalid proximity operator {up}/"), start));
                }
                toks.push(Token::Prox(format!("{up}/{num}")));
                i = j;
            } else {
                // field name if followed by '(' (skipping whitespace)
                let j = skip_ws(b, i);
                if j < n && b[j] == b'(' {
                    toks.push(Token::Field(up));
                    i = j; // leave '(' for the next iteration
                } else {
                    // bare (unquoted) keyword; consume attached wildcards: cereal*
                    // (a trailing '.' also sticks to the word, e.g. `articles.` in SDG07)
                    let mut kw = name.to_string();
                    while i < n && (b[i] == b'*' || b[i] == b'?' || b[i] == b'.') {
                        kw.push(b[i] as char);
                        i += 1;
                    }
                    toks.push(Token::Str { value: kw, exact: false });
                }
            }
        } else if c == b'*' || c == b'?' {
            // leading wildcards: *divers*, ?term*
            let mut j = i;
            while j < n && (b[j] == b'*' || b[j] == b'?') {
                j += 1;
            }
            if j < n && is_ident(b[j]) {
                let mut kw = String::from_utf8_lossy(&b[i..j]).into_owned();
                let mut k = j;
                while k < n && is_ident(b[k]) {
                    k += 1;
                }
                kw.push_str(&text[j..k]);
                while k < n && (b[k] == b'*' || b[k] == b'?') {
                    kw.push(b[k] as char);
                    k += 1;
                }
                toks.push(Token::Str { value: kw, exact: false });
                i = k;
            } else {
                return Err(ParseError::new("unexpected wildcard", i));
            }
        } else {
            return Err(ParseError::new(format!("unexpected character {:?}", c as char), i));
        }
    }
    Ok(toks)
}
