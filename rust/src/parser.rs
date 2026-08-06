//! Recursive-descent parser: tokens -> AST. Precedence: NOT > AND/W-n > OR.

use crate::ast::Node;
use crate::tokenizer::{ParseError, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Self { toks, pos: 0 }
    }

    pub fn parse(&mut self) -> Result<Node, ParseError> {
        let root = self.parse_or()?;
        if self.pos < self.toks.len() {
            return Err(ParseError::new(
                format!("unexpected trailing token {:?}", self.toks[self.pos]),
                self.pos,
            ));
        }
        Ok(root)
    }

    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Result<Token, ParseError> {
        if self.pos >= self.toks.len() {
            return Err(ParseError::new("unexpected end of query", self.pos));
        }
        let t = self.toks[self.pos].clone();
        self.pos += 1;
        Ok(t)
    }

    fn accept_op(&mut self, op: &str) -> bool {
        if let Some(Token::Op(o)) = self.peek() {
            if *o == op {
                self.pos += 1;
                return true;
            }
        }
        false
    }

    /// True when the parser sits at the end of a sub-expression: a closing
    /// paren or the end of the token stream. Used to tolerate dangling
    /// operators in real-world Elsevier data, e.g.
    /// `TITLE-ABS-KEY(mitigat*OR))` (keyword `mitigat*` glued to a stray
    /// `OR`), which Scopus queries contain in the wild.
    fn at_end(&self) -> bool {
        matches!(self.peek(), None | Some(Token::RParen))
    }

    fn parse_or(&mut self) -> Result<Node, ParseError> {
        let mut parts = vec![self.parse_and()?];
        while self.accept_op("OR") {
            // Dangling `OR` before ')' or EOF: drop the operator, the
            // previous term stands alone.
            if self.at_end() {
                break;
            }
            parts.push(self.parse_and()?);
        }
        if parts.len() == 1 {
            Ok(parts.pop().unwrap())
        } else {
            Ok(Node::Group { op: "OR".into(), children: parts })
        }
    }

    fn parse_and(&mut self) -> Result<Node, ParseError> {
        let mut terms = vec![self.parse_not()?];
        let mut ops: Vec<String> = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Token::Op(o)) if *o == "AND" => Some("AND".to_string()),
                Some(Token::Prox(p)) => Some(p.clone()),
                _ => None,
            };
            let Some(op) = op else { break };
            self.pos += 1;
            // Dangling `AND`/`W-n` before ')' or EOF: drop the operator.
            if self.at_end() {
                break;
            }
            ops.push(op);
            terms.push(self.parse_not()?);
        }
        // Right-nested: A AND B W/2 C  ->  AND(A, W/2(B, C))
        let mut node = terms.pop().unwrap();
        for k in (0..ops.len()).rev() {
            node = Node::Group { op: ops[k].clone(), children: vec![terms[k].clone(), node] };
        }
        Ok(node)
    }

    fn parse_not(&mut self) -> Result<Node, ParseError> {
        if self.accept_op("NOT") {
            return Ok(Node::Not { child: Box::new(self.parse_not()?) });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Node, ParseError> {
        let t = self.next()?;
        match t {
            Token::LParen => {
                let inner = self.parse_or()?;
                match self.next()? {
                    Token::RParen => Ok(inner),
                    other => Err(ParseError::new(format!("expected ')' got {other:?}"), self.pos)),
                }
            }
            Token::Field(name) => {
                match self.next()? {
                    Token::LParen => {}
                    other => {
                        return Err(ParseError::new(
                            format!("expected '(' after field {name} got {other:?}"),
                            self.pos,
                        ))
                    }
                }
                let inner = self.parse_or()?;
                match self.next()? {
                    Token::RParen => {}
                    other => {
                        return Err(ParseError::new(format!("expected ')' got {other:?}"), self.pos))
                    }
                }
                let fields = name.split('-').map(|s| s.to_string()).collect();
                Ok(Node::Field { fields, child: Box::new(inner) })
            }
            Token::Str { value, exact } => {
                // Merge adjacent plain (unquoted) terms into a phrase.
                // Real Elsevier data contains unquoted multi-word terms
                // such as `TITLE-ABS-KEY(pes scheme*)` (SDG02),
                // `TITLE-ABS(ethylene terephthalate)` (SDG12) and
                // `TITLE-ABS(neogobius melanostomus)` (SDG15); quoted and
                // braced terms keep their boundaries.
                let mut kw = value;
                let ex = exact;
                while let Some(Token::Str { value: v, exact: e }) = self.peek().cloned() {
                    if ex || e {
                        break;
                    }
                    self.pos += 1;
                    kw.push(' ');
                    kw.push_str(&v);
                }
                Ok(Node::Leaf { keyword: kw, exact: ex })
            }
            other => Err(ParseError::new(format!("unexpected token {other:?}"), self.pos)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn parse(src: &str) -> Result<Node, ParseError> {
        Parser::new(tokenize(src)?).parse()
    }

    fn leaves(n: &Node) -> Vec<&str> {
        let mut out = Vec::new();
        fn rec<'a>(n: &'a Node, out: &mut Vec<&'a str>) {
            match n {
                Node::Leaf { keyword, .. } => out.push(keyword),
                Node::Field { child, .. } | Node::Not { child } => rec(child, out),
                Node::Group { children, .. } => children.iter().for_each(|c| rec(c, out)),
            }
        }
        rec(n, &mut out);
        out
    }

    // -- basic grammar --------------------------------------------------

    #[test]
    fn simple_phrase() {
        let n = parse("\"tax evasion\"").unwrap();
        assert!(matches!(n, Node::Leaf { keyword, exact: false } if keyword == "tax evasion"));
    }

    #[test]
    fn or_group_flattens() {
        let n = parse("a OR b OR c").unwrap();
        match n {
            Node::Group { op, children } => {
                assert_eq!(op, "OR");
                assert_eq!(children.len(), 3);
            }
            other => panic!("expected OR group, got {other:?}"),
        }
    }

    #[test]
    fn precedence_not_over_and_over_or() {
        // a OR b AND NOT c  ->  OR(a, AND(b, NOT(c)))
        let n = parse("a OR b AND NOT c").unwrap();
        match n {
            Node::Group { op, children } if op == "OR" && children.len() == 2 => {
                match &children[1] {
                    Node::Group { op, children } if op == "AND" && children.len() == 2 => {
                        match &children[1] {
                            Node::Not { child } => {
                                assert!(matches!(&**child, Node::Leaf { keyword, .. } if keyword == "c"))
                            }
                            other => panic!("expected NOT, got {other:?}"),
                        }
                    }
                    other => panic!("expected AND, got {other:?}"),
                }
            }
            other => panic!("unexpected tree {other:?}"),
        }
    }

    #[test]
    fn and_prox_right_nested() {
        // A AND B W/2 C  ->  AND(A, W/2(B, C))
        let n = parse("A AND B W/2 C").unwrap();
        match n {
            Node::Group { op, children } if op == "AND" && children.len() == 2 => {
                match &children[1] {
                    Node::Group { op, children } if op == "W/2" && children.len() == 2 => {
                        assert_eq!(leaves(&children[0])[0], "B");
                        assert_eq!(leaves(&children[1])[0], "C");
                    }
                    other => panic!("expected W/2 group, got {other:?}"),
                }
            }
            other => panic!("unexpected tree {other:?}"),
        }
    }

    #[test]
    fn field_wrap_and_exact_brace() {
        let n = parse("TITLE-ABS-KEY(\"coral reef\" OR {exact term})").unwrap();
        match n {
            Node::Field { fields, child } => {
                assert_eq!(fields, vec!["TITLE", "ABS", "KEY"]);
                match &*child {
                    Node::Group { op, children } if op == "OR" && children.len() == 2 => {
                        assert!(matches!(&children[0], Node::Leaf { keyword, exact: false } if keyword == "coral reef"));
                        assert!(matches!(&children[1], Node::Leaf { keyword, exact: true } if keyword == "exact term"));
                    }
                    other => panic!("expected OR, got {other:?}"),
                }
            }
            other => panic!("expected field wrap, got {other:?}"),
        }
    }

    #[test]
    fn wildcard_and_proximity_tokens() {
        let n = parse("cereal* W/4 girl*").unwrap();
        assert!(matches!(&n, Node::Group { op, .. } if op == "W/4"));
        assert_eq!(leaves(&n), vec!["cereal*", "girl*"]);
        // leading wildcard
        let n = parse("*divers* OR ?term*").unwrap();
        assert_eq!(leaves(&n), vec!["*divers*", "?term*"]);
    }

    #[test]
    fn prox_operators_all_forms() {
        for op in ["W/2", "PRE/3", "POST/4", "NEAR/5", "ONEAR/6"] {
            let n = parse(&format!("a {op} b")).unwrap();
            assert!(matches!(&n, Node::Group { op: o, .. } if o == op), "{op}");
        }
    }

    // -- leniency for real-world Elsevier data --------------------------

    #[test]
    fn dangling_or_before_rparen_dropped() {
        // SDG02.txt: `TITLE-ABS-KEY(mitigat*OR))` — keyword glued to a
        // stray `OR` right before the closing paren.
        let n = parse("TITLE-ABS-KEY(rehab*) OR TITLE-ABS-KEY(adapt*) OR TITLE-ABS-KEY(mitigat*OR)").unwrap();
        assert_eq!(leaves(&n), vec!["rehab*", "adapt*", "mitigat*"]);
        assert!(matches!(&n, Node::Group { op, .. } if op == "OR"));
    }

    #[test]
    fn dangling_or_at_group_end() {
        let n = parse("(a OR b OR)").unwrap();
        assert_eq!(leaves(&n), vec!["a", "b"]);
        let n = parse("(a OR)").unwrap();
        assert_eq!(leaves(&n), vec!["a"]);
        let n = parse("a OR").unwrap(); // EOF
        assert_eq!(leaves(&n), vec!["a"]);
    }

    #[test]
    fn dangling_and_and_prox_dropped() {
        let n = parse("(a AND)").unwrap();
        assert_eq!(leaves(&n), vec!["a"]);
        let n = parse("(a W/2)").unwrap();
        assert_eq!(leaves(&n), vec!["a"]);
        let n = parse("a AND b AND").unwrap();
        assert_eq!(leaves(&n), vec!["a", "b"]);
    }

    #[test]
    fn unquoted_phrase_merged() {
        // Real Elsevier data: unquoted multi-word terms inside field wraps.
        for src in [
            "TITLE-ABS-KEY(pes scheme*)",
            "TITLE-ABS-KEY(pes program*)",
            "TITLE-ABS-KEY(Shahid Abbaspour)",
            "TITLE-ABS(ethylene terephthalate)",
            "TITLE-ABS(neogobius melanostomus)",
        ] {
            let n = parse(src).unwrap();
            match &n {
                Node::Field { child, .. } => match &**child {
                    Node::Leaf { keyword, exact } => {
                        let inner = src.split('(').nth(1).unwrap().trim_end_matches(')');
                        assert_eq!(keyword, inner, "{src}");
                        assert!(!exact, "{src}");
                    }
                    other => panic!("{src}: expected leaf, got {other:?}"),
                },
                other => panic!("{src}: expected field, got {other:?}"),
            }
        }
        // three-word phrase, and merge across a trailing dangling OR
        let n = parse("(a b c OR)").unwrap();
        assert_eq!(leaves(&n), vec!["a b c"]);
        // quoted/braced terms keep their boundaries
        let n = parse("(\"a b\" OR {c d})").unwrap();
        assert_eq!(leaves(&n), vec!["a b", "c d"]);
        assert!(parse("(\"a b\" {c d})").is_err()); // juxtaposed quoted+braced
    }

    #[test]
    fn trailing_dot_attached() {
        // SDG07.txt: `TITLE-ABS(articles.)`
        let n = parse("TITLE-ABS(articles.)").unwrap();
        assert_eq!(leaves(&n), vec!["articles."]);
        let n = parse("cereal*.").unwrap();
        assert_eq!(leaves(&n), vec!["cereal*."]);
    }

    // -- errors ---------------------------------------------------------

    #[test]
    fn errors_on_bad_input() {
        assert!(parse("unterminated \"quote").is_err());
        assert!(parse("unterminated {brace").is_err());
        assert!(parse("a)").is_err()); // stray close paren at top level
        assert!(parse("w/0").is_err()); // zero-width proximity
        assert!(parse("a W/2)").is_err()); // stray close paren at top level
    }
}

