//! AST for Scopus boolean queries.

#[derive(Debug, Clone)]
pub enum Node {
    /// A single search term: `"tax evasion"`, `cereal*`, `{BEPS}`.
    Leaf { keyword: String, exact: bool },
    /// Field prefix wrapping a sub-expression: `TITLE-ABS-KEY(...)`.
    Field { fields: Vec<String>, child: Box<Node> },
    /// `OR`, `AND` or a proximity operator (`W/4`, ...).
    Group { op: String, children: Vec<Node> },
    /// `NOT expr` — everything under it is an exclusion.
    Not { child: Box<Node> },
}
