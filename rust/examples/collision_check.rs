//! Regression check for the cross-query slot collision bug.
//! Two queries, each with one block containing a DISTINCT keyword, resolved
//! separately (as web.rs/main.rs do) but matched against a SHARED Memo.
//! With per-query slot reset both leaves get slot 0 and collide.
use sdg_tools::matcher::{self, Memo};
use sdg_tools::paper::Paper;
use sdg_tools::parser::Parser;
use sdg_tools::query::Query;
use sdg_tools::tokenizer::tokenize;

fn parse(text: &str) -> Vec<sdg_tools::ast::Node> {
    let root = Parser::new(tokenize(text).unwrap()).parse().unwrap();
    match &root {
        sdg_tools::ast::Node::Group { op, children } if op == "OR" => children.clone(),
        _ => vec![root],
    }
}

fn main() {
    // Distinct terms; q1's term is present in the paper, q2's is not.
    let mut q1 = Query { sdg: "01".into(), blocks: parse("TITLE(tax evasion)") };
    let mut q2 = Query { sdg: "02".into(), blocks: parse("TITLE(zzz nonexistent word)") };
    let table = matcher::compile_all(q1.blocks.iter().chain(q2.blocks.iter()));

    // Same as web.rs: resolve each query separately against ONE shared
    // counter (so slots never collide), then match with a SHARED Memo.
    let mut nslots = 0u32;
    matcher::resolve_blocks(&mut q1.blocks, &table, &mut nslots);
    matcher::resolve_blocks(&mut q2.blocks, &table, &mut nslots);

    let paper = Paper::from_text("tax evasion is a serious concern in many countries");
    let mut memo = Memo::new(&paper, 0);
    let r1 = matcher::scan_with_fields(&q1.blocks[0], &paper, &table, &mut memo);
    let r2 = matcher::scan_with_fields(&q2.blocks[0], &paper, &table, &mut memo);
    println!("q1 match={} hits={:?}", r1.3, r1.0.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>());
    println!("q2 match={} hits={:?}", r2.3, r2.0.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>());
    if !r1.3 {
        eprintln!("BUG: q1 (present term) missed");
        std::process::exit(1);
    }
    if r2.3 {
        eprintln!("BUG: q2 (nonexistent term) reported a match due to slot collision with q1");
        std::process::exit(1);
    }
    println!("OK: no cross-query collision");
}
