//! sdg_tools — parse Scopus SDG query files into a keyword AST, and check
//! papers against them. SIMD (AVX2) with scalar fallback.
//!
//! This library backs two binaries:
//!   - `sdg_tools` (src/main.rs): CLI for parse / match
//!   - `web` (src/bin/web.rs): the HTTP server (was web/app.py)

pub mod ac;
pub mod ast;
pub mod cpu;
pub mod cache;
pub mod matcher;
pub mod paper;
pub mod parser;
pub mod query;
pub mod simd;
pub mod tokenizer;
