//! Boot cache, v3: a single file that is mmap'd at boot and viewed
//! zero-copy. Layout (all little-endian, every section 4-byte aligned):
//!
//!   magic "SDGC" | u32 version | u32 n_mtime | (u32 name_len+pad, u64 mtime)*
//!   u32 blob_len | blob bytes (padded to 4)          <- the string blob
//!   u32 n_patterns | Pattern records (28 B each)     <- viewed as &[Pattern]
//!   u32 n_sdg | per SDG: u32 n_blocks | per block:
//!       u32 n_prog | Op records (8 B) | u32 n_leaf | LeafDesc records (12 B)
//!   dicts: u32 n_dicts | SdgDict::serialize records (owned rebuild)
//!   queries: u32 n_queries | per query: u32 sdg_len + sdg, u32 n_blocks, nodes

use crate::ast::Node;
use crate::matcher::{set_blob, FlatBlock, LeafDesc, Op, Pattern, SdgDict};
use crate::query::Query;
use memmap2::MmapOptions;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const MAGIC: &[u8; 4] = b"SDGC";
const VERSION: u32 = 5;

pub struct CacheData {
    /// Parsed + resolved query ASTs (pid/mask/slot already stamped).
    pub queries: Vec<Query>,
    pub patterns: &'static [Pattern],
    pub flats: Vec<Vec<FlatBlock>>,
    pub dicts: Vec<SdgDict>,
}

pub fn cache_path(dir: &Path) -> PathBuf {
    dir.join("sdg_cache.bin")
}

fn query_mtimes(dir: &Path) -> Option<Vec<(String, u64)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).ok()? {
        let p = entry.ok()?.path();
        if p.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        let meta = fs::metadata(&p).ok()?;
        let mtime = meta.modified().ok()?.duration_since(SystemTime::UNIX_EPOCH).ok()?.as_secs();
        let name = p.file_name()?.to_string_lossy().into_owned();
        out.push((name, mtime));
    }
    out.sort();
    Some(out)
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut Cursor<&[u8]>) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

/// Write a string padded to 4 bytes (keeps every section 4-aligned so the
/// record slices can be viewed as aligned structs).
fn write_str_pad<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())?;
    let pad = (4 - (s.len() % 4)) % 4;
    for _ in 0..pad {
        w.write_all(&[0u8])?;
    }
    Ok(())
}

fn read_str_pad(r: &mut Cursor<&[u8]>) -> Option<String> {
    let n = read_u32(r)? as usize;
    if n > (1 << 20) {
        return None;
    }
    let mut v = vec![0u8; n + ((4 - (n % 4)) % 4)];
    r.read_exact(&mut v).ok()?;
    Some(String::from_utf8(v[..n].to_vec()).ok()?)
}

fn write_node<W: Write>(w: &mut W, n: &Node) -> std::io::Result<()> {
    match n {
        Node::Leaf { keyword, exact, pid, mask, slot } => {
            w.write_all(&[0u8])?;
            write_str_pad(w, keyword)?;
            w.write_all(&[*exact as u8])?;
            write_u32(w, *pid)?;
            w.write_all(&[*mask])?;
            write_u32(w, *slot)?;
        }
        Node::Field { fields, child } => {
            w.write_all(&[1u8])?;
            write_u32(w, fields.len() as u32)?;
            for f in fields {
                write_str_pad(w, f)?;
            }
            write_node(w, child)?;
        }
        Node::Group { op, children } => {
            w.write_all(&[2u8])?;
            write_str_pad(w, op)?;
            write_u32(w, children.len() as u32)?;
            for c in children {
                write_node(w, c)?;
            }
        }
        Node::Not { child } => {
            w.write_all(&[3u8])?;
            write_node(w, child)?;
        }
    }
    Ok(())
}

fn read_node(r: &mut Cursor<&[u8]>) -> Option<Node> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag).ok()?;
    Some(match tag[0] {
        0 => Node::Leaf {
            keyword: read_str_pad(r)?,
            exact: {
                let mut b = [0u8; 1];
                r.read_exact(&mut b).ok()?;
                b[0] != 0
            },
            pid: read_u32(r)?,
            mask: {
                let mut b = [0u8; 1];
                r.read_exact(&mut b).ok()?;
                b[0]
            },
            slot: read_u32(r)?,
        },
        1 => {
            let n = read_u32(r)? as usize;
            let mut fields = Vec::with_capacity(n.min(16));
            for _ in 0..n {
                fields.push(read_str_pad(r)?);
            }
            Node::Field { fields, child: Box::new(read_node(r)?) }
        }
        2 => {
            let op = read_str_pad(r)?;
            let n = read_u32(r)? as usize;
            if n > 10_000_000 {
                return None;
            }
            let mut children = Vec::with_capacity(n.min(1024));
            for _ in 0..n {
                children.push(read_node(r)?);
            }
            Node::Group { op, children }
        }
        3 => Node::Not { child: Box::new(read_node(r)?) },
        _ => return None,
    })
}

/// Write the cache. `blob` is the process string blob (persisted verbatim).
pub fn write_cache(
    dir: &Path,
    blob: &[u8],
    queries: &[Query],
    patterns: &[Pattern],
    flats: &[Vec<FlatBlock>],
    dicts: &[SdgDict],
) -> std::io::Result<()> {
    let mtimes = query_mtimes(dir).unwrap_or_default();
    let path = cache_path(dir);
    let mut w = Vec::with_capacity(blob.len() + 4 * 1024 * 1024);
    w.write_all(MAGIC)?;
    write_u32(&mut w, VERSION)?;
    write_u32(&mut w, mtimes.len() as u32)?;
    for (name, mt) in &mtimes {
        write_str_pad(&mut w, name)?;
        w.write_all(&mt.to_le_bytes())?;
    }
    // blob (padded to 4)
    write_u32(&mut w, blob.len() as u32)?;
    w.write_all(blob)?;
    for _ in 0..((4 - (blob.len() % 4)) % 4) {
        w.write_all(&[0u8])?;
    }
    // patterns (28-byte records)
    write_u32(&mut w, patterns.len() as u32)?;
    for p in patterns {
        p.serialize(&mut w)?;
    }
    // flats
    write_u32(&mut w, flats.len() as u32)?;
    for sdg in flats {
        write_u32(&mut w, sdg.len() as u32)?;
        for fb in sdg {
            fb.serialize(&mut w)?;
        }
    }
    // dicts
    write_u32(&mut w, dicts.len() as u32)?;
    for d in dicts {
        d.serialize(&mut w)?;
    }
    // queries (AST)
    write_u32(&mut w, queries.len() as u32)?;
    for q in queries {
        write_str_pad(&mut w, &q.sdg)?;
        write_u32(&mut w, q.blocks.len() as u32)?;
        for b in &q.blocks {
            write_node(&mut w, b)?;
        }
    }
    fs::write(&path, w)?;
    Ok(())
}

/// Read and validate the cache; mmap it and view patterns/flats zero-copy.
pub fn read_cached(dir: &Path) -> Option<CacheData> {
    let path = cache_path(dir);
    let mtimes = query_mtimes(dir)?;
    let file = fs::File::open(&path).ok()?;
    let mmap: &'static memmap2::Mmap = Box::leak(Box::new(unsafe { MmapOptions::new().map(&file).ok()? }));
    let base: &'static [u8] = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
    let mut r = Cursor::new(base);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if &magic != MAGIC || read_u32(&mut r)? != VERSION {
        return None;
    }
    let n = read_u32(&mut r)? as usize;
    if n > 128 {
        return None;
    }
    for _ in 0..n {
        let name = read_str_pad(&mut r)?;
        let mut mt = [0u8; 8];
        r.read_exact(&mut mt).ok()?;
        let mtime = u64::from_le_bytes(mt);
        match mtimes.iter().find(|(nm, _)| *nm == name) {
            Some((_, m)) if *m == mtime => {}
            _ => return None,
        }
    }
    // blob
    let blob_len = read_u32(&mut r)? as usize;
    let blob_start = r.position() as usize;
    set_blob(unsafe { std::slice::from_raw_parts(base.as_ptr().add(blob_start), blob_len) });
    r.set_position((blob_start + ((blob_len + 3) / 4) * 4) as u64);
    // patterns (zero-copy)
    let np = read_u32(&mut r)? as usize;
    let pat_start = r.position() as usize;
    let patterns: &'static [Pattern] = unsafe {
        std::slice::from_raw_parts(base.as_ptr().add(pat_start) as *const Pattern, np)
    };
    r.set_position((pat_start + np * 28) as u64);
    // flats (zero-copy per block)
    let nsdg = read_u32(&mut r)? as usize;
    let mut flats: Vec<Vec<FlatBlock>> = Vec::with_capacity(nsdg);
    for _ in 0..nsdg {
        let nb = read_u32(&mut r)? as usize;
        let mut group = Vec::with_capacity(nb);
        for _ in 0..nb {
            let np2 = read_u32(&mut r)? as usize;
            let prog_start = r.position() as usize;
            let prog: &'static [Op] =
                unsafe { std::slice::from_raw_parts(base.as_ptr().add(prog_start) as *const Op, np2) };
            r.set_position((prog_start + np2 * 8) as u64);
            let nl = read_u32(&mut r)? as usize;
            let leaf_start = r.position() as usize;
            // LeafDesc records: pid u32, slot u32, mask u8, excluded u8,
            // 2 pad bytes = 12 B. Alignment is 4 (u32 fields) and the
            // section starts 4-aligned, so the slice view is valid.
            let leaves: &'static [LeafDesc] =
                unsafe { std::slice::from_raw_parts(base.as_ptr().add(leaf_start) as *const LeafDesc, nl) };
            r.set_position((leaf_start + nl * 12) as u64);
            group.push(FlatBlock { prog, leaves });
        }
        flats.push(group);
    }
    // dicts (owned rebuild)
    let nd = read_u32(&mut r)? as usize;
    let mut dicts = Vec::with_capacity(nd);
    for _ in 0..nd {
        dicts.push(SdgDict::deserialize(&mut r).ok()?);
    }
    // queries (AST rebuild)
    let nq = read_u32(&mut r)? as usize;
    let mut queries = Vec::with_capacity(nq);
    for _ in 0..nq {
        let sdg = read_str_pad(&mut r)?;
        let nb = read_u32(&mut r)? as usize;
        let mut blocks = Vec::with_capacity(nb.min(4096));
        for _ in 0..nb {
            blocks.push(read_node(&mut r)?);
        }
        queries.push(Query { sdg, blocks });
    }
    Some(CacheData { queries, patterns, flats, dicts })
}
