//! Minimal Turso (libSQL) HTTP client for the "validate a result" feature.
//!
//! Turso exposes a stateless HTTP pipeline API (`POST {url}/v2/pipeline`)
//! that takes a batch of SQL statements and returns their results — no
//! native SQLite driver, no persistent connection, works fine behind
//! Render's free ephemeral filesystem since the data lives in Turso, not
//! on local disk.
//!
//! Configured via two env vars:
//!   TURSO_DB_URL    e.g. https://sdg-paper-validations-xxx.turso.io
//!   TURSO_AUTH_TOKEN
//! If either is unset, `enabled()` returns false and callers should
//! degrade gracefully (e.g. fall back to local-only logging).

use std::sync::OnceLock;
use std::time::Duration;

struct Cfg {
    url: String,
    token: String,
}

static CFG: OnceLock<Option<Cfg>> = OnceLock::new();

fn cfg() -> Option<&'static Cfg> {
    CFG.get_or_init(|| {
        let url = std::env::var("TURSO_DB_URL").ok()?;
        let token = std::env::var("TURSO_AUTH_TOKEN").ok()?;
        if url.is_empty() || token.is_empty() {
            return None;
        }
        // The libsql:// scheme (CLI/show output) maps 1:1 to https:// for
        // the HTTP pipeline API.
        let url = url.replacen("libsql://", "https://", 1);
        Some(Cfg { url, token })
    })
    .as_ref()
}

pub fn enabled() -> bool {
    cfg().is_some()
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(8)))
        .build()
        .new_agent()
}

/// Run a batch of SQL statements (with positional `?` params) in one
/// pipeline call. Returns the raw `results` array from the Turso response.
fn pipeline(stmts: &[(&str, Vec<serde_json::Value>)]) -> Result<serde_json::Value, String> {
    let Some(c) = cfg() else {
        return Err("Turso not configured (TURSO_DB_URL / TURSO_AUTH_TOKEN unset)".into());
    };
    let mut requests: Vec<serde_json::Value> = stmts
        .iter()
        .map(|(sql, args)| {
            serde_json::json!({
                "type": "execute",
                "stmt": { "sql": sql, "args": args.iter().map(arg_val).collect::<Vec<_>>() }
            })
        })
        .collect();
    requests.push(serde_json::json!({"type": "close"}));
    let body = serde_json::json!({ "requests": requests });

    let url = format!("{}/v2/pipeline", c.url.trim_end_matches('/'));
    let mut resp = agent()
        .post(&url)
        .header("Authorization", &format!("Bearer {}", c.token))
        .header("Content-Type", "application/json")
        .send(body.to_string().as_bytes())
        .map_err(|e| format!("Turso request failed: {e}"))?;
    let text = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Turso response read failed: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("Turso bad JSON: {e} (body: {text})"))
}

/// Turso's HTTP API wraps each scalar arg as `{"type":"text"|"integer","value":...}`.
fn arg_val(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => serde_json::json!({"type": "text", "value": s}),
        serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => {
            serde_json::json!({"type": "integer", "value": n.to_string()})
        }
        serde_json::Value::Null => serde_json::json!({"type": "null"}),
        other => serde_json::json!({"type": "text", "value": other.to_string()}),
    }
}

/// Insert one validation record. `code` must be unique (caller generates
/// it from the hash, with a few retries on collision in the extremely
/// unlikely case one occurs).
pub fn insert_validation(
    code: &str,
    hash: &str,
    payload_json: &str,
    ts_ms: i64,
    uid: &str,
) -> Result<(), String> {
    let res = pipeline(&[(
        "INSERT INTO validations (code, hash, payload, created_at, uid) VALUES (?, ?, ?, ?, ?)",
        vec![
            serde_json::json!(code),
            serde_json::json!(hash),
            serde_json::json!(payload_json),
            serde_json::json!(ts_ms),
            serde_json::json!(uid),
        ],
    )])?;
    check_ok(&res)
}

fn check_ok(res: &serde_json::Value) -> Result<(), String> {
    if let Some(arr) = res.get("results").and_then(|r| r.as_array()) {
        for item in arr {
            if item.get("type").and_then(|t| t.as_str()) == Some("error") {
                let msg = item
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown Turso error");
                return Err(msg.to_string());
            }
        }
    }
    Ok(())
}

pub struct ValidationRecord {
    pub code: String,
    pub hash: String,
    pub payload: String,
    pub created_at: i64,
}

/// Look up one validation by its short code. Returns Ok(None) if not found.
pub fn get_validation(code: &str) -> Result<Option<ValidationRecord>, String> {
    let res = pipeline(&[(
        "SELECT code, hash, payload, created_at FROM validations WHERE code = ? LIMIT 1",
        vec![serde_json::json!(code)],
    )])?;
    check_ok(&res)?;
    let rows = res
        .get("results")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
        .and_then(|r| r.get("response"))
        .and_then(|r| r.get("result"))
        .and_then(|r| r.get("rows"))
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    // Each cell is {"type":"text","value":"..."}.
    let cell = |i: usize| -> String {
        row.get(i)
            .and_then(|c| c.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    };
    Ok(Some(ValidationRecord {
        code: cell(0),
        hash: cell(1),
        payload: cell(2),
        created_at: row
            .get(3)
            .and_then(|c| c.get("value"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0),
    }))
}
