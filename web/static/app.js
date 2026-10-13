/* SDG Paper Matcher — frontend logic (no frameworks, plain fetch) */

const FIELDS = ['title', 'authors', 'year', 'journal', 'doi', 'keywords', 'abstract'];

// official UN short names + brand colors
const SDGS = {
  '01': ['No Poverty', '#E5243B'], '02': ['Zero Hunger', '#DDA63A'],
  '03': ['Good Health and Well-being', '#4C9F38'], '04': ['Quality Education', '#C5192D'],
  '05': ['Gender Equality', '#FF3A21'], '06': ['Clean Water and Sanitation', '#26BDE2'],
  '07': ['Affordable and Clean Energy', '#FCC30B'], '08': ['Decent Work and Economic Growth', '#A21942'],
  '09': ['Industry, Innovation and Infrastructure', '#FD6925'], '10': ['Reduced Inequalities', '#DD1367'],
  '11': ['Sustainable Cities and Communities', '#FD9D24'], '12': ['Responsible Consumption and Production', '#BF8B2E'],
  '13': ['Climate Action', '#3F7E44'], '14': ['Life Below Water', '#0A97D9'],
  '15': ['Life on Land', '#56C02B'], '16': ['Peace, Justice and Strong Institutions', '#00689D'],
  '17': ['Partnerships for the Goals', '#19486A'],
};

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-body').forEach(b => b.hidden = b.id !== 'tab-' + name);
  if (name === 'advanced' && !ADV.loaded) loadAdvanced();
}

function setField(id, v) {
  const el = document.getElementById('f-' + id);
  if (el) el.value = Array.isArray(v) ? v.join(', ') : (v || '');
}

async function loadSample(name) {
  const r = await fetch('/sample?name=' + encodeURIComponent(name) + '&format=json');
  if (!r.ok) { alert('sample not found: ' + name); return; }
  const d = await r.json();
  FIELDS.forEach(k => { if (d[k] !== undefined) setField(k, d[k]); });
  const paperEl = document.getElementById('paper');
  if (paperEl) paperEl.value = d.raw || '';
  switchTab('form');
  document.getElementById('f-title').focus();
}

async function fetchDOI() {
  const inp = document.getElementById('f-doi-fetch');
  const btn = document.getElementById('doi-btn');
  const st = document.getElementById('doi-status');
  const doi = inp.value.trim() || document.getElementById('f-doi').value.trim();
  if (!doi) { st.textContent = 'enter a DOI first'; return; }
  btn.disabled = true; st.textContent = 'fetching…';
  try {
    const r = await fetch('/doi?doi=' + encodeURIComponent(doi));
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'lookup failed');
    FIELDS.forEach(k => { if (d[k] !== undefined) setField(k, d[k]); });
    st.textContent = '✓ filled from Crossref';
  } catch (err) {
    st.textContent = '✗ ' + err.message;
  } finally {
    btn.disabled = false;
  }
}

async function run() {
  const go = document.getElementById('go');
  go.disabled = true; go.textContent = 'Matching…';
  const fd = new FormData();
  FIELDS.forEach(k => fd.append(k, document.getElementById('f-' + k).value));
  fd.append('top', document.getElementById('top').value);
  fd.append('maxkw', document.getElementById('maxkw').value);
  try {
    const r = await fetch('/match', { method: 'POST', body: fd });
    const res = document.getElementById('results');
    res.innerHTML = await r.text();
    res.scrollIntoView({ behavior: 'smooth', block: 'start' });
    document.getElementById('validate-row').hidden = !r.ok;
    document.getElementById('validate-panel').innerHTML = '';
  } catch (err) {
    document.getElementById('results').innerHTML =
      '<div class="error-box">Request failed: ' + escapeHtml(String(err)) + '</div>';
    document.getElementById('validate-row').hidden = true;
  } finally {
    go.disabled = false; go.textContent = 'Match SDGs';
  }
}

/* ---- Validate a result: hash it server-side, store it, show a QR ---- */

async function validateResult() {
  const btn = document.getElementById('validate-btn');
  const panel = document.getElementById('validate-panel');
  btn.disabled = true; btn.textContent = 'Validating…';
  panel.innerHTML = '<div class="card"><p class="muted-text">hashing and storing…</p></div>';
  try {
    // Only title/keywords/abstract/matched SDGs are ever stored server-side
    // (see /api/validate) — authors/year/journal/DOI are deliberately left
    // out of what gets persisted, so only send what's actually kept.
    const sdgsMatched = Array.from(document.querySelectorAll('.chip.matched'))
      .map(el => el.textContent.replace(/[^\d]/g, '')).filter(Boolean);
    const payload = {
      title: document.getElementById('f-title').value,
      keywords: document.getElementById('f-keywords').value,
      abstract: document.getElementById('f-abstract').value,
      sdgs_matched: sdgsMatched,
    };

    const r = await fetch('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'validation failed');

    panel.innerHTML = `
      <div class="card validate-card">
        <h3>✅ Result validated</h3>
        <p class="hint">This paper's result has been hashed (SHA-256) and stored. Scan the QR code
        or copy the link below to let anyone verify it was submitted here.</p>
        <div class="validate-layout">
          <div class="validate-qr">${d.qr_svg}</div>
          <div class="validate-details">
            <div class="validate-field">
              <label>Verification code</label>
              <div class="validate-copy-row">
                <input readonly value="${escapeHtml(d.code)}">
                <button class="ghost" onclick="copyText('${escapeHtml(d.code)}')">Copy</button>
              </div>
            </div>
            <div class="validate-field">
              <label>Verify URL (copyable, works on any system)</label>
              <div class="validate-copy-row">
                <input readonly value="${escapeHtml(d.verify_url)}">
                <button class="ghost" onclick="copyText('${escapeHtml(d.verify_url)}')">Copy</button>
              </div>
            </div>
            <div class="validate-field">
              <label>SHA-256</label>
              <div class="validate-hash">${escapeHtml(d.hash)}</div>
            </div>
            <p class="validate-open-link"><a href="${escapeHtml(d.verify_url)}" target="_blank" rel="noopener">Open verify page ↗</a></p>
          </div>
        </div>
      </div>`;
  } catch (err) {
    panel.innerHTML = '<div class="error-box">✗ ' + escapeHtml(err.message) + '</div>';
  } finally {
    btn.disabled = false; btn.textContent = '✓ Validate & Get QR Code';
  }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    if (document.execCommand) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
  }
}

function clearAll() {
  FIELDS.forEach(k => document.getElementById('f-' + k).value = '');
  document.getElementById('results').innerHTML = '';
  document.getElementById('validate-row').hidden = true;
  document.getElementById('validate-panel').innerHTML = '';
  ADV.loaded = false; ADV.all = [];
  const adv = document.getElementById('adv-list');
  if (adv) adv.innerHTML = '';
  const st = document.getElementById('adv-status');
  if (st) st.textContent = '';
  document.getElementById('f-title').focus();
}

/* ---- Advanced tab: full per-SDG keyword browser (deterministic scores) ---- */

const ADV = { sdg: '10', all: [], shown: 200, loaded: false };

function buildSdgSelect() {
  const sel = document.getElementById('adv-sdg');
  sel.innerHTML = Object.entries(SDGS).map(([no, [name, color]]) =>
    `<option value="${no}">${no} — ${escapeHtml(name)}</option>`).join('');
  sel.value = ADV.sdg;
}

async function loadAdvanced() {
  const status = document.getElementById('adv-status');
  status.textContent = 'scoring keywords…';
  const fd = new FormData();
  FIELDS.forEach(k => fd.append(k, document.getElementById('f-' + k).value));
  fd.append('sdg', document.getElementById('adv-sdg').value);
  fd.append('limit', '2000');
  try {
    const r = await fetch('/api/keywords', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'request failed');
    ADV.sdg = d.sdg; ADV.all = d.keywords; ADV.shown = 200; ADV.loaded = true;
    status.textContent =
      `SDG ${d.sdg} — ${d.total} keywords · ${d.present} already present in your text (they qualify right now)`;
    renderAdvanced();
  } catch (err) {
    status.textContent = '✗ ' + err.message;
  }
}

function kwChip(k) {
  const badges =
    (k.present ? '<span class="flag ok" title="already in your text — qualifies right now">✓</span>' : '') +
    (k.excluded ? '<span class="flag warn" title="also an excluded (NOT) term in this SDG">⚠</span>' : '');
  return `<button type="button" class="kw sug" data-kw="${escapeHtml(k.keyword)}" title="add to Keywords field &amp; copy">
    ${escapeHtml(k.keyword)}<span class="score">${k.score}%</span>${badges}</button>`;
}

function renderAdvanced() {
  const q = document.getElementById('adv-search').value.trim().toLowerCase();
  const sort = document.getElementById('adv-sort').value;
  let list = ADV.all;
  if (q) list = list.filter(k => k.keyword.toLowerCase().includes(q));
  if (sort === 'az') list = list.slice().sort((a, b) => a.keyword.localeCompare(b.keyword));
  const shown = list.slice(0, ADV.shown);
  const el = document.getElementById('adv-list');
  el.innerHTML = shown.length ? shown.map(kwChip).join('') :
    '<span class="muted-text">no keywords match the filter</span>';
  document.getElementById('adv-more').hidden = list.length <= ADV.shown;
}

function showMoreAdvanced() {
  ADV.shown += 200;
  renderAdvanced();
}

async function copyKeyword(kw) {
  try {
    await navigator.clipboard.writeText(kw);
  } catch (e) {
    // Fallback for non-secure contexts / older engines.
    if (document.execCommand) {
      const ta = document.createElement('textarea');
      ta.value = kw;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
  }
  const kf = document.getElementById('f-keywords');
  if (kf) {
    const cur = kf.value.trim();
    kf.value = cur ? cur + ', ' + kw : kw;
  }
}

/* ---- boot: sample cards + legend ---- */

async function buildSamples() {
  try {
    const r = await fetch('/samples');
    const list = await r.json();
    const grid = document.getElementById('sample-grid');
    grid.innerHTML = list.map(s =>
      `<button class="sample-btn" onclick="loadSample('${escapeHtml(s.name)}')">
         <span class="sample-file">📄 ${escapeHtml(s.name)}</span>
         ${s.title ? `<span class="sample-title">${escapeHtml(s.title)}${s.year ? ' · ' + escapeHtml(s.year) : ''}</span>` : ''}
       </button>`).join('') || '<span class="muted-text">no sample papers found in papers/</span>';
  } catch (e) {
    document.getElementById('sample-grid').innerHTML = '<span class="muted-text">could not load samples</span>';
  }
}

function buildLegend() {
  document.getElementById('legend-grid').innerHTML = Object.entries(SDGS).map(([no, [name, color]]) =>
    `<span class="legend-item"><span class="dot" style="background:${color}"></span><b>${no}</b> ${escapeHtml(name)}</span>`
  ).join('');
}

document.getElementById('f-abstract').addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run();
});

/* "Show all N" keyword toggles rendered by the server (kw_tags / groups):
   the hidden .kw-more element sits right before its .kw-toggle button. */
document.addEventListener('click', e => {
  const btn = e.target.closest('.kw-toggle');
  if (!btn) return;
  const more = btn.previousElementSibling;
  if (!more || !more.classList.contains('kw-more')) return;
  const expanded = !more.hidden;
  more.hidden = expanded;
  btn.textContent = expanded ? btn.dataset.all : btn.dataset.few;
});

/* Keyword suggestion chips (report panels + Advanced tab): add to the
   Keywords field, copy, and flash the chip. */
document.addEventListener('click', e => {
  const chip = e.target.closest('button.kw.sug');
  if (!chip) return;
  copyKeyword(chip.dataset.kw);
  chip.classList.add('copied');
  setTimeout(() => chip.classList.remove('copied'), 900);
});

buildSamples();
buildLegend();

/* Usage footer: fetch the cumulative counters and show a compact line.
   Hidden when /api/stats is unavailable (older servers) or still zero. */
async function loadSiteStats() {
  const el = document.getElementById('site-stats');
  if (!el) return;
  try {
    const r = await fetch('/api/stats');
    if (!r.ok) return;
    const s = await r.json();
    const pages = Number(s.pages) || 0;
    const matches = (Number(s.match_html) || 0) + (Number(s.api_match) || 0);
    const users = Number(s.users_total) || 0;
    if (!pages && !users && !matches) return;
    const n = (v, word) => `${v.toLocaleString()} ${word}${v === 1 ? '' : 's'}`;
    el.textContent = `${n(pages, 'visit')} · ${n(users, 'unique visitor')} · ` +
      `${n(matches, 'paper matched')}`;
    el.title = 'today: ' + (Number(s.users_today) || 0).toLocaleString() +
      ' unique · 7d: ' + (Number(s.users_7d) || 0).toLocaleString() +
      ' unique · 30d: ' + (Number(s.users_30d) || 0).toLocaleString() +
      ' unique · total requests: ' + (Number(s.total) || 0).toLocaleString() +
      ' · resets on redeploy';
  } catch (e) { /* keep the footer clean on older servers */ }
}

loadSiteStats();
buildSdgSelect();
