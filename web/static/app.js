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
  document.getElementById('paper').value = d.raw || '';
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
  fd.append('paper', document.getElementById('paper').value);
  fd.append('top', document.getElementById('top').value);
  fd.append('maxkw', document.getElementById('maxkw').value);
  try {
    const r = await fetch('/match', { method: 'POST', body: fd });
    const res = document.getElementById('results');
    res.innerHTML = await r.text();
    res.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    document.getElementById('results').innerHTML =
      '<div class="error-box">Request failed: ' + escapeHtml(String(err)) + '</div>';
  } finally {
    go.disabled = false; go.textContent = 'Match SDGs';
  }
}

function clearAll() {
  FIELDS.forEach(k => document.getElementById('f-' + k).value = '');
  document.getElementById('paper').value = '';
  document.getElementById('results').innerHTML = '';
  document.getElementById('f-title').focus();
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

document.getElementById('file').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    document.getElementById('paper').value = rd.result;
    switchTab('paste');
  };
  rd.readAsText(f);
});

document.querySelectorAll('#f-abstract, #paper').forEach(t =>
  t.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run();
  }));

buildSamples();
buildLegend();
