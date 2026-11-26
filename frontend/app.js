/* EquiLens India Quant v2 Full-Stack Frontend (NSE / NIFTY 50 / INR) */
(function () {
  'use strict';

  const state = {
    view: 'shortlist',
    selectedSymbol: 'RELIANCE.NS',
    shortlist: [],
    sectors: [],
    benchmark: null,
    macroRegime: null,
    mlSummary: null,
    stockDetail: null,
    modelMetrics: null,
    stressData: null,
    journalData: null,
    alertsData: [],
    weights: { ml: 30, mom: 25, val: 15, qual: 15, sent: 15 },
    timeframe: '6M',
    overlays: { cone: true, barriers: true, sma20: true, sma50: true, sma200: false },
    oscillator: 'rsi',
    hoverIdx: null,
    searchFilter: '',
  };

  const PRESETS = {
    balanced: { ml: 30, mom: 25, val: 15, qual: 15, sent: 15 },
    ml: { ml: 55, mom: 20, val: 10, qual: 5, sent: 10 },
    momentum: { ml: 20, mom: 50, val: 5, qual: 10, sent: 15 },
    value: { ml: 15, mom: 10, val: 35, qual: 30, sent: 10 },
  };

  // =========================================================================
  // Indian Numbering & Currency Formatters (INR ₹, Lakhs, Crores)
  // =========================================================================
  function dispSym(sym) {
    if (!sym) return '';
    if (sym === '^NSEI') return 'NIFTY 50';
    return sym.replace(/\.NS$/i, '').replace(/\.BO$/i, ' (BSE)');
  }
  function fmtPrice(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
    return '₹' + Number(v).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function fmtMoney(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
    const n = Number(v);
    if (Math.abs(n) >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
    if (Math.abs(n) >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
    return '₹' + n.toLocaleString('en-IN', { maximumFractionDigits: 0 });
  }
  function fmtMcapCr(cr) {
    if (cr === null || cr === undefined || Number.isNaN(Number(cr))) return '--';
    const n = Number(cr);
    if (n >= 100000) return `₹${(n / 100000).toFixed(2)}L Cr`;
    return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })} Cr`;
  }
  function fmtPct(v, d = 2) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
    const n = Number(v);
    return `${n > 0 ? '+' : ''}${n.toFixed(d)}%`;
  }
  function fmtVol(v) {
    if (!v) return '--';
    if (v >= 1e7) return (v / 1e7).toFixed(2) + ' Cr';
    if (v >= 1e5) return (v / 1e5).toFixed(2) + ' L';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    return String(v);
  }
  function escHtml(s) {
    return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  let toastTimer = null;
  function showToast(msg, duration = 3800) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), duration);
  }

  function getApiToken() {
    const meta = document.querySelector('meta[name="equilens-token"]');
    return meta ? meta.getAttribute('content') || '' : '';
  }

  function safeExternalUrl(raw) {
    if (!raw || typeof raw !== 'string') return null;
    try {
      const u = new URL(raw.trim());
      return (u.protocol === 'https:' || u.protocol === 'http:') ? u.href : null;
    } catch {
      return null;
    }
  }

  async function apiFetch(url, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    if (method !== 'GET' && method !== 'HEAD') {
      const tok = getApiToken();
      if (tok) headers['X-EquiLens-Token'] = tok;
    }
    const res = await fetch(url, { ...options, headers });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    return data;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  const scratch = document.createElement('canvas');
  scratch.width = 1; scratch.height = 1;
  const sCtx = scratch.getContext('2d', { willReadFrequently: true });
  function withAlpha(color, alpha) {
    sCtx.clearRect(0, 0, 1, 1);
    sCtx.fillStyle = color;
    sCtx.fillRect(0, 0, 1, 1);
    const d = sCtx.getImageData(0, 0, 1, 1).data;
    return `rgba(${d[0]}, ${d[1]}, ${d[2]}, ${alpha})`;
  }

  function convictionBadge(c) {
    if (c === 'Strong Candidate') return 'badge-up';
    if (c === 'Accumulate / Watch') return 'badge-accent';
    if (c === 'Caution') return 'badge-down';
    return 'badge-neutral';
  }

  function sparklineSvg(vals, isUp) {
    if (!vals || vals.length < 2) return '';
    const min = Math.min(...vals), max = Math.max(...vals);
    const span = Math.max(0.01, max - min);
    const pts = vals.map((v, i) => `${((i / (vals.length - 1)) * 66).toFixed(1)},${(22 - ((v - min) / span) * 20).toFixed(1)}`).join(' ');
    return `<svg width="66" height="24" viewBox="0 0 66 24"><polyline fill="none" stroke="${isUp ? 'var(--up)' : 'var(--down)'}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" points="${pts}"/></svg>`;
  }

  function setupCanvas(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(120, Math.floor(rect.width));
    const h = Math.max(80, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function drawLine(ctx, xs, ys, color, width = 1.5, dash = []) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < xs.length; i++) {
      if (ys[i] !== null && ys[i] !== undefined && Number.isFinite(ys[i])) {
        if (!started) { ctx.moveTo(xs[i], ys[i]); started = true; }
        else ctx.lineTo(xs[i], ys[i]);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  // =========================================================================
  // API Loaders
  // =========================================================================
  async function loadShortlist() {
    const sector = document.getElementById('filter-sector').value || 'ALL';
    const setup = document.getElementById('filter-setup').value || 'ALL';
    const qs = new URLSearchParams({
      w_ml: (state.weights.ml / 100).toFixed(2),
      w_mom: (state.weights.mom / 100).toFixed(2),
      w_val: (state.weights.val / 100).toFixed(2),
      w_qual: (state.weights.qual / 100).toFixed(2),
      w_sent: (state.weights.sent / 100).toFixed(2),
      sector,
      setup,
    });
    const data = await apiFetch(`/api/shortlist?${qs}`);
    state.shortlist = data.items || [];
    state.sectors = data.sectors || [];
    state.benchmark = data.benchmark;
    state.macroRegime = data.macro_regime;
    state.mlSummary = data.ml_summary;

    const ab = document.getElementById('tab-alert-badge');
    if (data.unread_alerts > 0) {
      ab.textContent = String(data.unread_alerts);
      ab.style.display = 'inline-flex';
    } else {
      ab.style.display = 'none';
    }

    if (state.shortlist.length && !state.shortlist.some((x) => x.symbol === state.selectedSymbol)) {
      state.selectedSymbol = state.shortlist[0].symbol;
    }

    populateDropdowns();
    renderHeaderAndRegime();
    renderShortlistView();
  }

  async function loadStockDetail(symbol) {
    state.selectedSymbol = symbol.toUpperCase();
    state.hoverIdx = null;
    state.stockDetail = await apiFetch(`/api/stock/${encodeURIComponent(state.selectedSymbol)}`);
    document.getElementById('chart-symbol-select').value = state.selectedSymbol;
    renderStockDetailView();
  }

  async function loadModelMetrics() {
    state.modelMetrics = await apiFetch('/api/model/metrics');
    renderModelView();
  }

  async function loadStressTests() {
    const topK = document.getElementById('st-topk').value;
    const rebal = document.getElementById('st-rebal').value;
    const gate = document.getElementById('st-regime-gate').checked;
    const riskEl = document.getElementById('st-risk-profile');
    const riskProfile = riskEl ? riskEl.value : 'balanced';
    const btn = document.getElementById('btn-run-stress');
    btn.disabled = true;
    btn.textContent = 'Stress Testing...';
    try {
      state.stressData = await apiFetch(`/api/portfolio/stress?top_k=${topK}&rebalance_days=${rebal}&regime_gate=${gate}&risk_profile=${encodeURIComponent(riskProfile)}`);
      renderStressView();
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run Stress Suite';
    }
  }

  async function loadSingleBacktest() {
    const sym = document.getElementById('bt-symbol').value || state.selectedSymbol;
    const strat = document.getElementById('bt-strategy').value;
    const bt = await apiFetch(`/api/backtest/${encodeURIComponent(sym)}?strategy=${strat}`);
    const m = bt.metrics;
    document.getElementById('bt-mini-kpis').innerHTML = [
      ['Strategy Return', `<strong class="${m.strategy_return_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(m.strategy_return_pct)}</strong> (Final: ${fmtMoney(m.final_equity)})`],
      ['Buy & Hold Return', `<strong>${fmtPct(m.buy_hold_return_pct)}</strong> (Alpha: ${fmtPct(m.alpha_pct)})`],
      ['Sharpe / Max DD', `<strong>${m.sharpe_ratio.toFixed(2)}</strong> / <span class="text-down">${m.max_drawdown_pct.toFixed(1)}%</span>`],
      ['Trades / Win Rate', `${m.trade_count} trades (${m.win_rate_pct.toFixed(0)}% win rate · ${bt.commission_bps} bps cost)`],
    ].map(([k, v]) => `<div class="kv-row"><span class="text-muted">${k}</span><span>${v}</span></div>`).join('');
  }

  async function loadJournal() {
    state.journalData = await apiFetch('/api/journal');
    renderJournalView();
  }

  async function loadAlerts() {
    const data = await apiFetch('/api/alerts');
    state.alertsData = data.alerts || [];
    const ab = document.getElementById('tab-alert-badge');
    if (data.unread_count > 0) {
      ab.textContent = String(data.unread_count);
      ab.style.display = 'inline-flex';
    } else {
      ab.style.display = 'none';
    }
    renderAlertsView();
  }

  function populateDropdowns() {
    const sec = document.getElementById('filter-sector');
    const prevSec = sec.value;
    sec.innerHTML = `<option value="ALL">All Sectors</option>` + state.sectors.map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`).join('');
    if ([...sec.options].some((o) => o.value === prevSec)) sec.value = prevSec;

    const syms = state.shortlist.map((s) => ({ symbol: s.symbol, name: s.name }));
    if (state.benchmark && !syms.some((x) => x.symbol === state.benchmark.symbol)) {
      syms.push({ symbol: state.benchmark.symbol, name: 'NIFTY 50 Index' });
    }
    syms.sort((a, b) => dispSym(a.symbol).localeCompare(dispSym(b.symbol)));
    ['chart-symbol-select', 'bt-symbol'].forEach((id) => {
      const sel = document.getElementById(id);
      const prev = sel.value || state.selectedSymbol;
      sel.innerHTML = syms.map((s) => `<option value="${escHtml(s.symbol)}">${escHtml(dispSym(s.symbol))} — ${escHtml(s.name)}</option>`).join('');
      if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
    });
  }

  function renderHeaderAndRegime() {
    const parts = [];
    if (state.benchmark) {
      const b = state.benchmark;
      parts.push(`<span><strong>${b.label || 'NIFTY 50'}</strong> ${Number(b.price).toLocaleString('en-IN', { minimumFractionDigits: 2 })} <span class="${b.chg_1d_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(b.chg_1d_pct)}</span></span>`);
      parts.push(`<span>${b.as_of}</span>`);
    }
    if (state.mlSummary) {
      parts.push(`<span>OOS 20D Hit: <strong>${state.mlSummary.hit_rate}%</strong></span>`);
    }
    document.getElementById('bench-strip').innerHTML = parts.join(' &middot; ');

    const reg = state.macroRegime;
    if (reg) {
      const rb = document.getElementById('regime-badge');
      rb.textContent = reg.regime;
      rb.className = `badge ${reg.tone === 'bullish' ? 'badge-up' : reg.tone === 'bearish' ? 'badge-down' : 'badge-warn'}`;
      const brentHtml = reg.brent_usd !== null && reg.brent_usd !== undefined
        ? `<span>Brent: <strong>$${Number(reg.brent_usd).toFixed(1)}</strong></span>`
        : '';
      const hiloVal = Number(reg.hilo_sentiment_pct || 0);
      const gnVal = Number(reg.gold_nifty_63d_pct || 0);
      document.getElementById('regime-metrics').innerHTML = `
        <span>India VIX: <strong>${reg.vix.toFixed(2)}</strong></span>
        <span>USD/INR: <strong>₹${reg.usdinr.toFixed(2)}</strong></span>
        ${brentHtml}
        <span>NIFTY vs 200D: <strong class="${reg.bench_dist_sma200_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(reg.bench_dist_sma200_pct, 1)}</strong></span>
        <span>Breadth: <strong>${reg.breadth_50d_pct.toFixed(1)}%</strong></span>
        <span>52W Hi-Lo Sent: <strong class="${hiloVal >= 0 ? 'text-up' : 'text-down'}">${fmtPct(hiloVal, 1)}</strong></span>
        <span>Gold/NIFTY 3M: <strong class="${gnVal > 14 ? 'text-down' : 'text-muted'}">${fmtPct(gnVal, 1)}</strong></span>
      `;
    }
  }

  // =========================================================================
  // VIEW 1: NSE SHORTLIST
  // =========================================================================
  function renderShortlistView() {
    const q = state.searchFilter.trim().toLowerCase();
    const list = state.shortlist.filter((it) =>
      !q || it.symbol.toLowerCase().includes(q) || dispSym(it.symbol).toLowerCase().includes(q) || it.name.toLowerCase().includes(q) || it.sector.toLowerCase().includes(q)
    );
    document.getElementById('shortlist-count').textContent = `${list.length} NSE stock${list.length === 1 ? '' : 's'} ranked`;

    if (state.mlSummary) {
      document.getElementById('ml-badge-strip').innerHTML = `
        <span class="badge badge-neutral">IC: ${state.mlSummary.ic_spearman.toFixed(3)}</span>
        <span class="badge ${state.mlSummary.top5_spread >= 0 ? 'badge-up' : 'badge-warn'}">OOS 20D Spread: ${fmtPct(state.mlSummary.top5_spread)}</span>
      `;
    }

    const topGrid = document.getElementById('top-picks-grid');
    const top4 = list.slice(0, 4);
    topGrid.innerHTML = top4.length ? top4.map((it) => `
      <article class="pick-card" data-open-sym="${escHtml(it.symbol)}">
        <div class="pick-top">
          <div>
            <div style="display:flex; align-items:center; gap:6px;">
              <span class="text-muted mono" style="font-size:12px;">#${it.rank}</span>
              <span class="pick-ticker">${escHtml(dispSym(it.symbol))}</span>
              <span class="badge ${convictionBadge(it.conviction)}">${escHtml(it.conviction)}</span>
            </div>
            <div class="pick-name">${escHtml(it.name)} &middot; ${escHtml(it.sector)}</div>
          </div>
          <div class="score-ring tabular ${it.composite_score >= 68 ? 'badge-up' : 'badge-accent'}">${it.composite_score.toFixed(0)}</div>
        </div>

        <div style="display:flex; align-items:center; justify-content:space-between;" class="tabular">
          <div>
            <div style="font-size:17px; font-weight:700;">${fmtPrice(it.price)}</div>
            <div class="${it.chg_1d_pct >= 0 ? 'text-up' : 'text-down'}" style="font-size:12px; font-weight:600;">${fmtPct(it.chg_1d_pct)} (1D)</div>
          </div>
          ${sparklineSvg(it.sparkline, it.chg_1m_pct >= 0)}
        </div>

        <div class="pick-metrics tabular">
          <div><span class="text-muted">TB Win Odds:</span> <strong class="${it.ml.prob_tb_win >= 55 ? 'text-up' : ''}">${it.ml.prob_tb_win.toFixed(0)}%</strong></div>
          <div><span class="text-muted">ML Alpha:</span> <strong class="${it.ml.pred_alpha_20d >= 0 ? 'text-up' : 'text-down'}">${fmtPct(it.ml.pred_alpha_20d)}</strong></div>
          <div><span class="text-muted">Target:</span> <strong class="text-up">${fmtPrice(it.ml.target_price)}</strong></div>
          <div><span class="text-muted">Stop:</span> <strong class="text-down">${fmtPrice(it.ml.stop_price)}</strong></div>
        </div>

        <div class="pick-driver">${escHtml(it.bullish_drivers[0] || '')}</div>
      </article>
    `).join('') : `<div class="panel" style="grid-column:1/-1; text-align:center; color:var(--text-muted);">No stocks match active filters.</div>`;

    const tbody = document.getElementById('shortlist-table-body');
    tbody.innerHTML = list.length ? list.map((it) => {
      const setupTags = it.setups.slice(0, 2).map((s) =>
        `<span class="badge ${s.tone === 'bullish' ? 'badge-up' : s.tone === 'warning' ? 'badge-warn' : 'badge-neutral'}" style="margin-right:4px;">${escHtml(s.label)}</span>`
      ).join('') || `<span class="text-muted">—</span>`;
      return `<tr class="clickable" data-open-sym="${escHtml(it.symbol)}">
        <td class="text-muted">${it.rank}</td>
        <td>
          <div style="font-weight:700;">${escHtml(dispSym(it.symbol))} ${it.starred ? '<span class="text-warn">&#9733;</span>' : ''}</div>
          <div class="text-muted" style="max-width:155px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escHtml(it.name)}</div>
        </td>
        <td>${sparklineSvg(it.sparkline, it.chg_1m_pct >= 0)}</td>
        <td class="num" style="font-weight:600;">${fmtPrice(it.price)}</td>
        <td class="num ${it.chg_1d_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(it.chg_1d_pct)}</td>
        <td class="num ${it.ml.prob_tb_win >= 55 ? 'text-up' : ''}" style="font-weight:600;">${it.ml.prob_tb_win.toFixed(1)}%</td>
        <td class="num ${it.ml.pred_alpha_20d >= 0 ? 'text-up' : 'text-down'}">${fmtPct(it.ml.pred_alpha_20d)}</td>
        <td class="num"><span class="text-up">${fmtPrice(it.ml.target_price)}</span> / <span class="text-down">${fmtPrice(it.ml.stop_price)}</span></td>
        <td class="num text-muted">${fmtMcapCr(it.market_cap_cr)}</td>
        <td>${setupTags}</td>
        <td class="num"><span class="badge ${convictionBadge(it.conviction)}">${it.composite_score.toFixed(1)}</span></td>
      </tr>`;
    }).join('') : `<tr><td colspan="11" style="text-align:center; padding:20px; color:var(--text-muted);">No matching NSE stocks.</td></tr>`;

    document.querySelectorAll('[data-open-sym]').forEach((el) => {
      el.addEventListener('click', async () => {
        switchView('chart');
        await loadStockDetail(el.dataset.openSym);
      });
    });
  }

  // =========================================================================
  // VIEW 2: CHART, ATR BARRIERS, OWNERSHIP, EARNINGS & INSIDERS
  // =========================================================================
  function renderStockDetailView() {
    const d = state.stockDetail;
    if (!d) return;
    const t = d.ticker, st = d.stats, sc = d.score_card, ml = d.ml_prediction;
    const fund = t.fundamentals || {};

    document.getElementById('det-name').textContent = t.name;
    document.getElementById('det-sector').textContent = t.sector;
    document.getElementById('det-price').textContent = fmtPrice(st.close);
    const chg = st.close - st.prev_close;
    const chgPct = (chg / Math.max(1e-9, st.prev_close)) * 100;
    const chgEl = document.getElementById('det-chg');
    chgEl.textContent = `${chg >= 0 ? '+' : ''}₹${Math.abs(chg).toFixed(2)} (${fmtPct(chgPct)})`;
    chgEl.className = chg >= 0 ? 'text-up' : 'text-down';
    document.getElementById('det-source').textContent = `${t.data_source} · ${d.series[d.series.length - 1].date}`;
    document.getElementById('btn-det-star').innerHTML = t.starred ? '<span class="text-warn">&#9733;</span> Starred' : '&#9734; Star';
    if (sc) {
      const cb = document.getElementById('det-conviction-badge');
      cb.textContent = `Score ${sc.composite_score.toFixed(1)} · ${sc.conviction}`;
      cb.className = `badge ${convictionBadge(sc.conviction)}`;
    }

    const cone = (ml && ml.cone) || [];
    const c60 = cone[cone.length - 1];
    document.getElementById('det-ml-box').innerHTML = ml ? [
      ['Triple-Barrier Win Prob', `<strong class="${ml.prob_tb_win >= 55 ? 'text-up' : ''}">${ml.prob_tb_win.toFixed(1)}%</strong>`],
      ['ATR Profit Target (+2.5×)', `<strong class="text-up">${fmtPrice(ml.target_price)}</strong>`],
      ['ATR Stop-Loss (-1.5×)', `<strong class="text-down">${fmtPrice(ml.stop_price)}</strong> (${ml.risk_reward}R)`],
      ['20D ML Alpha vs NIFTY 50', `<strong class="${ml.pred_alpha_20d >= 0 ? 'text-up' : 'text-down'}">${fmtPct(ml.pred_alpha_20d)}</strong>`],
      ['P(Beat NIFTY 50 in 20D)', `<strong>${ml.prob_outperform.toFixed(1)}%</strong>`],
      ['60D P50 / Range', c60 ? `<strong>${fmtPrice(c60.p50)}</strong> (${fmtPrice(c60.p10)}–${fmtPrice(c60.p90)})` : '--'],
    ].map(([k, v]) => `<div class="kv-row"><span class="text-muted">${k}</span><span>${v}</span></div>`).join('') : '<div class="text-muted">No ML forecast yet.</div>';

    if (sc) {
      const p = sc.pillars;
      document.getElementById('det-pillars').innerHTML = [
        ['ML Triple-Barrier & Alpha', p.ml, 'var(--chart-1)'],
        ['Momentum & RS vs NIFTY', p.momentum, 'var(--chart-3)'],
        ['Valuation & Target', p.valuation, 'var(--chart-2)'],
        ['Quality & PEAD Results', p.quality, 'var(--accent)'],
        ['Insiders & News Catalysts', p.sentiment, 'var(--chart-5)'],
      ].map(([lbl, val, col]) => `
        <div class="meter-row tabular">
          <div style="display:flex; justify-content:space-between;"><span class="text-muted">${lbl}</span><strong>${val.toFixed(1)}</strong></div>
          <div class="meter-track"><div class="meter-fill" style="width:${val}%; background:${col};"></div></div>
        </div>
      `).join('');
      document.getElementById('det-drivers').innerHTML = sc.bullish_drivers.map((x) => `<li>${escHtml(x)}</li>`).join('');
      document.getElementById('det-risks').innerHTML = sc.risk_flags.map((x) => `<li class="text-muted">${escHtml(x)}</li>`).join('');
    }

    document.getElementById('det-setups').innerHTML = d.setups.length
      ? d.setups.map((s) => `<span class="badge ${s.tone === 'bullish' ? 'badge-up' : s.tone === 'warning' ? 'badge-warn' : 'badge-neutral'}" title="${escHtml(s.detail)}">${escHtml(s.label)}</span>`).join('')
      : `<span class="text-muted" style="font-size:12px;">No special setup triggered today.</span>`;

    // Shareholding (Promoter / Institutional) + Quarterly Results
    document.getElementById('det-ownership-box').innerHTML = [
      ['Promoter / Insider Holding', fund.promoter_held_pct !== null && fund.promoter_held_pct !== undefined ? `<strong>${fund.promoter_held_pct.toFixed(1)}%</strong>` : '--'],
      ['FII / DII Institutional Holding', fund.institution_held_pct !== null && fund.institution_held_pct !== undefined ? `<strong>${fund.institution_held_pct.toFixed(1)}%</strong>` : '--'],
      ['20D Volatility / ATR %', `<strong>${st.vol_20d_pct ?? '--'}%</strong> / ${st.atr_pct ?? '--'}%`],
    ].map(([k, v]) => `<div class="kv-row"><span class="text-muted">${k}</span><span>${v}</span></div>`).join('');

    const earn = (d.earnings || []).slice(0, 6);
    document.getElementById('det-earnings-box').innerHTML = earn.length ? `
      <table class="data-table tabular">
        <thead><tr><th>Date</th><th class="num">Est EPS</th><th class="num">Actual</th><th class="num">Surprise</th></tr></thead>
        <tbody>${earn.map((e) => `
          <tr>
            <td>${escHtml(e.earnings_date)}</td>
            <td class="num">${e.eps_estimate !== null ? fmtPrice(e.eps_estimate) : '--'}</td>
            <td class="num">${e.eps_actual !== null ? fmtPrice(e.eps_actual) : '<span class="badge badge-warn">Upcoming</span>'}</td>
            <td class="num ${e.surprise_pct > 0 ? 'text-up' : e.surprise_pct < 0 ? 'text-down' : ''}">${e.surprise_pct !== null ? fmtPct(e.surprise_pct, 1) : '--'}</td>
          </tr>
        `).join('')}</tbody>
      </table>
    ` : '<div class="text-muted" style="font-size:12px;">No quarterly EPS surprise history available.</div>';

    const ins = d.insiders || [];
    document.getElementById('det-insiders-box').innerHTML = ins.length ? `
      <table class="data-table tabular">
        <thead><tr><th>Date</th><th>Insider / Promoter</th><th>Type</th><th class="num">Shares / Value</th></tr></thead>
        <tbody>${ins.slice(0, 10).map((r) => `
          <tr>
            <td class="text-muted">${escHtml(r.trade_date)}</td>
            <td><div>${escHtml(r.insider)}</div><div class="text-muted">${escHtml(r.position)}</div></td>
            <td><span class="badge ${r.is_purchase ? 'badge-up' : 'badge-neutral'}">${escHtml(r.transaction_type)}</span></td>
            <td class="num">${r.value > 0 ? fmtMoney(r.value) : (r.shares ? Number(r.shares).toLocaleString('en-IN') + ' sh' : '--')}</td>
          </tr>
        `).join('')}</tbody>
      </table>
    ` : '<div class="text-muted" style="font-size:12px;">No recent NSE/SEBI insider filings parsed for this stock.</div>';

    const tUp = fund.analyst_target && st.close ? ((fund.analyst_target / st.close) - 1) * 100 : null;
    document.getElementById('det-fundamentals').innerHTML = [
      ['Market Cap (INR)', fmtMcapCr(fund.market_cap_cr)],
      ['Trailing / Forward P/E', `${fund.trailing_pe ? fund.trailing_pe.toFixed(1) + 'x' : '--'} / ${fund.forward_pe ? fund.forward_pe.toFixed(1) + 'x' : '--'}`],
      ['Revenue Growth (YoY)', fund.revenue_growth_pct !== null && fund.revenue_growth_pct !== undefined ? fmtPct(fund.revenue_growth_pct, 1) : '--'],
      ['Gross / Operating Margin', `${fund.gross_margin_pct ?? '--'}% / ${fund.operating_margin_pct ?? '--'}%`],
      ['ROE / Dividend Yield', `${fund.roe_pct ?? '--'}% / ${fund.dividend_yield_pct ?? '--'}%`],
      ['52-Week Range', `${fmtPrice(st.low_52w)} – ${fmtPrice(st.high_52w)}`],
      ['Analyst Mean Target', fund.analyst_target ? `${fmtPrice(fund.analyst_target)} (${fmtPct(tUp, 1)})` : '--'],
      ['Analyst Consensus', fund.recommendation ? `${fund.recommendation}${fund.analyst_count ? ` (${fund.analyst_count} analysts)` : ''}` : '--'],
    ].map(([k, v]) => `<div class="kv-row"><span class="text-muted">${k}</span><strong>${v}</strong></div>`).join('');

    const news = d.news || [];
    const avgS = news.length ? news.reduce((a, b) => a + b.sentiment_score, 0) / news.length : 0;
    const nb = document.getElementById('det-news-badge');
    nb.textContent = `${avgS >= 0.14 ? 'Bullish' : avgS <= -0.14 ? 'Bearish' : 'Neutral'} (${avgS.toFixed(2)})`;
    nb.className = `badge ${avgS >= 0.14 ? 'badge-up' : avgS <= -0.14 ? 'badge-down' : 'badge-neutral'}`;
    document.getElementById('det-news-list').innerHTML = news.length ? news.map((n) => {
      const bCls = n.sentiment_label === 'Bullish' ? 'badge-up' : n.sentiment_label === 'Bearish' ? 'badge-down' : 'badge-neutral';
      const cTags = (n.catalyst_tags || []).map((tg) => `<span class="badge badge-accent">${escHtml(tg)}</span>`).join(' ');
      const safeUrl = safeExternalUrl(n.url);
      return `<div class="news-item">
        <div class="news-meta">
          <span>${escHtml(n.publisher || 'News')} &middot; ${escHtml((n.published_at || '').slice(0, 10))}</span>
          <span class="pill-group" style="gap:4px;">${cTags}<span class="badge ${bCls}">${escHtml(n.sentiment_label)}</span></span>
        </div>
        ${safeUrl ? `<a class="news-title" href="${escHtml(safeUrl)}" target="_blank" rel="noopener noreferrer">${escHtml(n.title)}</a>` : `<div class="news-title">${escHtml(n.title)}</div>`}
      </div>`;
    }).join('') : '<div class="text-muted" style="font-size:12px;">No recent headlines.</div>';

    requestAnimationFrame(() => { drawDetailMainChart(); drawDetailOscChart(); });
  }

  function getVisibleSlice() {
    if (!state.stockDetail) return { bars: [] };
    const all = state.stockDetail.series;
    const map = { '3M': 65, '6M': 130, '1Y': 252, '3Y': Math.min(all.length, 756) };
    const cnt = Math.min(all.length, map[state.timeframe] || 130);
    return { bars: all.slice(Math.max(0, all.length - cnt)) };
  }

  function drawDetailMainChart() {
    if (state.view !== 'chart' || !state.stockDetail) return;
    const { ctx, w, h } = setupCanvas(document.getElementById('main-chart-canvas'));
    const { bars } = getVisibleSlice();
    if (!bars.length) return;

    const ml = state.stockDetail.ml_prediction;
    const cone = (state.overlays.cone && ml && ml.cone) || [];
    const showCone = cone.length > 1;
    const coneExtra = showCone ? Math.max(18, Math.round(bars.length * 0.24)) : 0;
    const totalSlots = bars.length + coneExtra;

    const cMuted = cssVar('--text-muted'), cBorder = cssVar('--border'), cSurface = cssVar('--surface');
    const cUp = cssVar('--up'), cDown = cssVar('--down'), cAccent = cssVar('--accent');
    const c1 = cssVar('--chart-1'), c2 = cssVar('--chart-2'), c3 = cssVar('--chart-3');

    ctx.clearRect(0, 0, w, h);
    const pad = { top: 14, right: 80, bottom: 26, left: 10 };
    const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;

    let minP = Infinity, maxP = -Infinity, maxVol = 0;
    for (const b of bars) {
      if (b.low < minP) minP = b.low;
      if (b.high > maxP) maxP = b.high;
      if (b.volume > maxVol) maxVol = b.volume;
    }
    if (showCone) { for (const pt of cone) { if (pt.p90 > maxP) maxP = pt.p90; if (pt.p10 < minP) minP = pt.p10; } }
    if (state.overlays.barriers && ml && ml.target_price && ml.stop_price) {
      if (ml.target_price > maxP) maxP = ml.target_price;
      if (ml.stop_price < minP) minP = ml.stop_price;
    }
    const span = Math.max(1, maxP - minP);
    minP -= span * 0.07; maxP += span * 0.07;

    const xFor = (slot) => pad.left + ((slot + 0.5) / totalSlots) * plotW;
    const yFor = (val) => pad.top + ((maxP - val) / (maxP - minP)) * plotH;

    ctx.font = '12px JetBrains Mono, monospace'; ctx.textBaseline = 'middle';
    for (let r = 0; r <= 5; r++) {
      const val = maxP - (r / 5) * (maxP - minP);
      const y = Math.round(pad.top + (r / 5) * plotH) + 0.5;
      ctx.strokeStyle = withAlpha(cBorder, 0.55);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillStyle = cMuted; ctx.textAlign = 'left';
      ctx.fillText(`₹${val.toFixed(val >= 1000 ? 0 : 1)}`, w - pad.right + 6, y);
    }

    const step = Math.max(1, Math.floor(bars.length / Math.max(3, Math.floor(plotW / 120))));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let i = 0; i < bars.length; i += step) {
      const x = Math.round(xFor(i)) + 0.5;
      ctx.strokeStyle = withAlpha(cBorder, 0.28);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
      ctx.fillStyle = cMuted; ctx.fillText(bars[i].date.slice(2), x, pad.top + plotH + 6);
    }

    if (maxVol > 0) {
      const vH = plotH * 0.16, bw = Math.max(1.2, (plotW / totalSlots) * 0.68);
      for (let i = 0; i < bars.length; i++) {
        const vh = (bars[i].volume / maxVol) * vH;
        ctx.fillStyle = withAlpha(bars[i].close >= bars[i].open ? cUp : cDown, 0.22);
        ctx.fillRect(xFor(i) - bw / 2, pad.top + plotH - vh, bw, vh);
      }
    }

    const xs = bars.map((_, i) => xFor(i));
    const cw = Math.max(1.5, Math.min(12, (plotW / totalSlots) * 0.68));
    for (let i = 0; i < bars.length; i++) {
      const b = bars[i], col = b.close >= b.open ? cUp : cDown;
      ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xs[i], yFor(b.high)); ctx.lineTo(xs[i], yFor(b.low)); ctx.stroke();
      const yO = yFor(b.open), yC = yFor(b.close);
      ctx.fillStyle = col;
      ctx.fillRect(xs[i] - cw / 2, Math.min(yO, yC), cw, Math.max(1.2, Math.abs(yC - yO)));
    }

    if (state.overlays.sma20) drawLine(ctx, xs, bars.map((b) => b.sma20 ? yFor(b.sma20) : null), c1, 1.5);
    if (state.overlays.sma50) drawLine(ctx, xs, bars.map((b) => b.sma50 ? yFor(b.sma50) : null), c2, 1.5);
    if (state.overlays.sma200) drawLine(ctx, xs, bars.map((b) => b.sma200 ? yFor(b.sma200) : null), c3, 1.6);

    if (state.overlays.barriers && ml && ml.target_price && ml.stop_price) {
      const xStart = xFor(Math.max(0, bars.length - 30)), xEnd = w - pad.right;
      const yT = Math.round(yFor(ml.target_price)) + 0.5, yS = Math.round(yFor(ml.stop_price)) + 0.5;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = withAlpha(cUp, 0.85); ctx.beginPath(); ctx.moveTo(xStart, yT); ctx.lineTo(xEnd, yT); ctx.stroke();
      ctx.strokeStyle = withAlpha(cDown, 0.85); ctx.beginPath(); ctx.moveTo(xStart, yS); ctx.lineTo(xEnd, yS); ctx.stroke();
      ctx.setLineDash([]);
    }

    if (showCone) {
      const maxD = cone[cone.length - 1].day || 60, anchor = bars.length - 1;
      const cXs = cone.map((pt) => xFor(anchor + (pt.day / maxD) * coneExtra));

      ctx.beginPath();
      cone.forEach((pt, idx) => idx === 0 ? ctx.moveTo(cXs[idx], yFor(pt.p90)) : ctx.lineTo(cXs[idx], yFor(pt.p90)));
      for (let i = cone.length - 1; i >= 0; i--) ctx.lineTo(cXs[i], yFor(cone[i].p10));
      ctx.closePath(); ctx.fillStyle = withAlpha(cAccent, 0.11); ctx.fill();

      ctx.beginPath();
      cone.forEach((pt, idx) => idx === 0 ? ctx.moveTo(cXs[idx], yFor(pt.p75)) : ctx.lineTo(cXs[idx], yFor(pt.p75)));
      for (let i = cone.length - 1; i >= 0; i--) ctx.lineTo(cXs[i], yFor(cone[i].p25));
      ctx.closePath(); ctx.fillStyle = withAlpha(cAccent, 0.18); ctx.fill();

      drawLine(ctx, cXs, cone.map((pt) => yFor(pt.p50)), cAccent, 2, [5, 3]);
      ctx.fillStyle = cAccent; ctx.font = '600 12px JetBrains Mono, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('+60D', cXs[cXs.length - 1], pad.top + plotH + 6);
    }

    const lastB = bars[bars.length - 1], lY = yFor(lastB.close);
    ctx.fillStyle = lastB.close >= lastB.open ? cUp : cDown;
    ctx.beginPath(); ctx.roundRect(w - pad.right + 2, lY - 10, pad.right - 4, 20, 4); ctx.fill();
    ctx.fillStyle = cSurface; ctx.font = '600 12px JetBrains Mono, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(`₹${lastB.close.toFixed(lastB.close >= 1000 ? 0 : 1)}`, w - pad.right + 2 + (pad.right - 4) / 2, lY);

    const hIdx = state.hoverIdx !== null && state.hoverIdx >= 0 && state.hoverIdx < bars.length ? state.hoverIdx : null;
    if (hIdx !== null) {
      const hx = Math.round(xs[hIdx]) + 0.5, hy = Math.round(yFor(bars[hIdx].close)) + 0.5;
      ctx.strokeStyle = withAlpha(cMuted, 0.7); ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hx, pad.top); ctx.lineTo(hx, pad.top + plotH); ctx.moveTo(pad.left, hy); ctx.lineTo(w - pad.right, hy); ctx.stroke();
      ctx.setLineDash([]);
    }

    const rB = hIdx !== null ? bars[hIdx] : lastB, up = rB.close >= rB.open;
    document.getElementById('det-readout-ohlcv').innerHTML = `
      <span class="text-muted">${rB.date}</span>
      <span>O <strong class="${up ? 'text-up' : 'text-down'}">₹${rB.open.toFixed(2)}</strong></span>
      <span>H <strong class="${up ? 'text-up' : 'text-down'}">₹${rB.high.toFixed(2)}</strong></span>
      <span>L <strong class="${up ? 'text-up' : 'text-down'}">₹${rB.low.toFixed(2)}</strong></span>
      <span>C <strong class="${up ? 'text-up' : 'text-down'}">₹${rB.close.toFixed(2)}</strong></span>
      <span class="text-muted">Vol <strong style="color:var(--text)">${fmtVol(rB.volume)}</strong></span>
    `;
    const parts = [];
    if (state.overlays.sma20 && rB.sma20) parts.push(`<span style="color:var(--chart-1)">SMA20 ₹${rB.sma20.toFixed(1)}</span>`);
    if (state.overlays.sma50 && rB.sma50) parts.push(`<span style="color:var(--chart-2)">SMA50 ₹${rB.sma50.toFixed(1)}</span>`);
    if (state.overlays.sma200 && rB.sma200) parts.push(`<span style="color:var(--chart-3)">SMA200 ₹${rB.sma200.toFixed(1)}</span>`);
    document.getElementById('det-readout-ind').innerHTML = parts.join('');
    document.getElementById('det-range-lbl').textContent = `${bars[0].date} — ${lastB.date}`;
  }

  function drawDetailOscChart() {
    const wrap = document.getElementById('osc-canvas-wrap');
    if (state.view !== 'chart' || !state.stockDetail) return;
    if (state.oscillator === 'none') { wrap.style.display = 'none'; return; }
    wrap.style.display = 'block';
    const { ctx, w, h } = setupCanvas(document.getElementById('osc-chart-canvas'));
    const { bars } = getVisibleSlice();
    if (!bars.length) return;
    const ml = state.stockDetail.ml_prediction;
    const coneExtra = (state.overlays.cone && ml && ml.cone && ml.cone.length > 1) ? Math.max(18, Math.round(bars.length * 0.24)) : 0;
    const totalSlots = bars.length + coneExtra;

    const cMuted = cssVar('--text-muted'), cBorder = cssVar('--border'), cUp = cssVar('--up'), cDown = cssVar('--down'), c1 = cssVar('--chart-1'), c2 = cssVar('--chart-2');
    ctx.clearRect(0, 0, w, h);
    const pad = { top: 20, right: 80, bottom: 8, left: 10 };
    const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
    const xs = bars.map((_, i) => pad.left + ((i + 0.5) / totalSlots) * plotW);
    const active = state.hoverIdx !== null && state.hoverIdx < bars.length ? bars[state.hoverIdx] : bars[bars.length - 1];
    const leg = document.getElementById('osc-legend');

    if (state.oscillator === 'rsi') {
      const yFor = (v) => pad.top + ((100 - v) / 100) * plotH;
      ctx.fillStyle = withAlpha(c1, 0.08); ctx.fillRect(pad.left, yFor(70), plotW, yFor(30) - yFor(70));
      ctx.font = '12px JetBrains Mono, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      [70, 50, 30].forEach((lvl) => {
        const y = Math.round(yFor(lvl)) + 0.5;
        ctx.strokeStyle = withAlpha(cBorder, 0.5); ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = cMuted; ctx.fillText(String(lvl), w - pad.right + 8, y);
      });
      drawLine(ctx, xs, bars.map((b) => b.rsi14 !== null ? yFor(b.rsi14) : null), c1, 1.7);
      leg.innerHTML = `<span class="text-muted">RSI (14)</span><strong style="color:var(--chart-1)">${active.rsi14 !== null ? active.rsi14.toFixed(1) : '--'}</strong>`;
    } else {
      const vals = bars.flatMap((b) => [b.macd, b.macd_signal, b.macd_hist]).filter((v) => v !== null && Number.isFinite(v));
      const maxAbs = Math.max(0.2, ...vals.map(Math.abs)) * 1.15;
      const yFor = (v) => pad.top + ((maxAbs - v) / (2 * maxAbs)) * plotH;
      const y0 = yFor(0);
      ctx.strokeStyle = withAlpha(cBorder, 0.7);
      ctx.beginPath(); ctx.moveTo(pad.left, Math.round(y0) + 0.5); ctx.lineTo(w - pad.right, Math.round(y0) + 0.5); ctx.stroke();
      const bw = Math.max(1.2, (plotW / totalSlots) * 0.6);
      bars.forEach((b, i) => {
        if (b.macd_hist !== null) {
          const y = yFor(b.macd_hist);
          ctx.fillStyle = withAlpha(b.macd_hist >= 0 ? cUp : cDown, 0.55);
          ctx.fillRect(xs[i] - bw / 2, Math.min(y, y0), bw, Math.max(1, Math.abs(y - y0)));
        }
      });
      drawLine(ctx, xs, bars.map((b) => b.macd !== null ? yFor(b.macd) : null), c1, 1.5);
      drawLine(ctx, xs, bars.map((b) => b.macd_signal !== null ? yFor(b.macd_signal) : null), c2, 1.4);
      leg.innerHTML = `<span class="text-muted">MACD</span><span style="color:var(--chart-1)">${active.macd !== null ? active.macd.toFixed(2) : '--'}</span><span style="color:var(--chart-2)">Sig ${active.macd_signal !== null ? active.macd_signal.toFixed(2) : '--'}</span>`;
    }
  }

  // =========================================================================
  // VIEW 3: ML VALIDATION
  // =========================================================================
  function renderModelView() {
    if (!state.modelMetrics) return;
    const { run, predictions } = state.modelMetrics;
    document.getElementById('ml-trained-lbl').textContent = `Trained ${run.trained_at.replace('T', ' ').slice(0, 19)} UTC · ${run.sample_count.toLocaleString()} obs · ${run.symbol_count} NSE symbols`;

    document.getElementById('ml-kpi-grid').innerHTML = [
      ['Triple-Barrier ROC-AUC', run.tb_auc.toFixed(3), `Top-5 TB Precision: ${run.tb_top5_precision}%`, run.tb_auc >= 0.52 ? 'text-up' : ''],
      ['Top-5 vs NIFTY 20D Spread', fmtPct(run.top5_spread), `Top-5 ${fmtPct(run.avg_top5_return)} vs NIFTY ${fmtPct(run.avg_bench_return)}`, run.top5_spread >= 0 ? 'text-up' : 'text-down'],
      ['OOS 20D Hit Rate', `${run.hit_rate.toFixed(1)}%`, 'Rebalance cohorts beating NIFTY 50', run.hit_rate >= 50 ? 'text-up' : 'text-warn'],
      ['Spearman Rank IC', run.ic_spearman.toFixed(3), '5 expanding folds · 25D embargo', run.ic_spearman > 0 ? 'text-up' : ''],
    ].map(([l, v, n, c]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value ${c}">${v}</div><div class="kpi-note">${n}</div></div>`).join('');

    const maxI = Math.max(1, ...run.feature_importances.map((f) => f.importance_pct));
    document.getElementById('ml-feature-bars').innerHTML = run.feature_importances.map((f) => `
      <div class="meter-row tabular">
        <div style="display:flex; justify-content:space-between;"><span>${escHtml(f.label)} <span class="text-muted">(${f.direction === 'positive' ? '+' : '-'})</span></span><strong>${f.importance_pct.toFixed(1)}%</strong></div>
        <div class="meter-track"><div class="meter-fill" style="width:${(f.importance_pct / maxI) * 100}%; background:${f.direction === 'positive' ? 'var(--up)' : 'var(--down)'};"></div></div>
      </div>
    `).join('');

    document.getElementById('ml-folds-body').innerHTML = run.folds.map((f) => `
      <tr>
        <td>Fold ${f.fold}</td>
        <td class="text-muted">${escHtml(f.test_period)}</td>
        <td class="num">${(f.tb_auc || 0.5).toFixed(3)}</td>
        <td class="num">${f.ic_spearman.toFixed(3)}</td>
        <td class="num ${f.top5_return_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(f.top5_return_pct)}</td>
        <td class="num">${fmtPct(f.bench_return_pct)}</td>
      </tr>
    `).join('');

    const rs = run.regime_stats || {};
    const rsRows = [];
    if (rs.calm_vix_lt20) rsRows.push(['Calm India VIX (< 20)', `Top-5 ${fmtPct(rs.calm_vix_lt20.top5_20d_pct)} vs NIFTY ${fmtPct(rs.calm_vix_lt20.spy_20d_pct)} (<strong class="${rs.calm_vix_lt20.spread_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(rs.calm_vix_lt20.spread_pct)}</strong>)`]);
    if (rs.elevated_vix_ge20) rsRows.push(['Elevated India VIX (≥ 20)', `Top-5 ${fmtPct(rs.elevated_vix_ge20.top5_20d_pct)} vs NIFTY ${fmtPct(rs.elevated_vix_ge20.spy_20d_pct)} (<strong class="${rs.elevated_vix_ge20.spread_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(rs.elevated_vix_ge20.spread_pct)}</strong>)`]);
    document.getElementById('ml-regime-box').innerHTML = rsRows.length
      ? rsRows.map(([k, v]) => `<div class="kv-row"><span class="text-muted">${k}</span><span>${v}</span></div>`).join('')
      : '<div class="text-muted">Not enough regime observations yet.</div>';

    document.getElementById('ml-preds-body').innerHTML = predictions.map((p) => {
      const c60 = (p.cone && p.cone[p.cone.length - 1]) || null;
      return `<tr class="clickable" data-ml-sym="${escHtml(p.symbol)}">
        <td style="font-weight:700;">${escHtml(dispSym(p.symbol))}</td>
        <td class="num ${p.prob_tb_win >= 55 ? 'text-up' : ''}" style="font-weight:600;">${p.prob_tb_win.toFixed(1)}%</td>
        <td class="num ${p.pred_alpha_20d >= 0 ? 'text-up' : 'text-down'}">${fmtPct(p.pred_alpha_20d)}</td>
        <td class="num">${p.prob_outperform.toFixed(1)}%</td>
        <td class="num text-up">${fmtPrice(p.target_price)}</td>
        <td class="num text-down">${fmtPrice(p.stop_price)}</td>
        <td class="num">${p.risk_reward ? p.risk_reward.toFixed(2) + 'R' : '--'}</td>
        <td class="num text-muted">${c60 ? `${fmtPrice(c60.p10)} – ${fmtPrice(c60.p90)}` : '--'}</td>
      </tr>`;
    }).join('');
    document.querySelectorAll('[data-ml-sym]').forEach((tr) => {
      tr.addEventListener('click', async () => { switchView('chart'); await loadStockDetail(tr.dataset.mlSym); });
    });
  }

  // =========================================================================
  // VIEW 4: PORTFOLIO & STRESS TESTS
  // =========================================================================
  function renderStressView() {
    const s = state.stressData;
    if (!s) return;
    const ov = s.overall, bp = s.base_portfolio, pm = bp.portfolio_metrics, sm = bp.spy_metrics, bs = s.bootstrap_stress;

    const vb = document.getElementById('stress-verdict-badge');
    vb.textContent = `Stress Verdict: ${ov.verdict} (${ov.checks_passed}/${ov.checks_total} checks · Rank Stability ${ov.ranking_stability_jaccard_pct}%)`;
    vb.className = `badge ${ov.verdict === 'Resilient' ? 'badge-up' : ov.verdict === 'Moderate Risk' ? 'badge-warn' : 'badge-down'}`;
    document.getElementById('st-period-lbl').textContent = `OOS Period: ${bp.config.period} · ${bp.config.rebalance_count} rebalances (avg turnover ${bp.config.avg_turnover_pct}%)`;

    const postTaxStr = pm.after_tax_cagr_pct !== undefined ? ` · Post-20% STCG: ${fmtPct(pm.after_tax_cagr_pct)}` : '';
    document.getElementById('st-kpi-grid').innerHTML = [
      ['OOS Portfolio CAGR', fmtPct(pm.cagr_pct), `Total: ${fmtPct(pm.total_return_pct)}${postTaxStr}`, pm.cagr_pct >= 0 ? 'text-up' : 'text-down'],
      ['NIFTY 50 CAGR', fmtPct(sm.cagr_pct), `Excess Alpha: ${fmtPct(bp.excess_cagr_pct)}/yr`, bp.excess_cagr_pct >= 0 ? 'text-up' : 'text-down'],
      ['Sharpe / Sortino', `${pm.sharpe.toFixed(2)} / ${pm.sortino.toFixed(2)}`, `Vol: ${pm.ann_vol_pct.toFixed(1)}% (NIFTY ${sm.ann_vol_pct.toFixed(1)}%)`, pm.sharpe >= sm.sharpe ? 'text-up' : ''],
      ['Max Drawdown', `${pm.max_drawdown_pct.toFixed(1)}%`, `NIFTY Max DD: ${sm.max_drawdown_pct.toFixed(1)}% · Calmar ${pm.calmar.toFixed(2)}`, 'text-down'],
    ].map(([l, v, n, c]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value ${c}">${v}</div><div class="kpi-note">${n}</div></div>`).join('');

    document.getElementById('st-slippage-body').innerHTML = s.slippage_stress.map((r) => `
      <tr>
        <td>${r.cost_bps} bps</td>
        <td class="num ${r.cagr_pct >= 0 ? 'text-up' : 'text-down'}" style="font-weight:600;">${fmtPct(r.cagr_pct)}</td>
        <td class="num ${r.excess_cagr_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(r.excess_cagr_pct)}</td>
        <td class="num">${r.sharpe.toFixed(2)}</td>
        <td class="num text-down">${r.max_drawdown_pct.toFixed(1)}%</td>
        <td class="num"><span class="badge ${r.survives_friction ? 'badge-up' : 'badge-warn'}">${r.survives_friction ? 'Pass' : 'Thin'}</span></td>
      </tr>
    `).join('');

    document.getElementById('st-regime-body').innerHTML = s.regime_replay.map((r) => `
      <tr>
        <td>${escHtml(r.regime)}</td>
        <td class="num text-muted">${r.trading_days}</td>
        <td class="num ${r.portfolio_ret_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(r.portfolio_ret_pct)}</td>
        <td class="num">${fmtPct(r.spy_ret_pct)}</td>
        <td class="num ${r.spread_pct >= 0 ? 'text-up' : 'text-down'}" style="font-weight:600;">${fmtPct(r.spread_pct)}</td>
      </tr>
    `).join('');

    document.getElementById('st-bootstrap-box').innerHTML = [
      ['Simulations / Horizon', `${bs.simulations.toLocaleString()} paths (5D block bootstrap · ${bs.horizon_trading_days}D)`],
      ['Median 1-Year Return', `<strong class="${bs.median_1y_return_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(bs.median_1y_return_pct)}</strong> (5th pct: ${fmtPct(bs.p05_1y_return_pct)})`],
      ['Prob. of Positive 1-Year Return', `<strong>${bs.prob_positive_1y_pct.toFixed(1)}%</strong>`],
      ['Median / 95% Worst Max Drawdown', `${bs.median_max_dd_pct.toFixed(1)}% / <strong class="text-down">${bs.worst_95_max_dd_pct.toFixed(1)}%</strong>`],
      ['99% Tail Worst-Case Drawdown', `<strong class="text-down">${bs.worst_99_max_dd_pct.toFixed(1)}%</strong>`],
      ['Daily 95% VaR / CVaR (Expected Shortfall)', `${bs.daily_var_95_pct.toFixed(2)}% / <strong class="text-down">${bs.daily_cvar_95_pct.toFixed(2)}%</strong>`],
    ].map(([k, v]) => `<div class="kv-row"><span class="text-muted">${k}</span><span>${v}</span></div>`).join('');

    requestAnimationFrame(drawStressEquityCanvas);
  }

  function drawStressEquityCanvas() {
    if (state.view !== 'stress' || !state.stressData) return;
    const curve = state.stressData.base_portfolio.equity_curve;
    if (!curve.length) return;
    const { ctx, w, h } = setupCanvas(document.getElementById('stress-equity-canvas'));
    const cMuted = cssVar('--text-muted'), cBorder = cssVar('--border'), c1 = cssVar('--chart-1');
    ctx.clearRect(0, 0, w, h);
    const pad = { top: 14, right: 86, bottom: 26, left: 10 };
    const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;

    const vals = curve.flatMap((p) => [p.portfolio, p.spy]);
    let minV = Math.min(...vals), maxV = Math.max(...vals);
    const sp = Math.max(10000, maxV - minV);
    minV -= sp * 0.08; maxV += sp * 0.08;

    const xs = curve.map((_, i) => pad.left + (i / Math.max(1, curve.length - 1)) * plotW);
    const yFor = (v) => pad.top + ((maxV - v) / (maxV - minV)) * plotH;

    ctx.font = '12px JetBrains Mono, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let r = 0; r <= 4; r++) {
      const v = maxV - (r / 4) * (maxV - minV);
      const y = Math.round(pad.top + (r / 4) * plotH) + 0.5;
      ctx.strokeStyle = withAlpha(cBorder, 0.5);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillStyle = cMuted; ctx.fillText(fmtMoney(v), w - pad.right + 6, y);
    }

    drawLine(ctx, xs, curve.map((p) => yFor(p.spy)), withAlpha(cMuted, 0.8), 1.6, [4, 3]);
    drawLine(ctx, xs, curve.map((p) => yFor(p.portfolio)), c1, 2.2);

    ctx.fillStyle = cMuted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    [0, Math.floor(curve.length / 2), curve.length - 1].forEach((idx) => {
      ctx.fillText(curve[idx].date.slice(2), xs[idx], pad.top + plotH + 6);
    });
  }

  // =========================================================================
  // VIEW 5: PREDICTION JOURNAL
  // =========================================================================
  function renderJournalView() {
    const j = state.journalData;
    if (!j) return;
    const s = j.summary;

    document.getElementById('journal-kpi-grid').innerHTML = [
      ['Logged / Graded Picks', `${s.total_logged} / ${s.graded_count}`, 'Live snapshots + monthly OOS cohorts', ''],
      ['Avg 20D Pick Return', fmtPct(s.avg_20d_return_pct), `NIFTY 50 avg 20D: ${fmtPct(s.avg_spy_20d_pct)}`, s.avg_20d_return_pct >= 0 ? 'text-up' : 'text-down'],
      ['Positive 20D Win Rate', `${s.win_rate_positive_pct.toFixed(1)}%`, 'NSE picks finishing green after 20D', s.win_rate_positive_pct >= 55 ? 'text-up' : ''],
      ['Beat NIFTY 50 Rate', `${s.beat_spy_rate_pct.toFixed(1)}%`, 'Picks outperforming NIFTY 50 over 20D', s.beat_spy_rate_pct >= 50 ? 'text-up' : 'text-warn'],
    ].map(([l, v, n, c]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value ${c}">${v}</div><div class="kpi-note">${n}</div></div>`).join('');

    document.getElementById('journal-cal-body').innerHTML = (j.calibration || []).map((b) => `
      <tr>
        <td style="font-weight:600;">${escHtml(b.bucket)}</td>
        <td class="num text-muted">${b.count.toLocaleString()}</td>
        <td class="num ${b.actual_tb_win_pct >= 50 ? 'text-up' : 'text-down'}" style="font-weight:600;">${b.actual_tb_win_pct.toFixed(1)}%</td>
        <td class="num ${b.avg_20d_ret_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(b.avg_20d_ret_pct)}</td>
        <td class="num ${b.avg_excess_vs_spy_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(b.avg_excess_vs_spy_pct)}</td>
      </tr>
    `).join('');

    document.getElementById('journal-entries-body').innerHTML = (j.entries || []).map((e) => {
      const outCls = e.outcome.includes('Win') || e.outcome.includes('Beat') ? 'badge-up' : e.outcome.includes('Stop') || e.outcome.includes('Lag') ? 'badge-down' : 'badge-accent';
      return `<tr class="clickable" data-j-sym="${escHtml(e.symbol)}">
        <td class="text-muted">${escHtml(e.logged_date)}</td>
        <td style="font-weight:700;">${escHtml(dispSym(e.symbol))}</td>
        <td><span class="badge ${e.is_live ? 'badge-accent' : 'badge-neutral'}">${e.is_live ? 'Live' : 'OOS'}</span></td>
        <td class="num">${fmtPrice(e.entry_price)}</td>
        <td class="num">${e.composite_score.toFixed(1)}</td>
        <td class="num">${e.ml_tb_prob.toFixed(1)}%</td>
        <td class="num ${e.ret_5d > 0 ? 'text-up' : e.ret_5d < 0 ? 'text-down' : ''}">${e.ret_5d !== null ? fmtPct(e.ret_5d, 1) : '--'}</td>
        <td class="num ${e.ret_20d > 0 ? 'text-up' : e.ret_20d < 0 ? 'text-down' : ''}" style="font-weight:600;">${e.ret_20d !== null ? fmtPct(e.ret_20d, 1) : '--'}</td>
        <td class="num text-muted">${e.spy_ret_20d !== null ? fmtPct(e.spy_ret_20d, 1) : '--'}</td>
        <td class="num"><span class="badge ${outCls}">${escHtml(e.outcome)}</span></td>
      </tr>`;
    }).join('');
    document.querySelectorAll('[data-j-sym]').forEach((tr) => {
      tr.addEventListener('click', async () => { switchView('chart'); await loadStockDetail(tr.dataset.jSym); });
    });
  }

  // =========================================================================
  // VIEW 6: ALERTS
  // =========================================================================
  function renderAlertsView() {
    const el = document.getElementById('alerts-list');
    if (!state.alertsData.length) {
      el.innerHTML = '<div class="text-muted" style="font-size:12px;">No active alerts. Click "Run Catalyst Scan" to scan the NSE universe.</div>';
      return;
    }
    el.innerHTML = state.alertsData.map((a) => {
      const bCls = a.severity === 'high' ? 'badge-down' : a.severity === 'bullish' ? 'badge-up' : 'badge-warn';
      return `<div class="kpi-card" style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; opacity:${a.acknowledged ? 0.55 : 1};">
        <div>
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <span class="badge ${bCls}">${escHtml(a.alert_type)}</span>
            <button type="button" class="btn" style="padding:2px 8px; font-weight:700;" data-a-sym="${escHtml(a.symbol)}">${escHtml(dispSym(a.symbol))}</button>
            <strong style="font-size:13px;">${escHtml(a.title)}</strong>
          </div>
          <div class="text-muted" style="font-size:12px;">${escHtml(a.detail)}</div>
        </div>
        ${a.acknowledged ? '<span class="text-muted" style="font-size:12px;">Acknowledged</span>' : `<button type="button" class="btn" data-ack="${escHtml(a.id)}">Dismiss</button>`}
      </div>`;
    }).join('');

    el.querySelectorAll('[data-a-sym]').forEach((btn) => {
      btn.addEventListener('click', async () => { switchView('chart'); await loadStockDetail(btn.dataset.aSym); });
    });
    el.querySelectorAll('[data-ack]').forEach((btn) => {
      btn.addEventListener('click', async () => {
        await apiFetch(`/api/alerts/${encodeURIComponent(btn.dataset.ack)}/ack`, { method: 'POST' });
        await loadAlerts();
      });
    });
  }

  // =========================================================================
  // VIEW 5: EXECUTION, TRADE TICKET & FORWARD-PROOF PAPER LEDGER
  // =========================================================================
  async function loadExecutionTicket() {
    const cap = Math.max(25000, Number(document.getElementById('ex-capital').value) || 1000000);
    const risk = document.getElementById('ex-risk-profile').value || 'balanced';
    const topK = document.getElementById('ex-topk').value || '10';
    const skipBlackout = document.getElementById('ex-skip-blackout').checked;
    const shieldEl = document.getElementById('ex-smart-shield');
    const smartShield = shieldEl ? shieldEl.checked : true;
    const usePaper = document.getElementById('ex-use-paper').checked;
    const qs = new URLSearchParams({
      capital_inr: String(cap),
      top_k: String(topK),
      risk_profile: risk,
      skip_earnings_blackout: String(skipBlackout),
      enforce_smart_money_shield: String(smartShield),
      use_paper_holdings: String(usePaper),
    });
    state.executionTicket = await apiFetch(`/api/execution/ticket?${qs}`);
    renderExecutionTicket();
  }

  async function loadPaperLedger() {
    state.paperLedger = await apiFetch('/api/execution/paper/status');
    renderPaperLedger();
  }

  function renderExecutionTicket() {
    const t = state.executionTicket;
    if (!t) return;
    const s = t.summary;
    const skippedSm = t.skipped_smart_money || [];
    const shieldNote = skippedSm.length
      ? ` · Shield skipped ${skippedSm.map((x) => dispSym(x.symbol)).join(', ')}`
      : '';
    const capNote = t.config.max_stock_weight_pct
      ? ` · Caps ≤${t.config.max_stock_weight_pct}%/Stk, ≤${t.config.max_sector_weight_pct}%/Sec`
      : '';
    const brakeNote = s.dd_brake_active ? ' · ⚠️ -10% DD BRAKE ACTIVE (0.5x Risk)' : '';
    document.getElementById('ex-asof-lbl').textContent = `As of ${t.as_of_date} · Top-${t.config.top_k} (Buffer Top-${t.config.rank_buffer}) · Max ${t.config.max_per_sector}/Sector${capNote}${shieldNote}${brakeNote}`;

    const lcAmt = s.liquidcase_alloc_inr !== undefined ? s.liquidcase_alloc_inr : s.liquidcase_cash_inr * 0.7;
    const gbAmt = s.goldbees_alloc_inr !== undefined ? s.goldbees_alloc_inr : s.liquidcase_cash_inr * 0.3;
    document.getElementById('ex-kpi-grid').innerHTML = [
      ['Target Equity Exposure', `${s.actual_equity_pct.toFixed(1)}%`, `${fmtMoney(s.equity_inr)} · Gate ${s.gate_exposure_pct}% / VolCap ${s.vol_cap_exposure_pct}%`, s.dd_brake_active ? 'text-warn' : 'text-up'],
      ['Defensive Reserve (70/30)', fmtMoney(s.liquidcase_cash_inr), `70% LIQUIDCASE ${fmtMoney(lcAmt)} · 30% GOLDBEES ${fmtMoney(gbAmt)}`, ''],
      ['Risk & Diversification', `${s.ex_ante_vol_pct.toFixed(1)}% Vol`, `${s.distinct_sectors} sectors · Avg Pairwise Corr ${s.avg_pairwise_corr.toFixed(2)}`, ''],
      ['Est. Indian STT & Tax', fmtMoney(s.est_total_fees_inr), `Est. 20% STCG: ${fmtMoney(s.est_stcg_tax_inr)} · ${s.trades_count} orders`, s.blackout_warnings > 0 ? 'text-warn' : ''],
    ].map(([l, v, n, c]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value ${c}">${v}</div><div class="kpi-note">${n}</div></div>`).join('');

    const badgeFor = (act) => {
      if (act === 'BUY' || act === 'ADD') return 'badge-up';
      if (act === 'SELL' || act === 'TRIM') return 'badge-down';
      return 'badge-neutral';
    };

    document.getElementById('ex-orders-body').innerHTML = t.orders.map((o) => {
      const deltaStr = o.delta_shares === 0 ? '0' : (o.delta_shares > 0 ? `+${o.delta_shares}` : `${o.delta_shares}`);
      const blackoutBadge = o.earnings_blackout
        ? `<span class="badge badge-warn" style="margin-right:5px;">Results in ${o.days_to_earnings}D</span>`
        : '';
      const delivBadge = (o.deliv_pct !== null && o.deliv_pct !== undefined)
        ? `<span class="badge ${o.deliv_pct >= 52 ? 'badge-up' : (o.deliv_pct < 35 ? 'badge-warn' : 'badge-neutral')}" style="margin-right:5px;">Deliv ${Number(o.deliv_pct).toFixed(0)}%</span>`
        : '';
      const shieldBadge = (o.shield_status && o.shield_status !== 'Clean')
        ? `<span class="badge badge-down" style="margin-right:5px;" title="${escHtml((o.shield_flags || []).join('; '))}">Shield: ${escHtml(o.shield_status)}</span>`
        : '';
      return `<tr>
        <td><span class="badge ${badgeFor(o.action)}">${escHtml(o.action)}</span></td>
        <td style="font-weight:700;"><a href="#" class="text-link" data-ex-sym="${escHtml(o.symbol)}">${escHtml(dispSym(o.symbol))}</a></td>
        <td class="text-muted">${escHtml(o.sector)}</td>
        <td class="num">${o.rank ? '#' + o.rank : '--'}</td>
        <td class="num">${fmtPrice(o.price)}</td>
        <td class="num">${o.target_weight_pct.toFixed(1)}%</td>
        <td class="num"><strong>${o.target_shares}</strong> <span class="text-muted">(${deltaStr})</span></td>
        <td class="num">${o.order_value_inr > 0 ? fmtMoney(o.order_value_inr) : '--'}</td>
        <td class="num"><span class="text-up">${fmtPrice(o.target_price)}</span> / <span class="text-down">${fmtPrice(o.stop_price)}</span></td>
        <td style="font-size:12px;">${blackoutBadge}${delivBadge}${shieldBadge}${escHtml(o.reason)}</td>
      </tr>`;
    }).join('');

    document.querySelectorAll('[data-ex-sym]').forEach((a) => {
      a.addEventListener('click', async (e) => {
        e.preventDefault();
        switchView('chart');
        await loadStockDetail(a.dataset.exSym);
      });
    });
  }

  async function loadMoneySignals() {
    state.moneySignals = await apiFetch('/api/money-signals');
    renderMoneySignals();
  }

  function renderMoneySignals() {
    const ms = state.moneySignals;
    const kpiEl = document.getElementById('money-kpi-grid');
    const delEl = document.getElementById('money-deliv-body');
    const dealEl = document.getElementById('money-deals-body');
    if (!ms || !kpiEl) return;

    const fii = ms.fii_positioning || {};
    const fl = fii.latest;
    const lr = ms.liquidity_ratios || {};
    const fiiPctStr = fl ? `${Number(fl.fii_idx_long_pct).toFixed(1)}% Long` : '--';
    const fiiSub = fl
      ? `Net: ${Number(fl.fii_net_idx_fut).toLocaleString('en-IN')} cts` + (fii.fii_long_pct_5d_chg !== null ? ` (5D ${fmtPct(fii.fii_long_pct_5d_chg, 1)})` : '')
      : 'Click Refresh NSE Money Feeds';
    const diiSub = fl ? `DII Net: ${Number(fl.dii_net_idx_fut).toLocaleString('en-IN')} · Retail: ${Number(fl.client_net_idx_fut).toLocaleString('en-IN')}` : '--';

    kpiEl.innerHTML = [
      ['FII Index Futures', fiiPctStr, fiiSub, fii.tone === 'bullish' ? 'text-up' : (fii.tone === 'bearish' ? 'text-down' : '')],
      ['DII vs Retail F&O', fl ? `${Number(fl.dii_net_idx_fut) >= 0 ? 'Net Long' : 'Net Short'}` : '--', diiSub, ''],
      ['Midcap / NIFTY (3M)', lr.midcap_vs_nifty_63d_pct !== null ? fmtPct(lr.midcap_vs_nifty_63d_pct, 1) : '--', 'India domestic liquidity gauge', (lr.midcap_vs_nifty_63d_pct || 0) >= 0 ? 'text-up' : 'text-down'],
      ['BankNIFTY / NIFTY (3M)', lr.bank_vs_nifty_63d_pct !== null ? fmtPct(lr.bank_vs_nifty_63d_pct, 1) : '--', 'India credit-cycle gauge', (lr.bank_vs_nifty_63d_pct || 0) >= 0 ? 'text-up' : 'text-down'],
    ].map(([l, v, n, c]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value ${c}">${v}</div><div class="kpi-note">${n}</div></div>`).join('');

    const topDel = ms.top_accumulation || [];
    delEl.innerHTML = topDel.length ? topDel.map((d) => `
      <tr>
        <td style="font-weight:700;">${escHtml(dispSym(d.symbol))}</td>
        <td class="num ${d.deliv_pct >= 55 ? 'text-up' : ''}" style="font-weight:600;">${d.deliv_pct.toFixed(1)}%</td>
        <td class="num ${d.deliv_surge_pct >= 0 ? 'text-up' : 'text-down'}">${fmtPct(d.deliv_surge_pct, 1)}</td>
        <td class="num">${fmtMoney(d.avg_trade_val_inr)}</td>
        <td><span class="badge ${d.tone === 'bullish' ? 'badge-up' : (d.tone === 'warning' ? 'badge-warn' : 'badge-neutral')}">${escHtml(d.tag)}</span></td>
      </tr>
    `).join('') : `<tr><td colspan="5" class="text-muted" style="text-align:center;">Click Refresh NSE Money Feeds to pull official NSE delivery data.</td></tr>`;

    const deals = (ms.universe_bulk_deals && ms.universe_bulk_deals.length) ? ms.universe_bulk_deals : (ms.top_nse_deals || []);
    dealEl.innerHTML = deals.length ? deals.map((d) => `
      <tr>
        <td class="text-muted">${escHtml(d.date)}</td>
        <td style="font-weight:600;">${escHtml(dispSym(d.symbol))}</td>
        <td><span class="badge ${d.action === 'BUY' ? 'badge-up' : 'badge-down'}">${escHtml(d.deal_type)} ${escHtml(d.action)}</span></td>
        <td style="font-size:12px;">${escHtml(d.client_name.slice(0, 32))}</td>
        <td class="num">₹${Number(d.value_cr).toFixed(1)} Cr</td>
      </tr>
    `).join('') : `<tr><td colspan="5" class="text-muted" style="text-align:center;">No recent NSE bulk/block deals synced yet.</td></tr>`;
  }

  function renderPaperLedger() {
    const p = state.paperLedger;
    const kpiEl = document.getElementById('paper-kpi-grid');
    const posEl = document.getElementById('paper-pos-body');
    const trEl = document.getElementById('paper-trades-body');

    if (!p || !p.initialized) {
      kpiEl.innerHTML = `<div class="kpi-card" style="grid-column: 1 / -1;"><div class="kpi-label">Paper Ledger Not Started</div><div class="kpi-note">Click <strong>Commit to Paper Ledger</strong> above to lock today's live V6 NSE fills in SQLite and begin zero-survivorship-bias forward tracking against NIFTY 50.</div></div>`;
      posEl.innerHTML = `<tr><td colspan="6" class="text-muted" style="text-align:center;">No open paper positions yet.</td></tr>`;
      trEl.innerHTML = `<tr><td colspan="6" class="text-muted" style="text-align:center;">No paper trades committed yet.</td></tr>`;
      return;
    }

    const a = p.account;
    kpiEl.innerHTML = [
      ['Paper Account NAV', fmtMoney(a.total_nav_inr), `Initial: ${fmtMoney(a.initial_capital_inr)} (${a.risk_profile.toUpperCase()} · Top-${a.top_k})`, ''],
      ['Forward Return vs NIFTY', fmtPct(a.total_return_pct), `NIFTY: ${fmtPct(a.nifty_return_pct)} · Excess: ${fmtPct(a.excess_vs_nifty_pct)}`, a.total_return_pct >= 0 ? 'text-up' : 'text-down'],
      ['LIQUIDCASE Cash', fmtMoney(a.cash_liquidcase_inr), `Equity: ${a.equity_weight_pct}% · Cash Yield: +${fmtMoney(a.accrued_cash_interest_inr)}`, ''],
      ['Realized / Fees / STCG', fmtMoney(a.realized_pnl_inr), `STT+Fees: ${fmtMoney(a.total_fees_inr)} · Est. 20% STCG: ${fmtMoney(a.est_stcg_tax_inr)}`, ''],
    ].map(([l, v, n, c]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value ${c}">${v}</div><div class="kpi-note">${n}</div></div>`).join('');

    posEl.innerHTML = p.positions.length ? p.positions.map((pos) => `
      <tr>
        <td style="font-weight:700;">${escHtml(dispSym(pos.symbol))}</td>
        <td class="num">${pos.shares}</td>
        <td class="num">${fmtPrice(pos.avg_cost)}</td>
        <td class="num">${fmtPrice(pos.current_price)}</td>
        <td class="num">${fmtMoney(pos.market_value_inr)}</td>
        <td class="num ${pos.unrealized_pnl_inr >= 0 ? 'text-up' : 'text-down'}">${fmtPct(pos.unrealized_pnl_pct)}</td>
      </tr>
    `).join('') : `<tr><td colspan="6" class="text-muted" style="text-align:center;">100% LIQUIDCASE cash.</td></tr>`;

    trEl.innerHTML = p.trades.length ? p.trades.map((tr) => `
      <tr>
        <td class="text-muted">${escHtml(tr.trade_date)}</td>
        <td><span class="badge ${tr.action === 'BUY' || tr.action === 'ADD' ? 'badge-up' : 'badge-down'}">${escHtml(tr.action)}</span></td>
        <td style="font-weight:600;">${escHtml(dispSym(tr.symbol))}</td>
        <td class="num">${tr.shares}</td>
        <td class="num">${fmtPrice(tr.price)}</td>
        <td class="num text-muted">${fmtPrice(tr.fee_inr)}</td>
      </tr>
    `).join('') : `<tr><td colspan="6" class="text-muted" style="text-align:center;">No trades logged.</td></tr>`;
  }

  // =========================================================================
  // Navigation & Bindings
  // =========================================================================
  function switchView(view) {
    state.view = view;
    document.querySelectorAll('.nav-tab').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
    document.querySelectorAll('.view-section').forEach((s) => s.classList.toggle('active', s.id === `view-${view}`));

    if (view === 'chart') {
      if (!state.stockDetail || state.stockDetail.ticker.symbol !== state.selectedSymbol) {
        loadStockDetail(state.selectedSymbol).catch((e) => showToast(e.message));
      } else {
        requestAnimationFrame(() => { drawDetailMainChart(); drawDetailOscChart(); });
      }
    } else if (view === 'model') {
      loadModelMetrics().catch((e) => showToast(e.message));
    } else if (view === 'stress') {
      if (!state.stressData) loadStressTests().catch((e) => showToast(e.message));
      else requestAnimationFrame(drawStressEquityCanvas);
      loadSingleBacktest().catch(() => {});
    } else if (view === 'execution') {
      Promise.all([loadExecutionTicket(), loadPaperLedger(), loadMoneySignals()]).catch((e) => showToast(e.message));
    } else if (view === 'journal') {
      loadJournal().catch((e) => showToast(e.message));
    } else if (view === 'alerts') {
      loadAlerts().catch((e) => showToast(e.message));
    }
  }

  function syncWeightLabels() {
    document.getElementById('w-ml').value = state.weights.ml;
    document.getElementById('w-mom').value = state.weights.mom;
    document.getElementById('w-val').value = state.weights.val;
    document.getElementById('w-qual').value = state.weights.qual;
    document.getElementById('w-sent').value = state.weights.sent;
    const sum = Object.values(state.weights).reduce((a, b) => a + b, 0) || 1;
    document.getElementById('lbl-w-ml').textContent = `${Math.round((state.weights.ml / sum) * 100)}%`;
    document.getElementById('lbl-w-mom').textContent = `${Math.round((state.weights.mom / sum) * 100)}%`;
    document.getElementById('lbl-w-val').textContent = `${Math.round((state.weights.val / sum) * 100)}%`;
    document.getElementById('lbl-w-qual').textContent = `${Math.round((state.weights.qual / sum) * 100)}%`;
    document.getElementById('lbl-w-sent').textContent = `${Math.round((state.weights.sent / sum) * 100)}%`;
  }

  let weightDebounce = null;
  function bindEvents() {
    document.querySelectorAll('.nav-tab').forEach((b) => b.addEventListener('click', () => switchView(b.dataset.view)));

    document.getElementById('btn-theme').addEventListener('click', () => {
      const root = document.documentElement;
      const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      document.getElementById('btn-theme').textContent = next === 'dark' ? 'Light Mode' : 'Dark Mode';
      drawDetailMainChart(); drawDetailOscChart(); drawStressEquityCanvas();
    });

    document.getElementById('add-ticker-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const inp = document.getElementById('add-ticker-input');
      const sym = inp.value.trim().toUpperCase();
      if (!sym) return;
      const btn = document.getElementById('btn-add-ticker');
      btn.disabled = true; btn.textContent = 'Fetching...';
      try {
        const res = await apiFetch('/api/tickers/add', { method: 'POST', body: JSON.stringify({ symbol: sym }) });
        inp.value = '';
        showToast(`Fetched ${dispSym(res.synced.symbol)} (${res.synced.bars} bars, ${res.synced.news_count} news) & updated India ML.`);
        await loadShortlist();
        switchView('chart');
        await loadStockDetail(res.synced.symbol);
      } catch (err) {
        showToast(err.message, 4500);
      } finally {
        btn.disabled = false; btn.textContent = '+ Fetch NSE';
      }
    });

    document.getElementById('btn-sync-all').addEventListener('click', async () => {
      const btn = document.getElementById('btn-sync-all');
      btn.disabled = true; btn.textContent = 'Syncing NSE...';
      try {
        const res = await apiFetch('/api/sync', { method: 'POST' });
        showToast(`Synced ${res.sync.synced_count} NSE stocks, retrained ML (OOS Hit ${res.ml.hit_rate}%), ${res.alerts_generated} alerts.`);
        state.stressData = null;
        await loadShortlist();
      } catch (err) {
        showToast(err.message, 4500);
      } finally {
        btn.disabled = false; btn.textContent = 'Sync & Retrain';
      }
    });

    document.querySelectorAll('#preset-group button').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.weights = { ...PRESETS[btn.dataset.preset] };
        document.querySelectorAll('#preset-group button').forEach((b) => b.classList.toggle('active', b === btn));
        syncWeightLabels();
        loadShortlist().catch((e) => showToast(e.message));
      });
    });

    [['w-ml', 'ml'], ['w-mom', 'mom'], ['w-val', 'val'], ['w-qual', 'qual'], ['w-sent', 'sent']].forEach(([id, key]) => {
      document.getElementById(id).addEventListener('input', (e) => {
        state.weights[key] = parseInt(e.target.value, 10);
        document.querySelectorAll('#preset-group button').forEach((b) => b.classList.remove('active'));
        syncWeightLabels();
        clearTimeout(weightDebounce);
        weightDebounce = setTimeout(() => loadShortlist().catch((err) => showToast(err.message)), 180);
      });
    });

    document.getElementById('filter-sector').addEventListener('change', () => loadShortlist().catch((e) => showToast(e.message)));
    document.getElementById('filter-setup').addEventListener('change', () => loadShortlist().catch((e) => showToast(e.message)));
    document.getElementById('filter-search').addEventListener('input', (e) => { state.searchFilter = e.target.value; renderShortlistView(); });

    document.getElementById('chart-symbol-select').addEventListener('change', (e) => loadStockDetail(e.target.value).catch((err) => showToast(err.message)));
    document.getElementById('btn-det-star').addEventListener('click', async () => {
      if (!state.stockDetail) return;
      const sym = state.stockDetail.ticker.symbol;
      const next = !state.stockDetail.ticker.starred;
      await apiFetch(`/api/tickers/${encodeURIComponent(sym)}/star`, { method: 'POST', body: JSON.stringify({ starred: next }) });
      state.stockDetail.ticker.starred = next;
      renderStockDetailView();
      loadShortlist().catch(() => {});
    });

    document.querySelectorAll('#tf-group button').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.timeframe = btn.dataset.tf; state.hoverIdx = null;
        document.querySelectorAll('#tf-group button').forEach((b) => b.classList.toggle('active', b === btn));
        drawDetailMainChart(); drawDetailOscChart();
      });
    });
    document.querySelectorAll('[data-ov]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const k = btn.dataset.ov;
        state.overlays[k] = !state.overlays[k];
        btn.classList.toggle('active', state.overlays[k]);
        drawDetailMainChart(); drawDetailOscChart();
      });
    });
    document.getElementById('osc-select').addEventListener('change', (e) => { state.oscillator = e.target.value; drawDetailOscChart(); });

    const onMove = (e) => {
      if (!state.stockDetail) return;
      const { bars } = getVisibleSlice();
      const ml = state.stockDetail.ml_prediction;
      const coneExtra = (state.overlays.cone && ml && ml.cone && ml.cone.length > 1) ? Math.max(18, Math.round(bars.length * 0.24)) : 0;
      const rect = e.currentTarget.getBoundingClientRect();
      const slot = Math.floor(((e.clientX - rect.left - 10) / Math.max(1, rect.width - 90)) * (bars.length + coneExtra));
      state.hoverIdx = (slot >= 0 && slot < bars.length) ? slot : null;
      drawDetailMainChart(); drawDetailOscChart();
    };
    ['main-chart-canvas', 'osc-chart-canvas'].forEach((id) => {
      const c = document.getElementById(id);
      c.addEventListener('mousemove', onMove);
      c.addEventListener('mouseleave', () => { state.hoverIdx = null; drawDetailMainChart(); drawDetailOscChart(); });
    });

    document.getElementById('btn-run-stress').addEventListener('click', () => loadStressTests().catch((e) => showToast(e.message)));
    document.getElementById('btn-run-bt').addEventListener('click', () => loadSingleBacktest().catch((e) => showToast(e.message)));

    // Execution & Paper-Trading Ledger bindings
    document.getElementById('btn-gen-ticket').addEventListener('click', () => loadExecutionTicket().catch((e) => showToast(e.message)));
    ['ex-risk-profile', 'ex-topk', 'ex-skip-blackout', 'ex-smart-shield', 'ex-use-paper'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', () => loadExecutionTicket().catch((e) => showToast(e.message)));
    });
    document.getElementById('btn-commit-paper').addEventListener('click', async () => {
      const btn = document.getElementById('btn-commit-paper');
      btn.disabled = true; btn.textContent = 'Committing...';
      try {
        const shieldEl = document.getElementById('ex-smart-shield');
        const res = await apiFetch('/api/execution/paper/commit', {
          method: 'POST',
          body: JSON.stringify({
            capital_inr: Math.max(25000, Number(document.getElementById('ex-capital').value) || 1000000),
            top_k: Number(document.getElementById('ex-topk').value) || 10,
            risk_profile: document.getElementById('ex-risk-profile').value || 'balanced',
            skip_earnings_blackout: document.getElementById('ex-skip-blackout').checked,
            enforce_smart_money_shield: shieldEl ? shieldEl.checked : true,
          }),
        });
        state.paperLedger = res.ledger;
        renderPaperLedger();
        await loadExecutionTicket();
        showToast('Committed live V6 rebalance to SQLite Paper Ledger.');
      } catch (err) {
        showToast(err.message, 4500);
      } finally {
        btn.disabled = false; btn.textContent = 'Commit to Paper Ledger';
      }
    });
    document.getElementById('btn-reset-paper').addEventListener('click', async () => {
      if (!window.confirm('Reset the forward-proof paper-trading ledger and erase all recorded paper positions & trades?')) return;
      const res = await apiFetch('/api/execution/paper/reset', {
        method: 'POST',
        body: JSON.stringify({ confirm: true }),
      });
      state.paperLedger = res.ledger;
      renderPaperLedger();
      await loadExecutionTicket();
      showToast('Paper-trading account reset.');
    });
    const btnSyncMoney = document.getElementById('btn-sync-money');
    if (btnSyncMoney) {
      btnSyncMoney.addEventListener('click', async () => {
        btnSyncMoney.disabled = true; btnSyncMoney.textContent = 'Syncing NSE...';
        try {
          const res = await apiFetch('/api/money-signals/sync', { method: 'POST' });
          state.moneySignals = res.data;
          renderMoneySignals();
          await loadExecutionTicket();
          showToast(`Synced ${res.synced.delivery_rows} NSE delivery rows & ${res.synced.fii_oi_days}D FII/DII OI.`);
        } catch (err) {
          showToast(err.message, 4500);
        } finally {
          btnSyncMoney.disabled = false; btnSyncMoney.textContent = 'Refresh NSE Money Feeds';
        }
      });
    }

    document.getElementById('btn-log-journal').addEventListener('click', async () => {
      state.journalData = await apiFetch('/api/journal/log', { method: 'POST' });
      renderJournalView();
      showToast('Logged & graded NSE prediction journal snapshot.');
    });
    document.getElementById('btn-scan-alerts').addEventListener('click', async () => {
      await apiFetch('/api/alerts/scan', { method: 'POST' });
      await loadAlerts();
      showToast('NSE catalyst & signal alert scan complete.');
    });

    window.addEventListener('resize', () => { drawDetailMainChart(); drawDetailOscChart(); drawStressEquityCanvas(); });
  }

  syncWeightLabels();
  bindEvents();
  loadShortlist().catch((err) => showToast(`Startup load error: ${err.message}`, 5000));
})();
