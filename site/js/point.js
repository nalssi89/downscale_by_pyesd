/* 지점 상세: 기간별 3분위 확률, 예측범위 분포, 통계표 (PyESD 상세화) */
(function () {
  'use strict';
  const K = window.KCO;
  const $ = (id) => document.getElementById(id);
  const q = new URLSearchParams(location.search);
  const st = { i: +(q.get('i') || 0), s: q.get('s'), f: q.get('f'), p: q.get('p') || '1m', v: q.get('v') || 't2m', period: q.get('period'), ts: 'daily' };
  const isStn = () => data && data.kind === 'station';
  const isFine = () => data && data.kind === 'fine';
  if (q.get('embed')) document.body.classList.add('embed');
  let manifest, grid, data;

  const EC_LABEL = {
    temperature_2m_anomaly_gt0: '기온 편차 > 0°C', temperature_2m_anomaly_gt1: '기온 편차 > +1°C', temperature_2m_anomaly_gt2: '기온 편차 > +2°C',
    temperature_2m_anomaly_ltm1: '기온 편차 < −1°C', temperature_2m_anomaly_ltm2: '기온 편차 < −2°C',
    precipitation_anomaly_gt0: '강수 편차 > 0 mm', precipitation_anomaly_gt10: '강수 편차 > 10 mm', precipitation_anomaly_gt20: '강수 편차 > 20 mm',
    temperature_2m_efi: '기온 EFI (−1~1)', precipitation_efi: '강수 EFI (−1~1)',
  };

  async function init() {
    try {
      [manifest, grid] = await Promise.all([K.getJSON(K.DATA + 'manifest.json'), K.getJSON('data/grid.json')]);
      data = await K.getJSON(K.DATA + (st.s ? `stations/${st.s}.json` : st.f != null ? `fine/${st.f}.json` : `points/${st.i}.json`));
    } catch (e) {
      $('ptTitle').textContent = '자료를 불러올 수 없습니다';
      console.error(e);
      return;
    }
    if (!data[st.p]) st.p = Object.keys(manifest.products).find((k) => data[k]);
    if (!manifest.variables[st.v] || (data.missing || []).includes(st.v)) st.v = 't2m';
    try { const t = localStorage.getItem('kco-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* 무시 */ }
    render();
    let rt;
    addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(renderCharts, 120); });
  }

  function seg(elm, items, cur, pick) {
    elm.innerHTML = '';
    for (const [val, label] of items) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = val === cur ? 'on' : '';
      b.setAttribute('aria-pressed', String(val === cur));
      b.innerHTML = label;
      b.onclick = () => pick(val);
      elm.appendChild(b);
    }
  }

  function render() {
    const P = manifest.products[st.p];
    const V = manifest.variables[st.v];
    const city = K.nearestCity(data.lat, data.lon);
    if (isStn()) {
      document.title = `ASOS ${data.name} · ${P.name}`;
      $('ptTitle').textContent = `ASOS ${data.name}(${data.stn})` + (data.ht != null ? ` · 해발 ${Math.round(data.ht)} m` : '');
    } else if (isFine()) {
      document.title = `${city || ''} 20 km 상세화 · ${P.name}`;
      $('ptTitle').textContent = `${city ? city + ' 부근 · ' : ''}${data.lat.toFixed(1)}°N ${data.lon.toFixed(1)}°E` +
        (data.ht != null ? ` · 평균 해발 ${Math.round(data.ht)} m` : '');
    } else {
      document.title = `${city || ''} ${data.lat}°N ${data.lon}°E · ${P.name}`;
      $('ptTitle').textContent = `${city ? city + ' 부근 · ' : ''}${data.lat.toFixed(2)}°N ${data.lon.toFixed(2)}°E · 0.25° 격자`;
    }
    $('ptSub').textContent = `${P.name} · ${P.model} · 발표 ${manifest.issued.slice(0, 10)}${manifest.demo ? ' · DEMO 합성자료' : ''} · ` +
      (isStn() ? 'ASOS 관측(1991–2020)으로 학습한 지점 PyESD 상세화' : '0.25° 격자 PyESD 상세화 (ERA5 1991–2020 학습)');
    seg($('pProductSeg'), Object.entries(manifest.products).map(([k, p]) => [k, p.name]), st.p, (v) => { st.p = v; sync(); render(); });
    const tabs = $('varTabs');
    tabs.innerHTML = '';
    for (const [k, vv] of Object.entries(manifest.variables)) {
      if ((data.missing || []).includes(k)) continue; // 관측하지 않는 요소(일사 등)
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', String(k === st.v));
      b.innerHTML = `<span aria-hidden="true">${K.VAR_META[k].icon}</span> ${vv.name} <small>${vv.unit}</small>`;
      b.onclick = () => { st.v = k; sync(); render(); };
      tabs.appendChild(b);
    }
    renderTercile();
    renderCharts();
    renderTable();
  }

  function sync() {
    const u = new URLSearchParams(location.search);
    u.set('p', st.p);
    u.set('v', st.v);
    history.replaceState(null, '', '?' + u.toString());
  }

  const periodsOf = () => manifest.products[st.p].periods;
  const rowOf = (pid) => data[st.p].periods[st.v][pid];

  // ---------------------------------------------------------------- 3분위 막대
  function renderTercile() {
    const P = K.POLES[st.v];
    const lowC = P.low[2], highC = P.high[2];
    const box = $('tercileBars');
    $('tercHint').textContent = `평년 확률은 각 33% · ${P.lowName}/비슷/${P.highName}`;
    box.innerHTML = periodsOf().map((p) => {
      const r = rowOf(p.id);
      if (!r) return '';
      const parts = [[r.below, lowC, P.lowName, '#fff'], [r.near, K.NEAR, '비슷', '#0b0b0b'], [r.above, highC, P.highName, '#fff']];
      const max = Math.max(r.below, r.near, r.above);
      const dom = max < 40 ? '뚜렷한 우세 범주 없음' : r.above === max ? `${P.highName} 우세` : r.below === max ? `${P.lowName} 우세` : '평년과 비슷 우세';
      return `<div class="tbar ${p.id === st.period ? 'cur' : ''}">
        <div class="tl"><b>${p.label}</b><small>${K.fmtDate(p.start)}–${K.fmtDate(p.end)}</small></div>
        <div class="tb" role="img" aria-label="${p.label}: ${P.lowName} ${r.below}%, 비슷 ${r.near}%, ${P.highName} ${r.above}%">
          ${parts.map(([v, c, n, ink]) => `<span style="flex:${Math.max(v, 0.5)};background:${c};color:${ink}" title="${n} ${v}%">${v >= 12 ? v + '%' : ''}</span>`).join('')}
          <i class="third" style="left:33.33%"></i><i class="third" style="left:66.67%"></i>
        </div>
        <div class="tr"><b>${dom}</b><small>중앙값 초과 ${r.gt_median}%</small></div>
      </div>`;
    }).join('');
  }

  // ---------------------------------------------------------------- 차트 (기간별 분포; 일별 시계열은 월·주 통계 상세화라 없음)
  function renderCharts() {
    const V = manifest.variables[st.v];
    const dg = K.VAR_META[st.v].fmt;
    const P = K.POLES[st.v];
    K.charts.distribution($('distChart'), periodsOf().map((p) => {
      const r = rowOf(p.id);
      return { label: p.label, sub: `${K.fmtDate(p.start)}–${K.fmtDate(p.end)}`, members: [], q: r.q, bounds: r.bounds, mean: r.mean, clim: r.clim, row: r };
    }), {
      low: P.low[2], high: P.high[2], near: '#a9a7a0', unit: `${V.unit}${V.agg === 'sum' ? ' (기간 누적)' : ' (기간 평균)'}`, zeroBase: V.agg === 'sum',
      tip: (r) => `<div class="th">${r.label} ${r.sub}</div>
        <div class="tr"><i></i><span>앙상블 평균</span><b>${K.fmt(r.mean, dg)}</b></div>
        <div class="tr"><i></i><span>관측 평년값</span><b>${K.fmt(r.clim, dg)}</b></div>
        <div class="tr"><i></i><span>예측범위 10–90%</span><b>${K.fmt(r.q[0], dg)} ~ ${K.fmt(r.q[4], dg)}</b></div>
        <div class="tr"><i style="background:${P.low[2]}"></i><span>평년 하위 1/3 경계</span><b>${K.fmt(r.bounds[1], dg)}</b></div>
        <div class="tr"><i style="background:${P.high[2]}"></i><span>평년 상위 1/3 경계</span><b>${K.fmt(r.bounds[3], dg)}</b></div>
        <div class="tr"><i></i><span>${P.lowName}/비슷/${P.highName}</span><b>${r.row.below}/${r.row.near}/${r.row.above}%</b></div>`,
    });
  }

  // ---------------------------------------------------------------- 표
  function tableRows() {
    const V = manifest.variables[st.v];
    const P = K.POLES[st.v];
    const dg = K.VAR_META[st.v].fmt;
    const per = periodsOf();
    const R = per.map((p) => rowOf(p.id));
    const f = (v, d = dg) => K.fmt(v, d);
    const rows = [
      ['group', '값'],
      ['앙상블 평균', R.map((r) => f(r.mean))],
      ['관측 평년값 (1991–2020)', R.map((r) => f(r.clim))],
      [V.mode === 'add' ? '평년편차' : '평년편차 (평년비)', R.map((r) => (r.anom > 0 ? '+' : '') + f(r.anom) + (V.mode === 'mul' ? ` (${f(r.anom_pct, 0)}%)` : ''))],
      ['앙상블 스프레드(σ)', R.map((r) => f(r.spread))],
      ['예측범위 10%', R.map((r) => f(r.q[0]))],
      ['예측범위 25%', R.map((r) => f(r.q[1]))],
      ['예측 중앙값', R.map((r) => f(r.q[2]))],
      ['예측범위 75%', R.map((r) => f(r.q[3]))],
      ['예측범위 90%', R.map((r) => f(r.q[4]))],
      ['전이함수 교차검증 상관', R.map((r) => (r.r_cv == null ? '–' : f(r.r_cv, 2)))],
      ['group', '평년 분포 경계'],
      ['하위 10% 경계', R.map((r) => f(r.bounds[0]))],
      ['하위 1/3 경계', R.map((r) => f(r.bounds[1]))],
      ['평년 중앙값', R.map((r) => f(r.bounds[2]))],
      ['상위 1/3 경계', R.map((r) => f(r.bounds[3]))],
      ['상위 10% 경계', R.map((r) => f(r.bounds[4]))],
      ['group', '확률 (%)'],
      [`${P.lowName} (하위 1/3)`, R.map((r) => r.below), 'low'],
      ['평년과 비슷', R.map((r) => r.near), 'near'],
      [`${P.highName} (상위 1/3)`, R.map((r) => r.above), 'high'],
      ['평년 중앙값 초과', R.map((r) => r.gt_median), 'p'],
      [`매우 ${P.lowName} (하위 10%)`, R.map((r) => r.lt_p10), 'p'],
      [`매우 ${P.highName} (상위 10%)`, R.map((r) => r.gt_p90), 'p'],
      ['group', '임계값별 발생확률 (%)'],
    ];
    const keys = [...new Set(R.flatMap((r) => Object.keys(r.thr)))];
    for (const k of keys) rows.push([K.thresholdLabel(k, V), R.map((r) => (r.thr[k] == null ? '–' : r.thr[k])), 'p']);
    return { per, rows };
  }

  function probCell(v, kind) {
    if (v === '–' || v == null) return '<td>–</td>';
    const P = K.POLES[st.v];
    const c = kind === 'low' ? P.low[2] : kind === 'high' ? P.high[2] : kind === 'near' ? '#898781' : 'var(--accent)';
    return `<td class="pc"><span class="pbar" style="width:${Math.min(v, 100)}%;background:${c}"></span><b>${v}</b></td>`;
  }

  function renderTable() {
    const { per, rows } = tableRows();
    const V = manifest.variables[st.v];
    const head = `<thead><tr><th>${V.name} (${V.unit})</th>${per.map((p) => `<th class="${p.id === st.period ? 'cur' : ''}">${p.label}<small>${K.fmtDate(p.start)}–${K.fmtDate(p.end)}</small></th>`).join('')}</tr></thead>`;
    const body = rows.map(([lab, vals, kind]) => {
      if (lab === 'group') return `<tr class="grp"><th colspan="${per.length + 1}">${vals}</th></tr>`;
      return `<tr><th>${lab}</th>${vals.map((v) => (kind ? probCell(v, kind) : `<td>${v}</td>`)).join('')}</tr>`;
    }).join('');
    $('statTable').innerHTML = head + '<tbody>' + body + '</tbody>';
    $('tableNote').innerHTML = `${V.agg === 'sum' ? '기간 누적' : '기간 평균'} 기준. 확률과 예측범위는 51개 앙상블 멤버를 각각 전이함수로 투영한 값에 교차검증 잔차 분포를 더한 정규혼합에서 계산. ` +
      (isStn() ? 'ASOS 지점 관측(1991–2020)으로 학습한 지점별 PyESD 전이함수(지역평균 인자·기후지수·EOF → ARD/능형회귀). 평년값·분포 경계는 지점 관측에서 직접 계산.'
        : 'ERA5 0.25° 격자 1991–2020 으로 학습한 격자별 PyESD 전이함수. 평년값·분포 경계는 같은 격자 ERA5 30년 분포.');
    $('csvBtn').onclick = downloadCSV;
  }

  function downloadCSV() {
    const { per, rows } = tableRows();
    const V = manifest.variables[st.v];
    const strip = (s) => String(s).replace(/<[^>]+>/g, '');
    const lines = [[`${V.name}(${V.unit})`, ...per.map((p) => `${p.label} ${p.start}~${p.end}`)].join(',')];
    for (const [lab, vals] of rows) if (lab !== 'group') lines.push([lab, ...vals.map(strip)].map((x) => `"${x}"`).join(','));
    const blob = new Blob(['﻿' + lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `outlook_${st.p}_${st.v}_${data.lat}N_${data.lon}E.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  init();
})();
