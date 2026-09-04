import argparse
import json
import os
import sys
import urllib.parse
import webbrowser
from html import escape as esc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml
from user_test import (SMS_TOPUP_BDT, allin_cost, covers, load_catalogue,
                       period_price, recommend, reference_rates,
                       repeats_needed, topup)

CARDS = 3
MODEL_ORDER = ["Random forest", "Decision tree", "Polynomial regression"]
VALIDITY_CHOICES = [1, 3, 7, 15, 30, 60, 90]
PRESETS = [
    ("Light", 2, 100, 20, 30),
    ("Balanced", 8, 300, 50, 30),
    ("Heavy data", 30, 100, 0, 30),
    ("Talker", 3, 1000, 100, 30),
]

STYLE = """
:root {
  --bg: #f4f5f3; --card: #ffffff; --raise: #ffffff;
  --ink: #14161a; --ink-2: #52565e; --ink-3: #8b9098;
  --line: #e4e5e1; --line-2: #d3d5d0;
  --accent: #1f6feb; --accent-ink: #ffffff; --accent-soft: #e8f0fe;
  --warn: #8a5a00; --warn-soft: #fdf4e3; --good: #1baf7a;
  --shadow: 0 1px 2px rgba(20,22,26,.05), 0 8px 24px rgba(20,22,26,.06);
  --radius: 14px;
}
[data-theme="dark"] {
  --bg: #0f1113; --card: #17191c; --raise: #1d2024;
  --ink: #f2f3f5; --ink-2: #b4b9c0; --ink-3: #7d838c;
  --line: #26292e; --line-2: #32363c;
  --accent: #4d8ef7; --accent-ink: #0b1220; --accent-soft: #1a2432;
  --warn: #e0b464; --warn-soft: #251f14; --good: #4fd1a0;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 54rem; margin: 0 auto; padding: 1.5rem 1.1rem 5rem; }
header { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1.4rem; }
h1 { font-size: 1.35rem; letter-spacing: -.015em; margin: 0 0 .3rem; }
.sub { color: var(--ink-2); font-size: .87rem; margin: 0; max-width: 42rem; }
.sub b { color: var(--ink); font-weight: 600; }
.toggle {
  margin-left: auto; flex: none; width: 2.3rem; height: 2.3rem; cursor: pointer;
  border: 1px solid var(--line); border-radius: 50%; background: var(--card);
  color: var(--ink-2); font-size: 1rem; line-height: 1; transition: .18s;
}
.toggle:hover { color: var(--ink); border-color: var(--line-2); transform: translateY(-1px); }
.panel {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 1.15rem 1.25rem 1.3rem; margin-bottom: 1.4rem;
}
.row + .row { margin-top: 1.05rem; }
.lbl {
  display: flex; align-items: baseline; gap: .5rem; margin-bottom: .45rem;
  font-size: .72rem; letter-spacing: .07em; text-transform: uppercase; color: var(--ink-3);
}
.lbl .val { margin-left: auto; font-size: .95rem; letter-spacing: 0;
            text-transform: none; color: var(--ink); font-weight: 650;
            font-variant-numeric: tabular-nums; }
.chips { display: flex; flex-wrap: wrap; gap: .4rem; }
.chip {
  font: inherit; font-size: .86rem; cursor: pointer; padding: .38rem .8rem;
  border-radius: 999px; border: 1px solid var(--line); background: var(--card);
  color: var(--ink-2); transition: .16s;
}
.chip:hover { border-color: var(--accent); color: var(--ink); }
.chip[aria-pressed="true"] {
  background: var(--accent); border-color: var(--accent); color: var(--accent-ink);
  font-weight: 600;
}
.chip.ghost[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent);
                                   border-color: transparent; }
.sliders { display: grid; gap: 1.05rem; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); }
input[type=range] {
  -webkit-appearance: none; appearance: none; width: 100%; height: 22px;
  background: transparent; cursor: pointer;
}
input[type=range]::-webkit-slider-runnable-track {
  height: 5px; border-radius: 999px;
  background: linear-gradient(to right, var(--accent) var(--pct,0%), var(--line) var(--pct,0%));
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 17px; height: 17px; margin-top: -6px;
  border-radius: 50%; background: var(--card); border: 2px solid var(--accent);
  box-shadow: 0 1px 3px rgba(0,0,0,.18); transition: transform .14s;
}
input[type=range]:active::-webkit-slider-thumb { transform: scale(1.15); }
input[type=range]:focus-visible { outline: 2px solid var(--accent); outline-offset: 4px; border-radius: 8px; }
input[type=number], select {
  font: inherit; padding: .4rem .55rem; color: var(--ink); background: var(--bg);
  border: 1px solid var(--line); border-radius: 9px; width: 6.2rem;
}
.hint { color: var(--ink-3); font-size: .78rem; margin: .9rem 0 0; }
button.go {
  font: inherit; font-weight: 600; cursor: pointer; margin-top: 1rem;
  padding: .55rem 1.3rem; border: 0; border-radius: 10px;
  background: var(--accent); color: var(--accent-ink);
}
.results { transition: opacity .18s; }
.results.busy { opacity: .45; }
.request { color: var(--ink-2); font-size: .87rem; margin: 0 0 .8rem; }
.request b { color: var(--ink); font-weight: 600; }
.pack {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1rem 1.15rem 1.1rem; margin-bottom: .8rem; box-shadow: var(--shadow);
  animation: rise .32s cubic-bezier(.2,.7,.3,1) backwards; transition: .18s;
}
.pack:hover { transform: translateY(-2px); border-color: var(--line-2); }
.pack:nth-child(2) { animation-delay: .05s; }
.pack:nth-child(3) { animation-delay: .1s; }
@keyframes rise { from { opacity: 0; transform: translateY(8px); } }
.pack.best { border-color: var(--accent); }
.tag { display: flex; align-items: center; gap: .5rem; font-size: .68rem;
       letter-spacing: .09em; text-transform: uppercase; color: var(--ink-3); }
.pack.best .tag { color: var(--accent); font-weight: 700; }
.pill { border-radius: 999px; padding: .1rem .5rem; background: var(--accent-soft);
        color: var(--accent); letter-spacing: .05em; font-weight: 600; }
.name { font-size: 1.08rem; font-weight: 650; letter-spacing: -.01em; margin: .2rem 0 .75rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(5.2rem, 1fr));
        gap: .65rem .9rem; margin-bottom: .85rem; }
.grid span { display: block; font-size: .68rem; letter-spacing: .05em;
             text-transform: uppercase; color: var(--ink-3); margin-bottom: .1rem; }
.grid b { font-weight: 640; font-variant-numeric: tabular-nums; }
.bar { height: 6px; border-radius: 999px; background: var(--line); overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 999px; background: var(--accent);
         width: 0; transition: width .5s cubic-bezier(.2,.8,.3,1); }
.score { font-size: .7rem; letter-spacing: .07em; text-transform: uppercase;
         color: var(--ink-3); margin-top: .35rem; }
.note { border-radius: 10px; padding: .6rem .75rem; font-size: .86rem;
        background: var(--warn-soft); color: var(--warn); margin-top: .7rem; }
.note.plain { background: var(--accent-soft); color: var(--ink-2); }
.note ul { margin: .3rem 0 .4rem; padding-left: 1.1rem; }
.note b { font-variant-numeric: tabular-nums; }
.compare { margin-top: 1.8rem; }
.compare h2 { font-size: .95rem; margin: 0 0 .2rem; }
.compare p { color: var(--ink-3); font-size: .8rem; margin: 0 0 .7rem; }
.cmp { display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); }
.cmp div { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
           padding: .75rem .85rem; }
.cmp .who { font-size: .7rem; letter-spacing: .06em; text-transform: uppercase;
            color: var(--ink-3); }
.cmp .what { font-weight: 620; margin: .2rem 0 .3rem; font-size: .92rem; }
.cmp .how { color: var(--ink-2); font-size: .82rem; font-variant-numeric: tabular-nums; }
.empty { color: var(--ink-3); padding: 1.5rem 0; }
footer { color: var(--ink-3); font-size: .78rem; margin-top: 2.5rem;
         border-top: 1px solid var(--line); padding-top: 1rem; }
code { background: var(--card); border: 1px solid var(--line); padding: .05rem .35rem;
       border-radius: 6px; font-size: .85em; }
@media (max-width: 30rem) { .grid { grid-template-columns: repeat(2, 1fr); } }
"""

SCRIPT = """
const $ = (s, r=document) => r.querySelector(s);
const form = $('#f'), results = $('#results'), compare = $('#compare');
const paint = r => r.style.setProperty('--pct',
  ((r.value - r.min) / (r.max - r.min) * 100) + '%');
function label(name, value) {
  const tag = $('#' + name + '_v');
  const n = Number(value);
  tag.textContent = (Number.isFinite(n) ? n : 0) + (tag.dataset.unit || '');
}
document.querySelectorAll('input[type=range]').forEach(r => {
  const box = $('#' + r.dataset.pair);
  paint(r);
  r.addEventListener('input', () => {
    box.value = r.value; paint(r); label(r.dataset.pair, r.value); queue();
  });
  box.addEventListener('input', () => {
    r.value = Math.min(box.value || 0, r.max); paint(r);
    label(r.dataset.pair, box.value); queue();
  });
});
document.querySelectorAll('.chip[data-field]').forEach(c => {
  c.addEventListener('click', () => {
    document.querySelectorAll(`.chip[data-field="${c.dataset.field}"]`)
      .forEach(o => o.setAttribute('aria-pressed', o === c));
    $('#' + c.dataset.field).value = c.dataset.value;
    go();
  });
});
document.querySelectorAll('.chip[data-preset]').forEach(c => {
  c.addEventListener('click', () => {
    const [d, m, s, v] = c.dataset.preset.split(',');
    setPair('data', d); setPair('minutes', m); setPair('sms', s);
    document.querySelectorAll('.chip[data-field=validity]').forEach(o =>
      o.setAttribute('aria-pressed', o.dataset.value === v));
    $('#validity').value = v;
    go();
  });
});
function setPair(name, value) {
  const box = $('#' + name), range = $(`input[type=range][data-pair=${name}]`);
  box.value = value; range.value = Math.min(value, range.max);
  paint(range); label(name, value);
}
let timer;
const queue = () => { clearTimeout(timer); timer = setTimeout(go, 220); };
async function go() {
  const q = new URLSearchParams(new FormData(form)).toString();
  results.classList.add('busy');
  history.replaceState(null, '', '?' + q);
  try {
    const data = await (await fetch('api/recommend?' + q)).json();
    render(data);
  } catch (e) {
    results.innerHTML = '<p class="empty">Could not reach the server.</p>';
  }
  results.classList.remove('busy');
}
const money = n => 'BDT ' + n.toLocaleString(undefined, {maximumFractionDigits: 0});
function render(d) {
  if (d.error) { results.innerHTML = `<p class="empty">${d.error}</p>`; compare.innerHTML = ''; return; }
  results.innerHTML = `<p class="request">${d.request}</p>` + d.results.map((p, i) => `
    <article class="pack${i === 0 ? ' best' : ''}">
      <div class="tag">${i === 0 ? 'Recommended' : 'Alternative ' + i}
        ${p.top_pick ? '<span class="pill">model&rsquo;s top pick</span>' : ''}</div>
      <div class="name">${p.name}</div>
      <div class="grid">
        <div><span>Data</span><b>${p.data}</b></div>
        <div><span>Minutes</span><b>${p.minutes}</b></div>
        <div><span>SMS</span><b>${p.sms}</b></div>
        <div><span>Validity</span><b>${p.validity} d</b></div>
        <div><span>Price</span><b>${money(p.price)}</b></div>
        <div><span>USSD</span><b>${p.ussd}</b></div>
      </div>
      <div class="bar"><i style="width:${Math.round(p.score * 100)}%"></i></div>
      <div class="score">${p.label} ${Math.round(p.score * 100)}%</div>
      ${p.repeats > 1 ? `<div class="note plain">Buy it <b>${p.repeats}&times;</b>
        to cover ${d.days} days &rarr; <b>${money(p.period_price)}</b> in total.</div>` : ''}
      ${p.topups.length ? `<div class="note">Top up:<ul>${p.topups.map(t =>
        `<li><b>${t.what}</b> short &mdash; ${t.how}, <b>${money(t.price)}</b></li>`).join('')}
        </ul>All in: <b>${money(p.allin)}</b>${d.nothing_covers ? ' &mdash; nothing this operator sells covers it' : ''}</div>` : ''}
    </article>`).join('');
  compare.innerHTML = d.compare.length < 2 ? '' : `
    <h2>What the other models would pick</h2>
    <p>Same request, each model&rsquo;s own highest-scoring pack.</p>
    <div class="cmp">${d.compare.map(c => `<div>
      <div class="who">${c.model}</div>
      <div class="what">${c.name}</div>
      <div class="how">${money(c.price)} &middot; ${c.label.toLowerCase()} ${Math.round(c.score * 100)}%</div>
    </div>`).join('')}</div>`;
}
form.addEventListener('submit', e => { e.preventDefault(); go(); });
const root = document.documentElement;
try { root.dataset.theme = localStorage.getItem('theme') ||
  (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch (e) {}
$('#theme').addEventListener('click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
});
if (location.search) go();
"""

PAGE = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIM offer recommender</title>
<style>STYLE_HERE</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>SIM offer recommender</h1>
      <p class="sub">SUBTITLE_HERE</p>
    </div>
    <button class="toggle" id="theme" type="button" title="Light or dark">&#9681;</button>
  </header>

  <form class="panel" id="f" method="get" action="">
    FORM_HERE
  </form>

  <div class="results" id="results">RESULTS_HERE</div>
  <section class="compare" id="compare">COMPARE_HERE</section>

  <footer>FOOTER_HERE</footer>
</main>
<script>SCRIPT_HERE</script>
</body>
</html>
"""


def number(value, fallback=0.0):
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return fallback


def read_form(query, operators, models):
    fields = urllib.parse.parse_qs(query)

    def one(name, fallback=""):
        return fields.get(name, [fallback])[0]

    operator = one("operator")
    source = one("source")
    validity = number(one("validity"), 30.0) or 30.0
    return {
        "submitted": bool(fields),
        "models": models,
        "operator": operator if operator in operators else operators[0],
        "source": (source if source in models
                   else (models[0] if models else ml.MATCHER)),
        "asked_for": source,
        "data": number(one("data")),
        "minutes": number(one("minutes")),
        "sms": number(one("sms")),
        "validity": validity,
        "strict": False,
    }


def slider(name, label, value, maximum, step, unit):
    return f"""<div>
      <div class="lbl"><label for="{name}">{label}</label>
        <span class="val" id="{name}_v" data-unit="{unit}">{value:g}{unit}</span></div>
      <input type="range" min="0" max="{maximum}" step="{step}" value="{min(value, maximum):g}"
             data-pair="{name}" aria-label="{label}">
      <input type="number" min="0" step="{step}" id="{name}" name="{name}" value="{value:g}">
    </div>"""


def chips(field, options, chosen, ghost=False):
    kind = " ghost" if ghost else ""
    buttons = "".join(
        f'<button type="button" class="chip{kind}" data-field="{field}" '
        f'data-value="{esc(str(value))}" '
        f'aria-pressed="{"true" if str(value) == str(chosen) else "false"}">'
        f'{esc(str(label))}</button>'
        for value, label in options)
    return f'<div class="chips">{buttons}</div>' \
           f'<input type="hidden" id="{field}" name="{field}" value="{esc(str(chosen))}">'


def render_form(operators, form):
    presets = "".join(
        f'<button type="button" class="chip ghost" '
        f'data-preset="{d},{m},{s},{v}">{esc(name)}</button>'
        for name, d, m, s, v in PRESETS)
    sources = form["models"] or [ml.MATCHER]
    return f"""
    <div class="row">
      <div class="lbl">Operator</div>
      {chips("operator", [(o, o) for o in operators], form["operator"])}
    </div>
    <div class="row">
      <div class="lbl">Ranked by</div>
      {chips("source", [(s, s) for s in sources], form["source"])}
    </div>
    <div class="row sliders">
      {slider("data", "Data", form["data"], 100, 0.5, " GB")}
      {slider("minutes", "Minutes", form["minutes"], 2000, 10, " min")}
      {slider("sms", "SMS", form["sms"], 500, 5, "")}
    </div>
    <div class="row">
      <div class="lbl">Validity</div>
      {chips("validity", [(v, f"{v} d") for v in VALIDITY_CHOICES],
             int(form["validity"]))}
    </div>
    <div class="row">
      <div class="lbl">Or start from</div>
      <div class="chips">{presets}</div>
    </div>
    <button class="go" type="submit">Recommend</button>
    <p class="hint">Everything updates as you move a slider. Without
       JavaScript the button submits the form and the server renders the
       same page.</p>"""


def answer(catalogue, rates, form):
    operator, source = form["operator"], form["source"]
    wants = (form["data"], form["minutes"], form["sms"], form["validity"])
    packs = [o for o in catalogue if o["operator"] == operator]
    if not packs:
        return {"error": f"{esc(operator)} sells no buyable pack."}

    rate = rates[operator]
    by_id = {offer["offer_id"]: offer for offer in packs}

    if source == ml.MATCHER:
        ranked = recommend(catalogue, rates, operator, wants, top=CARDS)
        label, favourite = "Value", None
    else:
        try:
            picked, label = ml.rank(source, operator, wants, set(by_id), top=CARDS)
        except Exception as exc:
            return {"error": f"{esc(source)} could not answer this request: "
                             f"{esc(str(exc))}."}
        ranked = [(by_id[offer_id], score) for offer_id, score in picked]
        favourite = max(ranked, key=lambda pair: pair[1])[0] if ranked else None
        ranked.sort(key=lambda pair: (allin_cost(pair[0], wants, packs, rate),
                                      pair[0]["price_bdt"]))

    results = []
    for row, score in ranked:
        repeats = repeats_needed(row, form["validity"])
        gaps = topup(row, wants, packs, rate)
        results.append({
            "name": esc(row["offer_name"]),
            "data": ("Unlimited" if row["category"] == "Unlimited"
                     else f"{row['data_gb']:g} GB"),
            "minutes": f"{row['minutes']:g}", "sms": f"{row['sms']:g}",
            "validity": f"{row['validity_days']:g}",
            "price": row["price_bdt"], "ussd": esc(row["ussd_code"]),
            "score": round(score, 4), "label": label,
            "top_pick": row is favourite,
            "repeats": repeats,
            "period_price": period_price(row, form["validity"]),
            "allin": round(allin_cost(row, wants, packs, rate), 2),
            "topups": [{"what": esc(line.what), "price": round(line.price, 2),
                        "how": (esc(line.how) if line.pack_id is None
                                else f"add &ldquo;{esc(line.how)}&rdquo;")}
                       for line in gaps],
        })

    compare = []
    for name in form["models"]:
        if name == source:
            continue
        try:
            picked, other_label = ml.rank(name, operator, wants, set(by_id), top=1)
        except Exception:
            continue
        if picked:
            offer_id, score = picked[0]
            compare.append({"model": esc(name),
                            "name": esc(by_id[offer_id]["offer_name"]),
                            "price": by_id[offer_id]["price_bdt"],
                            "score": round(score, 4), "label": other_label})

    fallback = ""
    if source == ml.MATCHER and form["asked_for"] not in ("", ml.MATCHER):
        fallback = (f" &middot; {esc(form['asked_for'])} is not available "
                    "here, so the cost matcher answered")
    request = ("<b>{}</b> &middot; {:g} GB &middot; {:g} min &middot; {:g} SMS "
               "&middot; {:g} days &middot; picked by <b>{}</b>, cheapest "
               "first{}").format(esc(operator), *wants, esc(source), fallback)

    return {"request": request, "days": form["validity"], "results": results,
            "compare": compare,
            "nothing_covers": not any(covers(o, wants) for o in packs)}


def render_results(data):
    if "error" in data:
        return f'<p class="empty">{data["error"]}</p>', ""

    cards = []
    for rank, pack in enumerate(data["results"]):
        heading = "Recommended" if rank == 0 else f"Alternative {rank}"
        pill = ('<span class="pill">model&rsquo;s top pick</span>'
                if pack["top_pick"] else "")
        cells = [("Data", pack["data"]), ("Minutes", pack["minutes"]),
                 ("SMS", pack["sms"]), ("Validity", pack["validity"] + " d"),
                 ("Price", f"BDT {pack['price']:g}"), ("USSD", pack["ussd"])]
        grid = "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in cells)

        extras = ""
        if pack["repeats"] > 1:
            extras += ('<div class="note plain">Buy it <b>{}&times;</b> to cover '
                       '{:g} days &rarr; <b>BDT {:g}</b> in total.</div>').format(
                           pack["repeats"], data["days"], pack["period_price"])
        if pack["topups"]:
            items = "".join(f'<li><b>{t["what"]}</b> short &mdash; {t["how"]}, '
                            f'<b>BDT {t["price"]:g}</b></li>'
                            for t in pack["topups"])
            tail = (" &mdash; nothing this operator sells covers it"
                    if data["nothing_covers"] else "")
            extras += (f'<div class="note">Top up:<ul>{items}</ul>All in: '
                       f'<b>BDT {pack["allin"]:g}</b>{tail}</div>')

        cards.append(
            f'<article class="pack{" best" if rank == 0 else ""}">'
            f'<div class="tag">{heading} {pill}</div>'
            f'<div class="name">{pack["name"]}</div>'
            f'<div class="grid">{grid}</div>'
            f'<div class="bar"><i style="width:{pack["score"] * 100:.0f}%"></i></div>'
            f'<div class="score">{pack["label"]} {pack["score"]:.0%}</div>'
            f"{extras}</article>")

    results = f'<p class="request">{data["request"]}</p>' + "".join(cards)

    compare = ""
    if len(data["compare"]) >= 2:
        boxes = "".join(
            f'<div><div class="who">{c["model"]}</div>'
            f'<div class="what">{c["name"]}</div>'
            f'<div class="how">BDT {c["price"]:g} &middot; '
            f'{c["label"].lower()} {c["score"]:.0%}</div></div>'
            for c in data["compare"])
        compare = ("<h2>What the other models would pick</h2>"
                   "<p>Same request, each model&rsquo;s own highest-scoring "
                   f'pack.</p><div class="cmp">{boxes}</div>')
    return results, compare


def subtitle(catalogue, operators, models, problem):
    if not models:
        return (f"287 packs across {len(operators)} operators. "
                f"No trained model loads here ({esc(problem)}), so the cost "
                "matcher is answering.")
    scored = ml.metrics()
    parts = ", ".join(f"{name.lower()} <b>{scored[name]['accuracy']:.0%}</b>"
                      for name in models if name in scored)
    warning = f" <b>Warning:</b> {esc(problem)}." if problem else ""
    return (f"{len(catalogue)} packs across {len(operators)} operators, ranked "
            f"by a model trained on 25,000 requests &mdash; {parts} top-1 "
            f"agreement on held-out requests.{warning}")


def render_page(catalogue, rates, operators, query):
    models, problem = ml.available()
    models = ([n for n in MODEL_ORDER if n in models]
              + [n for n in models if n not in MODEL_ORDER])
    form = read_form(query, operators, models)

    if form["submitted"]:
        results, compare = render_results(answer(catalogue, rates, form))
    else:
        results, compare = ('<p class="empty">Set what you want above and it '
                            'prices every pack that operator sells.</p>', "")

    footer = ("Ranked by the model chosen above, over the packs that operator "
              "sells. Where a pack falls short, the top-up under it is the "
              "catalogue&rsquo;s own cheapest filler, or BDT "
              f"{SMS_TOPUP_BDT:g} an SMS where no pack fits. Trained by "
              "<code>python3 scripts/train_model.py</code>.")

    return (PAGE
            .replace("STYLE_HERE", STYLE)
            .replace("SCRIPT_HERE", SCRIPT)
            .replace("SUBTITLE_HERE",
                     subtitle(catalogue, operators, models, problem))
            .replace("FORM_HERE", render_form(operators, form))
            .replace("RESULTS_HERE", results)
            .replace("COMPARE_HERE", compare)
            .replace("FOOTER_HERE", footer)).encode("utf-8")


def route(path, query, catalogue, rates, operators):
    if path.endswith("/favicon.ico"):
        return 204, "text/plain", b""
    if path.endswith("/api/recommend"):
        models, _ = ml.available()
        models = ([n for n in MODEL_ORDER if n in models]
                  + [n for n in models if n not in MODEL_ORDER])
        form = read_form(query, operators, models)
        payload = answer(catalogue, rates, form)
        return 200, "application/json", json.dumps(payload).encode("utf-8")
    return 200, "text/html; charset=utf-8", render_page(catalogue, rates,
                                                        operators, query)


def make_handler(catalogue, rates, operators):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            path, _, query = self.path.partition("?")
            status, kind, body = route(path, query, catalogue, rates, operators)
            self.send_response(status)
            if body:
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description="One web page for the recommender and its trained models.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    catalogue = load_catalogue()
    rates = reference_rates(catalogue)
    operators = sorted({o["operator"] for o in catalogue})

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(catalogue, rates, operators))
    print(f"SIM offer recommender on {url}   (ctrl-c to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
