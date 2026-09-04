"""One web page for the matcher: a form, and the packs it recommends.

    python3 scripts/web.py          # opens http://127.0.0.1:8000

Same five inputs as user_test.py, same three results, same arithmetic --
this only draws them. Every number on the page comes from user_test.py's
functions, so there is one matching algorithm in this project, not two, and
the page cannot drift away from the command line.

Standard library only, like the matcher: http.server, no Flask, no pip.
The page is rendered on the server and posted back as a plain form, so it
needs no JavaScript either. It listens on 127.0.0.1, so nothing outside
this machine can reach it.

    python3 scripts/web.py --port 8080 --no-open
"""

import argparse
import html
import os
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml
from user_test import (SMS_TOPUP_BDT, allin_cost, covers, load_catalogue,
                       period_price, recommend, reference_rates,
                       repeats_needed, topup)

# Three packs get a card each. MORE is how many further packs are listed in
# a table under them, ranked the same way; 0 turns that table off, which is
# how the page currently stands. Raise it to bring the table back.
CARDS = 3
MORE = 0

# The models the page can rank with, best first, so the first is the default.
# The cost matcher still answers when no model can be loaded -- it is what
# the page falls back to, not something to choose.
MODEL_ORDER = ["Random forest", "Decision tree", "Polynomial regression"]

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIM offer recommender</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #ffffff; --ink: #16191d; --muted: #6b7280;
    --line: #e3e6ea; --accent: #1d6f4f; --accent-soft: #e7f2ec;
    --warn: #8a5a00; --warn-soft: #fdf3e0;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171a; --card: #1c2024; --ink: #e9ecef; --muted: #9aa3ad;
      --line: #2b3138; --accent: #5fd1a0; --accent-soft: #1d2b25;
      --warn: #e0b464; --warn-soft: #2c2519;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem 4rem; background: var(--bg); color: var(--ink);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  main { max-width: 46rem; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
  .sub { color: var(--muted); margin: 0 0 1.5rem; font-size: .9rem; }
  form {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 1.25rem; display: grid; gap: .9rem;
    grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr));
  }
  label { display: block; font-size: .78rem; text-transform: uppercase;
          letter-spacing: .04em; color: var(--muted); margin-bottom: .35rem; }
  input, select {
    width: 100%; padding: .5rem .6rem; font: inherit; color: var(--ink);
    background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  }
  .wide { grid-column: 1 / -1; display: flex; gap: 1rem; align-items: center;
          flex-wrap: wrap; }
  .check { display: flex; align-items: center; gap: .45rem; color: var(--muted);
           font-size: .85rem; }
  .check input { width: auto; }
  button {
    padding: .55rem 1.4rem; font: inherit; font-weight: 600; cursor: pointer;
    color: #fff; background: var(--accent); border: 0; border-radius: 8px;
  }
  @media (prefers-color-scheme: dark) { button { color: #10221a; } }
  .request { margin: 1.75rem 0 .75rem; color: var(--muted); font-size: .9rem; }
  .pack {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 1.1rem 1.25rem; margin-bottom: 1rem;
  }
  .pack.best { border-color: var(--accent); }
  .rank { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
          color: var(--muted); }
  .pack.best .rank { color: var(--accent); font-weight: 700; }
  .name { font-size: 1.1rem; font-weight: 600; margin: .15rem 0 .8rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(5.5rem, 1fr));
          gap: .7rem 1rem; margin-bottom: .9rem; }
  .grid div span { display: block; font-size: .72rem; text-transform: uppercase;
                   letter-spacing: .04em; color: var(--muted); }
  .grid div b { font-weight: 600; font-variant-numeric: tabular-nums; }
  .bar { height: 6px; border-radius: 3px; background: var(--line);
         overflow: hidden; margin: .2rem 0 .1rem; }
  .bar i { display: block; height: 100%; background: var(--accent); }
  .note { border-radius: 8px; padding: .6rem .75rem; font-size: .87rem;
          background: var(--accent-soft); margin-top: .6rem; }
  .note.gap { background: var(--warn-soft); color: var(--warn); }
  .note b { font-variant-numeric: tabular-nums; }
  .empty { color: var(--muted); margin-top: 2rem; }
  h2 { font-size: .95rem; margin: 2rem 0 .2rem; }
  .more-note { color: var(--muted); font-size: .85rem; margin: 0 0 .75rem; }
  .scroll { overflow-x: auto; background: var(--card); border-radius: 12px;
            border: 1px solid var(--line); }
  table { border-collapse: collapse; width: 100%; font-size: .88rem; }
  th, td { padding: .55rem .75rem; text-align: right; white-space: nowrap;
           border-top: 1px solid var(--line); }
  th { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
       color: var(--muted); font-weight: 600; border-top: 0; }
  th:first-child, td:first-child { text-align: left; white-space: normal;
                                   min-width: 12rem; }
  td b { font-variant-numeric: tabular-nums; }
  td .plus { display: block; color: var(--warn); font-size: .8rem; }
  footer { color: var(--muted); font-size: .8rem; margin-top: 2.5rem;
           border-top: 1px solid var(--line); padding-top: 1rem; }
  code { background: var(--card); padding: .1rem .35rem; border-radius: 5px; }
</style>
</head>
<body>
<main>
  <h1>SIM offer recommender</h1>
  <p class="sub">MATCHER_LINE</p>
  <form method="get" action="/">
    FORM_FIELDS
  </form>
  RESULTS
  <footer>
    Ranked by the trained model chosen above, over the packs the operator
    actually sells. Where a pack falls short of the request, the top-up
    priced under it comes from the catalogue, not the model: the cheapest
    pack that fills the gap, or BDT SMS_RATE an SMS where no pack does.
    Trained by <code>python3 scripts/train_model.py</code>.
  </footer>
</main>
</body>
</html>
"""


def esc(value):
    return html.escape(str(value), quote=True)


def form_fields(operators, form):
    options = "".join(
        f'<option value="{esc(op)}"{" selected" if form["operator"] == op else ""}>'
        f'{esc(op)}</option>' for op in operators)

    def field(name, label, value, step="any"):
        # :g so the box reads 10 and 30, not 10.0 and 30.0.
        return (f'<div><label for="{name}">{label}</label>'
                f'<input id="{name}" name="{name}" type="number" min="0" '
                f'step="{step}" value="{esc(format(value, "g"))}"></div>')

    sources = form["models"] or [ml.MATCHER]
    ranked_by = "".join(
        f'<option value="{esc(name)}"'
        f'{" selected" if form["source"] == name else ""}>{esc(name)}</option>'
        for name in sources)

    return (
        f'<div><label for="operator">Operator</label>'
        f'<select id="operator" name="operator">{options}</select></div>'
        + field("data", "Data (GB)", form["data"])
        + field("minutes", "Minutes", form["minutes"])
        + field("sms", "SMS", form["sms"])
        + field("validity", "Validity (days)", form["validity"], step="1")
        + '<div><label for="source">Ranked by</label>'
          f'<select id="source" name="source">{ranked_by}</select></div>'
        + '<div class="wide">'
          '<button type="submit">Recommend</button>'
          '</div>')


def render_pack(row, value, wants, packs, rates, rank, nothing_covers,
                value_label="Value", top_pick=False):
    validity_want = wants[3]
    unlimited = row["category"] == "Unlimited"
    data = "Unlimited" if unlimited else f"{row['data_gb']:g} GB"
    heading = "Recommended" if rank == 0 else f"Alternative {rank}"
    # Cards are ordered by price, so the model's own favourite is not
    # necessarily the first one. Say which it was, or a 54% sitting under a
    # 16% reads as a bug.
    if top_pick:
        heading += " &middot; the model's top pick"

    cells = [("Data", data), ("Minutes", f"{row['minutes']:g}"),
             ("SMS", f"{row['sms']:g}"),
             ("Validity", f"{row['validity_days']:g} days"),
             ("Price", f"BDT {row['price_bdt']:g}"), ("USSD", row["ussd_code"])]
    grid = "".join(f"<div><span>{esc(k)}</span><b>{esc(v)}</b></div>"
                   for k, v in cells)

    notes = []
    repeats = repeats_needed(row, validity_want)
    if repeats > 1:
        total_data = "Unlimited" if unlimited else f"{row['data_gb'] * repeats:g} GB"
        notes.append(
            '<div class="note">To cover {:g} days, buy it <b>{}x</b> &rarr; {}, '
            '{:g} min, {:g} SMS, <b>BDT {:g}</b> in total.</div>'.format(
                validity_want, repeats, esc(total_data), row["minutes"] * repeats,
                row["sms"] * repeats, row["price_bdt"] * repeats))

    missing = topup(row, wants, packs, rates)
    if missing:
        lines = []
        for line in missing:
            how = (esc(line.how) if line.pack_id is None
                   else f'add &ldquo;{esc(line.how)}&rdquo;')
            lines.append(f"<li><b>{esc(line.what)}</b> short &mdash; {how}, "
                         f"<b>BDT {line.price:g}</b></li>")
        allin = period_price(row, validity_want) + sum(l.price for l in missing)
        tail = (f" &mdash; {esc(row['operator'])} sells nothing that covers "
                f"this request" if nothing_covers else "")
        notes.append(
            '<div class="note gap">Top up:<ul style="margin:.35rem 0 .5rem;'
            'padding-left:1.1rem">{}</ul>All in: <b>BDT {:g}</b>{}</div>'.format(
                "".join(lines), allin, tail))

    return (
        f'<article class="pack{" best" if rank == 0 else ""}">'
        f'<div class="rank">{heading}</div>'
        f'<div class="name">{esc(row["offer_name"])}</div>'
        f'<div class="grid">{grid}</div>'
        f'<div class="bar"><i style="width:{value * 100:.0f}%"></i></div>'
        f'<div class="rank">{esc(value_label)} {value:.0%}</div>'
        f'{"".join(notes)}</article>')


def render_row(row, value, wants, packs, rates):
    """One line of the table under the cards."""
    """One line of the table under the cards: what it gives, what it all
    costs, and what it would take to fill whatever it misses."""
    unlimited = row["category"] == "Unlimited"
    validity_want = wants[3]
    repeats = repeats_needed(row, validity_want)

    missing = topup(row, wants, packs, rates)
    allin = period_price(row, validity_want) + sum(l.price for l in missing)
    gaps = ("<span class=\"plus\">+ {} &mdash; BDT {:g}</span>".format(
        ", ".join(esc(line.what) for line in missing),
        sum(line.price for line in missing)) if missing else "")
    buys = f" &times;{repeats}" if repeats > 1 else ""

    cells = ["Unlimited" if unlimited else f"{row['data_gb']:g} GB",
             f"{row['minutes']:g}", f"{row['sms']:g}",
             f"{row['validity_days']:g}", f"BDT {row['price_bdt']:g}{buys}"]
    return ("<tr><td>{}{}</td>{}<td><b>BDT {:g}</b></td>"
            "<td>{:.0%}</td></tr>").format(
                esc(row["offer_name"]), gaps,
                "".join(f"<td>{c}</td>" for c in cells), allin, value)


def render_more(results, wants, packs, rates, strict, value_label="Value"):
    if not results:
        return ""
    rows = "".join(render_row(row, value, wants, packs, rates)
                   for row, value in results)

    # In strict mode the order comes from the hard rule -- coverage first,
    # then the cost model -- so the all-in and value columns are genuinely
    # not in order, and the note should not pretend they are.
    if value_label != "Value":
        order = ("Ordered by the model, like the three above. All in is what "
                 "the matcher says each pack really costs, which the model "
                 "never sees.")
    elif strict:
        order = ("Ordered by the strict rule, coverage first, so the last "
                 "two columns are not in order.")
    else:
        order = "Ranked the same way as the three above, by the all-in cost."
    return (
        "<h2>More options</h2>"
        f'<p class="more-note">{order} <b>All in</b> includes '
        "filling anything the pack misses; a price shown &times;n is what "
        "the repeat purchases cost over the whole period.</p>"
        '<div class="scroll"><table><thead><tr>'
        "<th>Pack</th><th>Data</th><th>Min</th><th>SMS</th><th>Days</th>"
        f"<th>Price</th><th>All in</th><th>{esc(value_label)}</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>")


def render_results(catalogue, rates, form):
    if not form["submitted"]:
        return ('<p class="empty">Tell it what you want and it will price '
                'every pack that operator sells.</p>')

    operator = form["operator"]
    wants = (form["data"], form["minutes"], form["sms"], form["validity"])
    packs = [o for o in catalogue if o["operator"] == operator]

    if form["source"] == ml.MATCHER:
        results = recommend(catalogue, rates, operator, wants,
                            top=CARDS + MORE, strict=form["strict"])
        value_label = "Value"
        asked = form["asked_for"]
        aside = ("" if asked in ("", ml.MATCHER) else
                 f" &middot; {esc(asked)} is not available in this "
                 "interpreter, so the cost matcher answered")
    else:
        # The model ranks; the matcher's numbers still describe each pack it
        # returns, because those are facts about the pack, not opinions.
        by_id = {offer["offer_id"]: offer for offer in packs}
        try:
            picked, value_label = ml.rank(form["source"], operator, wants,
                                          set(by_id), top=CARDS + MORE)
        except Exception as exc:
            # A model that will not answer must not take the page down with
            # it: say what happened, and let the matcher still be one click
            # away in the selector above.
            return (f'<p class="empty">{esc(form["source"])} could not answer '
                    f"this request: {esc(exc)}.<br>Retrain with "
                    "<code>python3 scripts/train_model.py</code>, or pick "
                    "the cost matcher above.</p>")
        results = [(by_id[offer_id], score) for offer_id, score in picked]

        # The model chooses which packs are worth showing; price decides the
        # order they are shown in, cheapest first. Cheapest means what the
        # request really costs with that pack -- repeat purchases and any
        # top-up included -- so a BDT 52 pack that has to be bought five
        # times does not lead a BDT 208 one that lasts the month.
        results.sort(key=lambda pair: (allin_cost(pair[0], wants, packs,
                                                  rates[operator]),
                                       pair[0]["price_bdt"]))
        aside = (" &middot; picked by " + esc(form["source"])
                 + ", cheapest first"
                 + (" (strict applies to the cost matcher only)"
                    if form["strict"] else ""))

    if not results:
        return f'<p class="empty">{esc(operator)} sells no buyable pack.</p>'
    nothing_covers = not any(covers(o, wants) for o in packs)

    header = ('<p class="request">{} &middot; {:g} GB &middot; {:g} min '
              '&middot; {:g} SMS &middot; {:g} days{}</p>').format(
                  esc(operator), *wants, aside)
    shown = results[:CARDS]
    favourite = max(range(len(shown)), key=lambda i: shown[i][1]) if shown else -1
    cards = "".join(
        render_pack(row, value, wants, packs, rates[operator], rank,
                    nothing_covers, value_label,
                    top_pick=(rank == favourite and value_label != "Value"))
        for rank, (row, value) in enumerate(shown))
    return header + cards + render_more(results[CARDS:], wants, packs,
                                        rates[operator], form["strict"],
                                        value_label)


def read_form(query, operators, models):
    fields = urllib.parse.parse_qs(query)

    def number(name, default=0.0):
        try:
            return max(float(fields.get(name, [""])[0]), 0.0)
        except (ValueError, IndexError):
            return default

    operator = fields.get("operator", [""])[0]
    source = fields.get("source", [""])[0]
    return {
        "submitted": bool(fields),
        "models": models,
        "source": (source if source in models
                   else (models[0] if models else ml.MATCHER)),
        # What was asked for, so a fallback to the matcher can be admitted
        # rather than passed off as the model's answer.
        "asked_for": source,
        "operator": operator if operator in operators else operators[0],
        "data": number("data"),
        "minutes": number("minutes"),
        "sms": number("sms"),
        "validity": number("validity", 30.0) or 30.0,
        # A matcher-only setting, kept for the command line and for the
        # fallback below; the page no longer offers it.
        "strict": False,
    }


def source_line(models, problem):
    """One line saying what is doing the ranking, with what each scored."""
    if not models:
        return (f"No trained model can be loaded here ({esc(problem)}), so "
                "the cost matcher is answering.")
    warning = (f' <b>Warning:</b> {esc(problem)}.' if problem else "")
    scored = ml.metrics()
    parts = [f"{name.lower()} {scored[name]['accuracy']:.0%}"
             for name in models if name in scored]
    return ("Ranked by a model trained on 25,000 requests &mdash; "
            + ", ".join(parts)
            + " top-1 agreement with the cost model on held-out requests."
            + warning)


def render_page(catalogue, rates, operators, query):
    """The whole page for one request, as UTF-8 bytes.

    Kept apart from the server so anything that can answer an HTTP GET can
    serve it: the local server below, and api/index.py on Vercel, which gets
    one invocation per request and no server to run.
    """
    models, problem = ml.available()
    models = ([name for name in MODEL_ORDER if name in models]
              + [name for name in models if name not in MODEL_ORDER])
    form = read_form(query, operators, models)
    matcher_line = (f"303 buyable packs across "
                    f"{len(operators)} operators. "
                    + source_line(models, problem))
    return (PAGE
            .replace("MATCHER_LINE", matcher_line)
            .replace("FORM_FIELDS", form_fields(operators, form))
            .replace("RESULTS", render_results(catalogue, rates, form))
            .replace("SMS_RATE", f"{SMS_TOPUP_BDT:g}")
            ).encode("utf-8")


def make_handler(catalogue, rates, operators):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if path != "/":
                self.send_error(404)
                return

            body = render_page(catalogue, rates, operators, query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass          # one line per keystroke-ish reload is just noise

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true",
                        help="do not open a browser")
    args = parser.parse_args()

    catalogue = load_catalogue()
    rates = reference_rates(catalogue)
    operators = sorted({o["operator"] for o in catalogue})

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(catalogue, rates, operators))
    print(f"SIM offer recommender on {url}   (ctrl-c to stop)")
    print(f"{len(catalogue)} buyable packs, {len(operators)} operators, "
          f"matching algorithm only.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
