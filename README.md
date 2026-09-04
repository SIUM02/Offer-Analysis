# Bangladesh SIM Offer Dataset

Live offer data for the four operators that publish it — Grameenphone,
Banglalink, Teletalk and Robi — scraped from each operator's own site and
normalised into one schema, then used to train models that recommend the best
package for a given usage. Every price, quota, validity and USSD code comes
from the operator; nothing is estimated.

## Start here

**`data/all_operators_official.csv`** — 287 packages, 14 columns, no blank
cells, every row a domestic package with a real price (৳7–৳2,997):
Grameenphone 93, Teletalk 86, Banglalink 80, Robi 28.

Categories: 108 Data, 68 Combo, 45 Minute, 25 Call Rate, 16 SMS, 8 Validity,
6 Unlimited, 6 Bondho SIM, 2 Entertainment, 2 New SIM, 1 Cashback.

### Columns

| Column | Meaning |
|---|---|
| `operator` | Grameenphone / Banglalink / Teletalk / Robi |
| `offer_id` | the operator's own pack ID where it publishes one (`GPD250026R`, `TK119DATAUSSD72HRR`), otherwise generated |
| `offer_name` | the pack as the operator labels it |
| `category` | see the category notes below |
| `data_gb`, `minutes`, `sms` | quota included; `0` means the pack does not include that |
| `validity_days` | validity in days; hours are converted (24h → 1.0) |
| `ussd_code` | activation dial code, or `Unknown` (89 of 309 — Robi publishes none, some others sell in-app only) |
| `source_page` | which operator page or endpoint the row came from, for auditing |
| `price_bdt`, `price_per_gb`, `price_per_minute`, `price_per_sms` | pack price in taka, and ৳ per unit derived from it; `0` when the pack has no such quota |

### Category notes

Most categories are ordinary bundles. Three mislead a model if treated as
plain packs: **`Call Rate` (25)** buys a cheaper per-minute tariff rather than
minutes (`minutes = 0`, the value is in `price_per_minute`); **`Unlimited`
(6)** stores `data_gb = 0` meaning uncapped, not empty, so exclude it before
any ৳/GB statistic or it reads as free; **`Validity` (8)** extends account
validity and usually carries no quota.

## The 4x expanded table

**`data/all_operators_expanded.csv`** — 1,137 rows, the same 14 columns: the
287 real rows unchanged, plus **850 synthetic siblings**, three per real pack.
For a bigger training set only; anything meant to be true about what operators
sell comes from `all_operators_official.csv`. (The generator that produced it
is no longer in the repo; the file is kept as it was built.)

Nothing was invented from scratch. Each sibling is a real pack resized: quota
scaled by 0.5–3x, price by that factor to the power 0.85 (bigger packs cost
less per unit), the label rewritten in the operator's own wording — `5 GB+250
minute+300 SMS` becomes `10 GB+500 minute+300 SMS`. Nothing leaves the range
that operator publishes: quota capped at its largest real pack, price clamped
to its real ৳ range, validity moved only to a neighbouring term it already
sells. Packs whose numbers are a tariff rather than a quota (`Call Rate`,
`Validity`, `Bondho SIM`, `New SIM`, `Cashback`, `Unlimited`,
`Entertainment`) keep quota and term and vary only in price.

Median ৳/unit lands where the real catalogue already is:

| | GP | Banglalink | Teletalk | Robi |
|---|---|---|---|---|
| ৳/GB, data packs ≥5 GB — real | 32.98 | 15.33 | 10.84 | 7.98 |
| ৳/GB, data packs ≥5 GB — synthetic | 36.23 | 13.68 | 10.22 | 9.16 |
| ৳/minute, minute packs — real | 0.75 | 0.71 | 0.64 | 0.73 |
| ৳/minute, minute packs — synthetic | 0.76 | 0.68 | 0.61 | 0.74 |

**Telling them apart.** Synthetic rows are marked inside the schema:
`offer_id` ends in `-V2`/`-V3`/`-V4` (Teletalk and Robi number sequentially,
so theirs continue past the real range — `TT-087` and up, `ROBI-029` and up),
and `ussd_code` is always `Unknown`, because a made-up dial code is a dialable
one. The first 287 rows are the official file byte for byte.

**What it is not.** These packs are not for sale and no operator ever
published them. Plausible, not real: train on them, never quote a price from
them. To use them, point `CATALOGUE` in `train_model.py` at the expanded file
— 824 of 1,137 rows survive the category and app-restriction filters, against
203 of 287.

## Per-operator files

The four sources keep **17 columns** — the 14 above plus `speed_cap_mbps`,
`roaming` and `auto_renewal`, which the merge drops.

| File | Rows | Source | How |
|---|---|---|---|
| `data/gp_offers_official.csv` | 112 | grameenphone.com | Pack table + tariff pages + 11 offer pages, each with its own adapter |
| `data/bl_offer_official.csv` | 97 | banglalink.net | Its public JSON API — packs arrive already structured |
| `data/tt_offer_official.csv` | 96 | teletalk.com.bd | Plain server-rendered HTML tables |
| `data/robi_offer_official.csv` | 28 | robi.com.bd | Headless browser — client-rendered, no public API |

`data/official_raw/` holds source captures for auditing; `data/kaggle_raw/`
holds a 2023 Kaggle snapshot kept for historical comparison, not part of the
pipeline. The scrapers and `merge_official.py` that produced these files are
not in this repo, and operators change prices often — re-scrape before
relying on the numbers.

## What the merge does

The four per-operator files were concatenated with four corrections, all
reproducible; the per-operator files were never modified. (The scrapers and
`merge_official.py` are not in this repo — the CSVs are their output.)

**1. Dropped three near-empty columns:** `roaming` (exactly equivalent to
`category == "Roaming"`, 22 of 22), `speed_cap_mbps` (0 in 328 of 333) and
`auto_renewal` (`Unknown` in 275 of 333 — only GP publishes it).

**2. Dropped 7 non-purchasable entries**, all Banglalink. Its "Others" tab
mixes service and marketing pages in with real packs — eSIM, Emergency
Balance, handset instalments — with no price, quota, validity or tariff. The
test is about content rather than category, so rows that are legitimately
price-0 but sell a rate survive it.

**3. Relabelled 4 miscategorised packs to `Validity`:** Banglalink files
main-account validity packs under Voice, Robi files multi-year Shodesh packs
under Bundles, and neither contains minutes or data.

**4. Excluded 22 roaming packs.** Only GP (12) and Banglalink (10) publish
priced roaming; Teletalk documents eligibility rules instead and Robi hides
its packs behind a country picker. Kept in, they would teach a model that two
operators offer roaming and two do not — a publishing gap, not a real
difference — and they are priced for a different job (up to ৳9,994 against a
৳2,997 domestic maximum). The table is **domestic only**; roaming rows remain
in the two per-operator files.

**17 tariff plans** went to `tariff_plans_official.csv`. A tariff plan sells
no quota — it is the rate you are on — and 11 of 17 have no joining fee, so
in the pack table they looked like free offers a recommender would rank top.
Their value is `price_per_minute` / `price_per_sms` (Teletalk Gen-Z ৳0.50/min
against GP Nishchinto ৳2.00/min). Join on `operator`.

## Two traps before comparing operators

**1. Robi's 28 rows are a curated sample, not a small catalogue.**
robi.com.bd personalises its public listing and the full list needs a login
with a Robi number; two tabs return nothing to a signed-out visitor. Every
Robi row is real, but any "Robi is cheaper / has less" conclusion is
unsupported.

**2. Median ৳/GB across all Data packs is misleading.** Run naively it says
GP charges 13× Teletalk. That is mostly *pack mix* — GP lists many small
streaming packs (0.08–1 GB), and small packs always cost more per GB:

| Operator | All data packs | Packs ≥ 5 GB |
|---|---|---|
| Grameenphone | ৳171.14 (n=39) | ৳32.98 (n=9) |
| Banglalink | ৳19.60 (n=29) | ৳15.33 (n=23) |
| Teletalk | ৳13.05 (n=38) | ৳10.84 (n=19) |
| Robi | ৳20.33 (n=2) | ৳7.98 (n=1) |

GP is still dearest, but ~3× Teletalk rather than ~13×. The third trap,
uneven roaming coverage, is handled by excluding roaming entirely (merge
step 4).

## Figures

```bash
python3 scripts/plot_dataset.py          # -> figures/*.png
python3 scripts/plot_dataset.py --dark   # -> figures/dark/*.png
```

| File | What it shows |
|---|---|
| `1_catalogue.png` | What each operator sells, split by what the pack is for |
| `2_price_volume.png` | Price against data volume, one panel per operator, both axes log |
| `3_price_per_gb.png` | What a gigabyte costs — median and middle half, data-only packs |
| `4_validity.png` | How long packs last |
| `5_labels.png` | The packs the cost model picks most often — the label the models learn |

Needs matplotlib and pandas; nothing else in the project does.
**Two things they make obvious.** GP's median data-only pack is BDT 98/GB
against Banglalink's BDT 9 — a median dragged up by small content packs, not
a bulk rate, which is why top-ups are priced from the 25th percentile. And
the labels are a long tail: 158 of 203 packs are ever chosen and the top 15
cover a third of all 25,000 requests, so a model that learned only those
would already look accurate — the argument for the money column. Each
operator keeps one hue throughout; figure 2 is small multiples because this
palette's orange and yellow are not separable for a red-green colourblind
reader when any two series may be compared.

## The recommendation model

```bash
python3 scripts/train_model.py       # fits all three and prints the scores
python3 scripts/export_models.py     # writes models/models.json.gz
```

Five inputs — operator, data (GB), minutes, SMS, validity (days) — and each
model returns the packs it would recommend with its own score.

### Three models, compared

`scripts/train_model.py` trains all three on the same 25,000 requests, the
same five features and the same split, and scores them on the 5,000 held
out. Every prediction is filtered to the chosen operator's packs, identically
for all three, so no model can answer a Robi request with a GP offer.

| Model | Top-1 | Top-3 | Mean BDT lost | 90th pct | Fit |
|---|---|---|---|---|---|
| Decision tree | 92.9% | 95.2% | 9.93 | 0.00 | 0.1 s |
| Random forest (150 trees) | 91.6% | 97.7% | 25.79 | 0.00 | 1.0 s |
| Polynomial regression (degree 2) | 37.8% | 66.7% | 153.86 | 538.50 | 0.1 s |

Top-1 and Top-3 are how often the pick, or the top three, contain the pack
the cost model chose — agreement with the thing that labelled the data, not
real-world correctness. **Mean BDT lost** prices the disagreements: what the
model's pack costs over the pack it should have chosen, same request, in
money. A correct pick scores 0. It is the more honest column, because a wrong
pack that costs the same is not a bad answer.

The tree models learn the cost rule almost exactly; the polynomial regression
does not come close, which is the useful result rather than a disappointing
one. A degree-2 surface must separate 158 packs with one smooth function of
five inputs, while the rule it chases is full of hard edges — a pack is
disqualified the moment it falls one SMS short, and trees cut on exactly
those edges. Note too that the decision tree is *less* accurate than the
forest but loses *less money*: the forest's extra correct picks are on
requests where being wrong was cheap anyway.

**Why 150 trees.** A forest stores a probability for all 158 packs at every
leaf, so size follows leaf count, not accuracy:

| trees | min leaf | top-1 | top-3 | in memory | on disk |
|---|---|---|---|---|---|
| 300 | 2 | 93.5% | 98.8% | 1333 MB | 41 MB |
| 150 | 5 | 91.6% | 97.7% | 395 MB | 15 MB |
| 100 | 8 | 89.9% | 97.0% | 191 MB | 9 MB |
| 60 | 12 | 87.3% | 96.4% | 86 MB | 4 MB |

The 300-tree forest is the most accurate and needs 1.3 GB of memory to answer
one question, which no web process should carry; 150 trees gives up 1.9
points of top-1 for a fifth of the size.

**Reproducing it.** Training needs scikit-learn, pandas and joblib — the
matcher, the page and the exported models do not. It writes
`models/offer_recommender.joblib` (15 MB, all three) and
`data/training_profiles.csv`, both gitignored as build products, in about
eleven seconds. A pickle is only portable between matching library versions,
so retrain rather than move that file between machines; the results are
stable across versions (identical figures under scikit-learn 1.6 and 1.9),
and `models/models.json.gz` is the portable artefact.

## Trying it on your own input

```bash
python3 scripts/user_test.py                     # asks the five questions
python3 scripts/user_test.py --operator Teletalk \
    --data 5 --minutes 250 --sms 50 --validity 30
```

Blank leaves a field at 0 (validity 30), `q` quits, and it loops; use the
flags when the run has no keyboard. This one uses **no model** — matching
alone, on the standard library and the CSV.

**It ranks on the all-in cost** — not the sticker price, but what the whole
request costs with that pack: the pack over the period asked for, plus the
price of buying whatever it leaves out. So a pack that misses on one thing
can still win. Ask Banglalink for 10 GB **and 50 SMS**; Banglalink sells no
SMS pack at all, so the 50 SMS are priced at what they cost (`SMS_TOPUP_BDT`,
BDT 1 each) and the arithmetic is printed:

```
RECOMMENDED  Banglalink Prepaid 10GB Tk. 398!
    Value      : 100%
    Top up     : 50 SMS short -> pay as you go at BDT 1, BDT 50
    All in     : BDT 448  (Banglalink sells nothing that covers this)
```

**A gap is priced at the pack that fills it**, not at an average rate.
Banglalink data averages BDT 7.52/GB, so a per-unit rate values a 6 GB gap
at BDT 46 and makes a BDT 137 minute pack look like a fine answer to a 6 GB
request. Nobody sells 6 GB for BDT 46; the cheapest pack that covers it is
BDT 229, and that is the number used — and named, so it is something you can
go and buy. Two packs at BDT 366 genuinely beat the one pack that covers it
at BDT 499. Per-unit pricing is the fallback only when nothing the operator
sells is large enough, which is the ordinary case for SMS.

Repeat purchases are priced the same way (*"buy 5x -> 15 GB, 500 min, 500
SMS, BDT 1290 total"*), and `--strict` offers only packs that meet the
request in full.

## The web page

```bash
python3 scripts/web.py            # opens http://127.0.0.1:8000
python3 scripts/web.py --port 8080 --no-open
```
One page: operator and model as chips, sliders for data, minutes and SMS,
validity chips, four presets, three packs as cards, and a theme toggle that
remembers the choice. Move a slider and the results update in place — the
page fetches `api/recommend` and re-renders without a reload, keeping the
address bar in step so any state is a shareable link; with JavaScript off the
button submits the form and the server renders the same markup. Under the
cards, **what the other models would pick** answers the same request with
each model you did not choose, which is the clearest way to see them
disagree.

**Ranked by** chooses which trained model answers. They genuinely disagree —
ask Teletalk for 5 GB, 250 minutes and 50 SMS over 30 days and the decision
tree leads with the BDT 208 pack at 100% confidence while the random forest
leads with the BDT 312 pack at 54%. The bar says what it shows: *Confidence*
for the tree models (a probability), *Score* for the polynomial regression
(its output squashed into 0–1 for ranking — not a probability).

**The model picks; price orders.** The cards are the model's packs, cheapest
first — cheapest meaning the all-in cost, so a BDT 39 pack bought five times
that still leaves 125 minutes to buy costs BDT 301 and sits behind the BDT
208 pack that covers the month. The model's own favourite is therefore not
always first, and is labelled *the model's top pick*.

The cost matcher is not in the selector; it is the fallback when no model can
be loaded, and the page says so. Top-up and all-in figures come from the
catalogue rather than the model — facts about a pack, not opinions. The page
imports `recommend`, `topup` and `period_price` from `user_test.py`, so there
is one matching algorithm in the project. Served by `http.server` with no
Flask and nothing to install, bound to `127.0.0.1`.

### Putting it online

```bash
npx vercel          # https://offer-analysis.vercel.app/
npx vercel --prod   # live
```

`api/index.py` is the entry point: Vercel runs Python as serverless
functions, so there is nothing to `serve_forever` — it imports that file and
calls its `handler`, a `BaseHTTPRequestHandler`, once per request. The page
comes from `web.render_page`, so the deployed site and the local server
render from the same code. `vercel.json` rewrites every path to the function
and ships `data/`, `scripts/` and `models/` with it via `includeFiles`.

**The models are deployed as data, not as a pickle.** The obvious route does
not work: `models/offer_recommender.joblib` needs scikit-learn to read, which
drags in numpy, scipy and pandas — over 250 MB unzipped, past what a Vercel
function may be, and the forest alone wants 400 MB of memory. So
`scripts/export_models.py` writes the models as gzipped JSON:

```bash
python3 scripts/train_model.py      # fit them
python3 scripts/export_models.py    # write models/models.json.gz
```

3.2 MB, read by `ml.py` with `gzip` and `json`, predicted in pure Python — no
scikit-learn, no version to match, no `requirements.txt`. It loses nothing: a
tree is thresholds and child indices, and with `min_samples_leaf` at 5 a leaf
holds 2.12 classes on average rather than the 156 scikit-learn stores densely
at every one of 150,592 leaves. Measured on the deployed path with no ML
libraries installed: 19 ms cold start, 247 ms on the first request that
touches the models, 1 ms per prediction after. `models/models.json.gz` is
committed for this reason; the 15 MB `.joblib` stays ignored.

Any host that runs a normal process — Railway, Render, Fly, PythonAnywhere —
can skip all of this and run `python3 scripts/web.py --port $PORT`, with the
bind changed to `0.0.0.0`.

## Testing it

`train_model.py` is the scoreboard: it re-splits, refits and prints top-1,
top-3 and the BDT columns on 5,000 held-out requests. `export_models.py` is
the other check — it will not write the exported models unless they answer
identically to the scikit-learn ones (currently 400 of 400 identical picks,
scores differing by at most 2 × 10⁻¹⁶).

### ⚠️ What that accuracy does and does not mean

The catalogue records what operators sell, not what anyone should have
bought, so there is no natural ground truth. `train_model.py` samples
synthetic requests, labels each with the best pack according to an explicit
cost model, and trains the models to reproduce that labelling.

**So 92.9% measures agreement with the cost model, not real-world
correctness.** The cost model is the actual recommendation logic; each model
is a fast approximation of it. Say this plainly in any write-up. Swapping the
synthetic requests for real request-and-purchase records would make the same
script train a genuinely empirical model.

### The cost model

Everything is costed over the period requested, which is what makes short and
long packs comparable: a 7-day pack covering a 30-day request is bought 5×,
so it costs 5× and delivers 5× the quota. On top of that period price it
charges for a **shortfall** (topping up separately), mild **waste** (quota
nobody asked for), and being **locked in longer** than requested.

Three things the tuning had to get right, all caught by testing:

- **Meeting the request is a requirement, not a preference.** A proportional shortfall penalty fails: even at 80× it covered only 58% of requests a pack could have met, because a big shortfall on one axis can look cheaper than the pack that fixes it. Packs that leave the customer short now rank behind every pack that does not, lifting coverage from 22% to the 65% catalogue ceiling.
- **Shortfall still needs pricing** for when no pack can meet the request, costed at what topping up would take.
- **Unit rates must come from single-purpose packs.** A combo like "0.05 GB + 15 minutes" has a per-GB price in the thousands, because its price is really buying the minutes. Using every pack put GP's reference at ৳230/GB and ৳9.98/min — nonsense that made every option score alike. Rates use the 25th percentile of single-purpose packs of a sensible size (GP: ৳39.76/GB, ৳0.68/min).

### What the model can choose from

203 of the 287 packages. It drops `Call Rate`, `Validity`, `Bondho SIM`,
`New SIM`, `Cashback` and `Entertainment` — none is a general data/voice/SMS
bundle — and **44 app-restricted packs** whose quota only works inside
particular services ("4GB: imo + BiP + Telegram only", Hoichoi/Bongo
subscriptions). Their GB is not general internet, so offering one to somebody
who asked for 20 GB would be wrong even though the number matches.

Gaps it reports honestly rather than hides: Robi publishes no SMS packs and
no USSD codes, so an SMS request on Robi comes back with a `100 SMS short`
note and `USSD: Unknown`.
