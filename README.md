# Bangladesh SIM Offer Dataset

Live offer data for the four operators that publish it — Grameenphone,
Banglalink, Teletalk and Robi — scraped from each operator's own site and
normalised into one schema. Built for a model that recommends the best
package for a user given their usage.

Every price, quota, validity and USSD code here comes from the operator.
Nothing is estimated or filled in.

## Start here

**`data/all_operators_official.csv`** — 287 packages, 14 columns, no blank
cells, every row a domestic package with a real price (৳7–৳2,997).

| Operator | Packages |
|---|---|
| Grameenphone | 93 |
| Teletalk | 86 |
| Banglalink | 80 |
| Robi | 28 |
| **Total** | **287** |

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
| `price_per_gb`, `price_per_minute`, `price_per_sms` | ৳ per unit, derived; `0` when the pack has no such quota |
| `price_bdt` | pack price in taka |

### Category notes

Most categories are ordinary bundles. Four behave differently and will
mislead a model if treated as plain packs:

- **`Call Rate` (25)** — buys a cheaper per-minute tariff, not minutes. `minutes = 0`, `price_bdt` is the required recharge, and the value is in `price_per_minute` (e.g. ৳0.60 for a 1-paisa/sec offer).
- **`Unlimited` (6)** — `data_gb = 0` means uncapped, not empty. Exclude these before computing any ৳/GB statistic or they read as free.
- **`Validity` (8)** — extends account validity; usually carries no quota.

## The 4x expanded table

**`data/all_operators_expanded.csv`** — 1,137 rows, the same 14 columns:
the 287 real rows unchanged, plus **850 synthetic siblings**, three per real
pack. For a bigger training set only; anything that is meant to be true about
what operators sell comes from `all_operators_official.csv`.

```bash
python3 scripts/expand_dataset.py    # -> data/all_operators_expanded.csv
```

Nothing is invented from scratch. Each sibling is a real pack resized: the
quota is scaled by one of 0.5–3x, the price by that factor to the power 0.85
(bigger packs cost less per unit), and the label rewritten in the operator's
own wording — `5 GB+250 minute+300 SMS` becomes `10 GB+500 minute+300 SMS`.
Nothing leaves the range that operator publishes: quota is capped at its
largest real pack, price clamped to its real ৳ range, validity moved only to
the neighbouring term it already sells in that category. Packs whose numbers
are a tariff rather than a quota (`Call Rate`, `Validity`, `Bondho SIM`,
`New SIM`, `Cashback`, `Unlimited`, `Entertainment`) keep their quota and
term and vary only in price. The generator is seeded, so the file rebuilds
identically.

Median ৳/unit lands where the real catalogue already is:

| | GP | Banglalink | Teletalk | Robi |
|---|---|---|---|---|
| ৳/GB, data packs ≥5 GB — real | 32.98 | 15.33 | 10.84 | 7.98 |
| ৳/GB, data packs ≥5 GB — synthetic | 36.23 | 13.68 | 10.22 | 9.16 |
| ৳/minute, minute packs — real | 0.75 | 0.71 | 0.64 | 0.73 |
| ৳/minute, minute packs — synthetic | 0.76 | 0.68 | 0.61 | 0.74 |

**Telling the two apart.** The schema is unchanged, so synthetic rows are
marked inside it: `offer_id` ends in `-V2`/`-V3`/`-V4` (Teletalk and Robi
number packs sequentially, so theirs continue past the real range — `TT-087`
and up, `ROBI-029` and up), and `ussd_code` is always `Unknown`, because a
made-up dial code is a dialable one. The first 287 rows are the official file
byte for byte.

**What it is not.** These packs are not for sale and no operator has ever
published them. They are plausible, not real — use them to train, never to
quote a price. `README` figures, per-operator comparisons and anything a user
sees should keep coming from the official table.

To train on it, point `CATALOGUE` in `scripts/train_model.py` at
`all_operators_expanded.csv`; 824 of the 1,137 rows survive the model's
category and app-restriction filters, against 203 of 287.

## Per-operator files

The four sources keep **17 columns** — the 14 above plus `speed_cap_mbps`,
`roaming` and `auto_renewal`, which the merge drops (see below).

| File | Rows | Source | How |
|---|---|---|---|
| `data/gp_offers_official.csv` | 112 | grameenphone.com | Pack table + tariff pages + 11 offer pages, each with its own adapter |
| `data/bl_offer_official.csv` | 97 | banglalink.net | Its public JSON API — packs arrive already structured |
| `data/tt_offer_official.csv` | 96 | teletalk.com.bd | Plain server-rendered HTML tables (English pages; the Bengali ones use Bengali numerals) |
| `data/robi_offer_official.csv` | 28 | robi.com.bd | Headless browser — the site is client-rendered with no public API |

`data/official_raw/` holds source captures for auditing, including
`tt_internet_raw.csv` (Teletalk's table verbatim, with its section banners
and original column order). `data/kaggle_raw/` holds a 2023 Kaggle snapshot
of all five operators, kept for historical comparison — it is not part of
the scraped pipeline.

## What the merge does

`scripts/merge_official.py` concatenates the four operator files and applies
three corrections. All are reproducible; the per-operator files are never
modified.

**1. Drops three near-empty columns.**

| Dropped | Why |
|---|---|
| `roaming` | exactly equivalent to `category == "Roaming"` (22 of 22) |
| `speed_cap_mbps` | 0 in 328 of 333 rows |
| `auto_renewal` | `Unknown` in 275 of 333 — only Grameenphone publishes it |

**2. Drops 7 non-purchasable entries** (all Banglalink). Its "Others" tab
mixes service and marketing pages in with real packs — eSIM, Emergency
Balance, Balance Lapse Period, handset instalments. They have no price, no
quota, no validity and no tariff. The test is about content rather than
category, so rows that are legitimately price-0 but sell a rate survive it.

**3. Relabels 4 miscategorised packs to `Validity`.** Banglalink files its
main-account validity packs under Voice and Robi files its multi-year
Shodesh packs under Bundles, though neither contains minutes or data.

**4. Excludes 22 roaming packs.** Only Grameenphone (12) and Banglalink (10)
publish priced roaming; Teletalk documents it as eligibility rules and a
partner list, and Robi hides its packs behind a country picker. Kept in, they
would teach a model that two operators offer roaming and two do not — a
publishing gap, not a real difference — and they are priced for a different
job than domestic packs (up to ৳9,994 against a ৳2,997 domestic maximum).
The table is therefore **domestic packages only**. Roaming rows remain in
`gp_offers_official.csv` and `bl_offer_official.csv` if you need them.

It also splits **17 tariff plans** into `tariff_plans_official.csv`. A tariff
plan is not a package — it is the rate you are on, sells no quota, and 11 of
17 have no published joining fee, so in the pack table they looked like free
offers a recommender could rank top. Their value is `price_per_minute` /
`price_per_sms` (Teletalk Gen-Z ৳0.50/min vs GP Nishchinto ৳2.00/min). Join
on `operator`. The script asserts no package survives at price 0.

## Two traps before comparing operators

**1. Robi's 28 rows are a curated sample, not a small catalogue.**
robi.com.bd personalises its public listing; the full list needs a login with
a Robi number. Two tabs return nothing to a signed-out visitor — Watch Pack
("No results found") and Roaming (gated behind a country picker). Every Robi
row is real, but any "Robi is cheaper/has less" conclusion is unsupported.

**2. Median ৳/GB across all Data packs is misleading.** Run naively it says
Grameenphone charges 13× Teletalk. That is mostly *pack mix* — GP lists many
small streaming/content packs (0.08–1 GB), and small packs always cost more
per GB. Compare like sizes:

| Operator | All data packs | Packs ≥ 5 GB |
|---|---|---|
| Grameenphone | ৳171.14 (n=39) | ৳32.98 (n=9) |
| Banglalink | ৳19.60 (n=29) | ৳15.33 (n=23) |
| Teletalk | ৳13.05 (n=38) | ৳10.84 (n=19) |
| Robi | ৳20.33 (n=2) | ৳7.98 (n=1) |

GP is still the most expensive, but ~3× Teletalk rather than ~13×.

(The third trap — uneven roaming coverage — is handled by excluding roaming
from this table entirely. See merge step 4.)

## Regenerating

```bash
python3 scripts/scrape_grameenphone.py   # -> gp_offers_official.csv
python3 scripts/scrape_banglalink.py     # -> bl_offer_official.csv
python3 scripts/scrape_teletalk.py       # -> tt_offer_official.csv
python3 scripts/scrape_robi.py           # -> robi_offer_official.csv
python3 scripts/merge_official.py        # -> all_operators_official.csv
```

Operators change prices often, so re-scrape before relying on the numbers.
Only the Robi scraper needs extra setup:

```bash
python3 -m pip install playwright && python3 -m playwright install chromium
```

`scripts/generate_dataset.py` and `scripts/merge_datasets.py` are from the
earlier synthetic/Kaggle stage of the project. `merge_datasets.py` no longer
runs — the cleaned Kaggle file it reads has been deleted — and neither feeds
the current pipeline. They can be removed.

## The recommendation model

```bash
python3 scripts/train_model.py                 # trains and saves the model
python3 scripts/recommend.py --operator Teletalk \
    --data 5 --minutes 250 --sms 50 --validity 30
```

Five inputs — operator, data (GB), minutes, SMS, validity (days) — and it
returns the best-matching pack with its data, minutes, SMS, price and USSD
code, plus two alternatives and a confidence.

```
Best match:
  5 GB+250 minute+300 SMS
    operator   : Teletalk
    data       : 5 GB      minutes : 250      SMS : 300
    validity   : 30 days   price   : BDT 208  USSD: *111*104#
    confidence : 68%
```

`RandomForestClassifier` over `offer_id`, 25,000 training requests.
Predictions are filtered to the chosen operator's packs, so it can never
suggest another operator's offer.

## Trying it on your own input

```bash
python3 scripts/user_test.py
```

Asks the five questions and prints the recommendation. Blank leaves a field
at 0 (validity defaults to 30 days), `q` quits, and it loops so you can try
several requests in a row. Pass the request as flags instead if the run has
no keyboard (an IDE "Run" button often gives only an output pane):

```bash
python3 scripts/user_test.py --operator Teletalk \
    --data 5 --minutes 250 --sms 50 --validity 30
```

This one **does not use the model**. It ranks the operator's packs by
matching alone, so it needs no `models/offer_recommender.joblib` and no
joblib, numpy, pandas or scikit-learn: the standard library and the CSV are
enough. `Value` is a cost ratio, not a model probability — see below.

```
  Operator [1 Grameenphone / 2 Banglalink / 3 Teletalk / 4 Robi]: 3
  Data wanted (GB)     : 5
  Minutes wanted       : 250
  SMS wanted           : 50
  Validity (days) [30] : 30

RECOMMENDED
  10 GB+350 minute+100 SMS
    Operator   : Teletalk       Data    : 10 GB
    Minutes    : 350            SMS     : 100
    Validity   : 30 days        Price   : BDT 312
    USSD       : *111*105#      Value   : 54%
```

**What it ranks on: the all-in cost.** Not the pack price — what the whole
request costs if you buy that pack: the pack over the period you asked for,
plus the price of buying separately whatever it leaves out. A pack that
covers the request therefore wins only when it is cheaper than a smaller
pack plus the top-up, and `Value` is the ratio between those totals, best
in the list at 100%.

That is what lets a pack that misses on one thing still be the right
answer. Ask Banglalink for 10 GB **and 50 SMS**: Banglalink publishes no
SMS pack at all, so the old rule threw every pack out and picked by
shortfall. Now the 50 SMS are priced at what 50 SMS cost — `SMS_TOPUP_BDT`,
BDT 1 each — and the 10 GB pack wins on the arithmetic, which is printed:

```
RECOMMENDED
  Banglalink Prepaid 10GB Tk. 398!
    ...
    Value      : 100%
    Top up     : 50 SMS short -> pay as you go at BDT 1, BDT 50
    All in     : BDT 448  (Banglalink sells nothing that covers this)
```

**A gap is priced at the pack that fills it**, not at an average rate. This
matters more than it sounds. Banglalink data averages BDT 7.52/GB, so a
per-unit rate makes a 6 GB gap look like BDT 46 — and a BDT 137 minute pack
looks like a fine answer to a 6 GB request. Nobody sells 6 GB for BDT 46.
The cheapest Banglalink pack that really covers it is BDT 229, and that is
the number used, and named:

```
    Top up     : 6 GB short -> add "Banglalink Prepaid 30GB Tk. 229!", BDT 229
    All in     : BDT 366
```

Two packs for BDT 366 genuinely beats the one pack that covers it at BDT
499, so that is what it says — with the second pack named, so it is
something you can actually go and buy. Only when the operator sells nothing
large enough to fill a gap does it fall back to a per-unit price, which is
the ordinary case for SMS.

It also prices repeat purchases: if the best pack is shorter than the
period you asked for, *"To cover 30 days: buy 5x -> 15 GB, 500 min, 500
SMS, BDT 1290 total"*. And `--strict` turns all of this off — it goes back
to the hard rule, offering only packs that meet the request in full,
whatever a top-up would have cost.

## The web page

```bash
python3 scripts/web.py            # opens http://127.0.0.1:8000
python3 scripts/web.py --port 8080 --no-open
```

One page: the same five inputs as a form, the same three packs as cards,
then seven more in a table under them &mdash; ten in all, ranked the same
way, with each row showing what it gives, what it costs, what filling its
gaps would add and the all-in total. `CARDS` and `MORE` at the top of the
file set how many of each. The same arithmetic runs underneath. It imports `recommend`, `topup` and
`period_price` straight out of `user_test.py`, so there is one matching
algorithm in this project and the page cannot drift away from the command
line — change the matcher and both change together.

Standard library only, like the matcher: `http.server`, no Flask, nothing to
install. The page is rendered on the server and submitted as a plain form,
so it needs no JavaScript, and the server binds to `127.0.0.1`, so nothing
outside your machine can reach it. The "only packs that meet the request in
full" checkbox is `--strict`.

### Putting it online

It deploys to Vercel as it stands:

```bash
npx vercel          # preview URL
npx vercel --prod   # live
```

`api/index.py` is the entry point. Vercel runs Python as serverless
functions rather than long-lived servers, so there is nothing to
`serve_forever` — it imports that file and calls its `handler` class once
per request. That class is a `BaseHTTPRequestHandler`, which is what the
runtime expects, and the page itself comes from `web.render_page`, so the
deployed site and `python3 scripts/web.py` render from the same code.

`vercel.json` does the two things the deployment needs: it rewrites every
path to the function, and it ships `data/` and `scripts/` with it —
`includeFiles`, without which the function would start with no catalogue to
read. There are no dependencies to install, so there is no
`requirements.txt`. The catalogue is parsed once per cold start, not once
per request.

Anywhere that runs a normal process — Railway, Render, Fly, PythonAnywhere
— can skip all of this and run `python3 scripts/web.py --port $PORT` as-is,
minus the `127.0.0.1` bind, which would have to become `0.0.0.0` for
traffic to reach it from outside the machine.

## Testing it

```bash
python3 scripts/test.py --evaluate            # accuracy on unseen requests
python3 scripts/test.py --operator Teletalk \
    --data 5 --minutes 250 --sms 50 --validity 30   # one custom request
python3 scripts/test.py                       # prompts for the five inputs
```

`user_test.py` is the matching algorithm on its own; `test.py` is for judging
the model — it also shows the cost model's optimal pick alongside, so you can
see when the two disagree.

Custom mode shows the model's pick beside the cost model's optimal one, and
says how far apart they are:

```
Model pick     : 10 GB+350 minute+100 SMS   BDT 312  (cost 330)  [54% confident]
Cost-model best:  5 GB+250 minute+300 SMS   BDT 208  (cost 213)
-> differs from optimal by 55.1% cost (+117 BDT-equivalent).
   optimal pack is in the model's top 3.
```

`--evaluate` scores 5,000 freshly generated requests under a different seed,
so none was seen in training:

| Metric | Result |
|---|---|
| Exact match | **94.5%** |
| Top-3 match | **98.8%** |
| Median cost regret when it disagrees | 11.5% |
| Mean regret overall | 1.6% |
| **Requests fully covered** | **63.4%** (ceiling 65.0%) |
| Covered, of requests any pack could satisfy | **97.6%** |

Exact match is the harshest reading, so the script also reports two metrics
that say whether a disagreement matters:

- **Cost regret** — how much worse the pick is, in money terms. Half the disagreements cost under 11.5%, and the optimal pack is nearly always still in the top 3.
- **Coverage** — how often the pack actually meets the whole request. This is the number to quote: of requests that *any* pack could satisfy, the model covers **97.6%**. The gap between 63.4% and 65.0% is not model error — 35% of randomly generated requests are impossible for that operator (2,000 SMS on Robi, which sells none; 100 GB in one day).

Per-operator coverage varies with catalogue depth, not model quality:
Teletalk 94.7%, Grameenphone 74.5%, Banglalink 44.4%, Robi 41.6%.

### ⚠️ What that accuracy does and does not mean

The catalogue records what operators sell, not what anyone should have
bought, so there is no natural ground truth. `train_model.py` samples
synthetic requests, labels each with the best pack according to an explicit
cost model, and trains the classifier to reproduce that labelling.

**So 94.5% measures agreement with the cost model, not real-world
correctness.** The cost model is the actual recommendation logic; the
classifier is a fast approximation of it. Say this plainly in any write-up.
Swapping the synthetic requests for real request-and-purchase records would
make the same script train a genuinely empirical model.

### The cost model

Everything is costed over the period requested, which is what makes short
and long packs comparable. A 7-day pack covering a 30-day request is bought
5×, so it costs 5× and delivers 5× the quota — the recommender reports that
explicitly ("buy 5x -> 10 GB, 400 min, 500 SMS, BDT 1085 total"). On top of
the period price it charges for a **shortfall** (topping up separately, at a
premium), mild **waste** (quota nobody asked for), and being **locked in
longer** than requested.

Three things the tuning had to get right, all caught by testing rather than
assumed:

- **Meeting the request is a requirement, not a preference.** A proportional shortfall penalty does not work: even at 80× the rule covered only 58% of requests a pack could actually have met, because a big shortfall on one axis can look cheaper than the pack that fixes it. Packs that leave the customer short are now ranked behind every pack that does not, which lifted coverage from 22% to the 65% catalogue ceiling.
- **Shortfall still needs pricing.** When no pack can meet the request, the ranking falls back to how badly each one misses, costed at what topping up would take.
- **Unit rates must come from single-purpose packs.** A combo like "0.05 GB + 15 minutes" has a per-GB price in the thousands, because its price is really buying the minutes. Using every pack put Grameenphone's reference at ৳230/GB and ৳9.98/min — nonsense that made every option score alike. Rates now use the 25th percentile of single-purpose packs of a sensible size (GP: ৳39.76/GB, ৳0.68/min).

### What the model can choose from

203 of the 287 packages. It drops `Call Rate`, `Validity`, `Bondho SIM`,
`New SIM`, `Cashback` and `Entertainment` — none is a general data/voice/SMS
bundle — and **44 app-restricted packs** whose quota only works inside
particular services ("4GB: imo + BiP + Telegram only", "YouTube + TikTok
only", Hoichoi/Bongo subscriptions). Their GB is not general internet, so
offering one to somebody who asked for 20 GB would be wrong even though the
number matches.

Known gaps it will report honestly rather than hide: Robi publishes no SMS
packs and no USSD codes, so an SMS request on Robi comes back with a
`100 SMS short` note and `USSD: Unknown`.
