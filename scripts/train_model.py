

import os
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOGUE = os.path.join(BASE, "data", "all_operators_official.csv")
MODEL_DIR = os.path.join(BASE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "offer_recommender.joblib")
PROFILES_PATH = os.path.join(BASE, "data", "training_profiles.csv")

N_PROFILES = 25000
RANDOM_STATE = 42

# Packs a subscriber can actually buy for data/minutes/SMS. The rest of the
# catalogue is excluded: Call Rate sells a tariff rather than a quota,
# Validity extends an account, Bondho SIM only works on a dead SIM, and
# New SIM / Cashback / Entertainment are one-off promotions.
USABLE_CATEGORIES = {"Data", "Combo", "Minute", "SMS", "Unlimited"}

# An "Unlimited" pack stores data_gb = 0 meaning uncapped; give it a large
# finite quota so it competes on price like any other pack.
UNLIMITED_GB = 1000.0

# Packs whose quota only works inside particular apps or services. Their GB
# is not general internet, so recommending one to somebody who asked for
# "20 GB" would be wrong even though the number matches.
RESTRICTED_PATTERN = (
    r"\bonly\b|youtube|tiktok|\bimo\b|\bbip\b|telegram|facebook|messenger"
    r"|hoichoi|bongo|deeptoplay|iscreen|toffee|binge|chorki|lionsgate"
    r"|sonyliv|t-sports|subscription|streaming|content|play pack"
)

# Cost-model weights.
#
# Meeting the request is treated as a requirement, not a preference. Any pack
# that leaves the customer short is pushed behind every pack that does not,
# by MISS_PENALTY, so a covering pack always wins when one exists. Tuning a
# proportional penalty instead was tried and does not work: even at 80x the
# rule covered only 58% of requests that a pack could actually have met,
# because a big shortfall on one axis can look cheaper than an expensive
# pack that fixes it.
MISS_PENALTY = 1e6

# Among packs that all fall short (no pack can meet the request), rank by how
# badly, priced at what topping up would cost.
SHORTFALL_PENALTY = 5.0
WASTE_PENALTY = 0.15      # mild dislike of paying for far more than asked
VALIDITY_PENALTY = 0.40   # mild dislike of being locked in longer than asked

VALIDITY_CHOICES = [1, 3, 7, 15, 30, 60, 90]
VALIDITY_WEIGHTS = [0.05, 0.12, 0.22, 0.11, 0.35, 0.08, 0.07]

FEATURES = ["operator", "data_wanted_gb", "minutes_wanted",
            "sms_wanted", "validity_days"]


def load_catalogue():
    df = pd.read_csv(CATALOGUE)
    df = df[df["category"].isin(USABLE_CATEGORIES)].copy()

    restricted = df["offer_name"].str.contains(
        RESTRICTED_PATTERN, case=False, regex=True, na=False)
    df = df[~restricted].copy()

    df["effective_gb"] = np.where(
        df["category"] == "Unlimited", UNLIMITED_GB, df["data_gb"])

    # A pack with no published validity cannot be matched on renewal cycle;
    # treat it as monthly, the commonest cycle, rather than dropping it.
    df.loc[df["validity_days"] <= 0, "validity_days"] = 30.0

    return df.reset_index(drop=True)


def reference_rates(catalogue):
    """What one more GB / minute / SMS costs, per operator.

    Used to price a shortfall, so it must reflect what topping up actually
    costs. Two things would otherwise wreck it:

      - Mixed packs. A combo like "0.05 GB + 15 minutes" has a per-GB price
        of thousands, because its price is really buying the minutes. Only
        single-purpose packs are used, so each rate prices one resource.
      - Micro packs. Grameenphone lists many small content packs, which drags
        its median to ~৳95/GB against a real bulk rate near ৳10-40. The 25th
        percentile of a decent-sized pack approximates the good-value rate a
        customer would actually top up at.

    Robi and Banglalink publish no standalone SMS packs, so their SMS rate
    falls back to the all-operator figure.
    """
    def rate(group, quantity_col, price_col, other_cols, minimum):
        pure = group[(group[quantity_col] >= minimum)
                     & (group[other_cols[0]] == 0)
                     & (group[other_cols[1]] == 0)]
        return float(pure[price_col].quantile(0.25)) if len(pure) else None

    everything = catalogue
    fallbacks = (
        rate(everything, "data_gb", "price_per_gb", ("minutes", "sms"), 1) or 30.0,
        rate(everything, "minutes", "price_per_minute", ("data_gb", "sms"), 30) or 1.0,
        rate(everything, "sms", "price_per_sms", ("data_gb", "minutes"), 50) or 0.3,
    )

    rates = {}
    for operator, group in catalogue.groupby("operator"):
        rates[operator] = (
            rate(group, "data_gb", "price_per_gb", ("minutes", "sms"), 1) or fallbacks[0],
            rate(group, "minutes", "price_per_minute", ("data_gb", "sms"), 30) or fallbacks[1],
            rate(group, "sms", "price_per_sms", ("data_gb", "minutes"), 50) or fallbacks[2],
        )
    return rates


def score_offers(offers, wants, rates, hard=True):
    """Cost of covering one request with each offer. Lowest wins.

    Everything is costed over the period the customer asked for, which is
    what makes short and long packs comparable. A 3-day pack asked to cover
    30 days has to be bought ten times, so it is priced ten times -- and it
    delivers ten times the quota, which is why a cheap short pack can still
    win when it is genuinely good value.

    On top of that period price:
      - a shortfall has to be topped up separately, at a premium;
      - a large surplus is money spent on quota that was not asked for;
      - a pack outlasting the request ties up money for longer than needed.
    """
    data_want, minute_want, sms_want, validity = wants
    gb_rate, min_rate, sms_rate = rates
    validity = max(validity, 1)

    pack_validity = offers["validity_days"].to_numpy()
    repeats = np.ceil(validity / pack_validity)
    repeats = np.clip(repeats, 1, None)

    price = offers["price_bdt"].to_numpy() * repeats
    gb = offers["effective_gb"].to_numpy() * repeats
    mins = offers["minutes"].to_numpy() * repeats
    sms = offers["sms"].to_numpy() * repeats

    short_gb = np.clip(data_want - gb, 0, None)
    short_min = np.clip(minute_want - mins, 0, None)
    short_sms = np.clip(sms_want - sms, 0, None)

    shortfall = SHORTFALL_PENALTY * (
        short_gb * gb_rate + short_min * min_rate + short_sms * sms_rate)

    # A pack that leaves the customer short ranks behind every pack that does
    # not, whatever the price difference. Turn `hard` off to get the cost in
    # ordinary money terms -- what evaluation needs to measure how much a
    # wrong choice really costs, without this ranking constant swamping it.
    if hard:
        misses = (short_gb > 0) | (short_min > 0) | (short_sms > 0)
        shortfall = shortfall + misses * MISS_PENALTY

    # Surplus is capped at the pack's own price, so an unlimited pack is not
    # penalised out of all proportion for being large.
    waste = WASTE_PENALTY * np.minimum(
        np.clip(gb - data_want, 0, None) * gb_rate
        + np.clip(mins - minute_want, 0, None) * min_rate
        + np.clip(sms - sms_want, 0, None) * sms_rate,
        price,
    )

    # Only over-long packs are penalised here; under-long ones already paid
    # for it through repeats.
    excess = np.clip(pack_validity / validity - 1, 0, None)
    validity_cost = VALIDITY_PENALTY * np.log1p(excess) * price

    return price + shortfall + waste + validity_cost


def build_profiles(catalogue, rates, n=N_PROFILES, seed=RANDOM_STATE):
    """Synthetic requests, each labelled with its best-matching pack."""
    rng = np.random.default_rng(seed)
    operators = sorted(catalogue["operator"].unique())
    by_operator = {op: catalogue[catalogue["operator"] == op].reset_index(drop=True)
                   for op in operators}

    rows = []
    for _ in range(n):
        operator = str(rng.choice(operators))

        # Lognormal: most requests are modest, with a long tail of heavy ones.
        # A third of requests ask for no minutes or no SMS at all, which is
        # what makes data-only and combo packs both reachable.
        data_want = float(np.clip(rng.lognormal(1.2, 1.0), 0.0, 120))
        minute_want = float(np.clip(rng.lognormal(4.2, 1.1), 0, 2000))
        sms_want = float(np.clip(rng.lognormal(3.5, 1.3), 0, 3000))
        if rng.random() < 0.35:
            minute_want = 0.0
        if rng.random() < 0.45:
            sms_want = 0.0
        if rng.random() < 0.15:
            data_want = 0.0

        validity = int(rng.choice(VALIDITY_CHOICES, p=VALIDITY_WEIGHTS))

        offers = by_operator[operator]
        wants = (data_want, minute_want, sms_want, validity)
        cost = score_offers(offers, wants, rates[operator])

        rows.append({
            "operator": operator,
            "data_wanted_gb": round(data_want, 3),
            "minutes_wanted": round(minute_want, 1),
            "sms_wanted": round(sms_want, 1),
            "validity_days": validity,
            "offer_id": offers.loc[int(np.argmin(cost)), "offer_id"],
        })
    return pd.DataFrame(rows)


def build_models():
    """The three models trained and compared, all on the same features.

    Decision tree           one tree, so the rules it learned can be read.
    Random forest           300 of those trees, voting.
    Polynomial regression   every feature, every square and every pairwise
                            product (degree 2), then one least-squares
                            regression per pack, standardised first because
                            the raw features run from 0 to 3000 and squaring
                            that gives 9,000,000. The pack whose regression
                            predicts highest wins, which is what
                            RidgeClassifier does: least squares against +/-1
                            targets, argmax at predict time.
    """
    return {
        "Decision tree": DecisionTreeClassifier(
            min_samples_leaf=2, random_state=RANDOM_STATE),
        # 150 trees, leaves of 5, because a forest stores a probability
        # for all 158 packs at every leaf and that is what drives its size:
        # 300 trees with leaves of 2 scores 93.5%/98.8% but weighs 1.3 GB in
        # memory, which no web process should carry. These settings give up
        # 1.9 points of top-1 for a model a fifth the size. Measured:
        #
        #   trees  leaf   top-1   top-3   in memory   on disk
        #     300     2   93.5%   98.8%     1333 MB     41 MB
        #     150     5   91.6%   97.7%      395 MB     15 MB
        #     100     8   89.9%   97.0%      191 MB      9 MB
        #      60    12   87.3%   96.4%       86 MB      4 MB
        "Random forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1),
        "Polynomial regression": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE)),
    }


def class_scores(model, X):
    """One score per pack per request, however the model expresses it.

    predict_proba for the two tree models, the regression's own output for
    the polynomial one. Only the order matters -- these are ranked, never
    read as probabilities.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def sellable_masks(model, catalogue, operator_index):
    """For each operator, which of this model's classes it actually sells.

    Without this a model can answer a Robi request with a Grameenphone pack,
    which is not a recommendation anybody can act on. Every model gets the
    same restriction, so the comparison stays fair.
    """
    classes = np.asarray(model.classes_)
    return {code: np.isin(classes,
                          catalogue.loc[catalogue["operator"] == operator,
                                        "offer_id"].to_numpy())
            for operator, code in operator_index.items()}


def ranked_ids(model, X, masks, top=3):
    """Each request's top packs, restricted to the operator it asked for."""
    scores = class_scores(model, X)
    classes = np.asarray(model.classes_)
    picks = []
    for row, code in zip(scores, X[:, 0].astype(int)):
        masked = np.where(masks[code], row, -np.inf)
        picks.append(classes[np.argsort(masked)[::-1][:top]])
    return np.array(picks)


def money_lost(catalogue, rates, operators, X, picks, labels):
    """What the model's pick costs over the pack it should have chosen.

    Accuracy alone says a model was wrong; it does not say whether being
    wrong cost the customer 5 taka or 500. This prices it: the model's pack
    against the labelled pack for the same request, both in money terms (the
    hard coverage rule off, so the ranking constant a shortfall carries does
    not swamp the difference). A correct pick scores 0.

    A negative figure means the model picked something cheaper than the
    label, which happens because the label is chosen coverage-first and the
    cheaper pack leaves the customer short of something they asked for.
    """
    by_id = catalogue.set_index("offer_id")
    extra = []
    for features, pick, label in zip(X, picks, labels):
        operator = operators[int(features[0])]
        wants = tuple(features[1:])
        pair = by_id.loc[[pick, label]].reset_index()
        costs = score_offers(pair, wants, rates[operator], hard=False)
        extra.append(float(costs[0]) - float(costs[1]))
    return float(np.mean(extra)), float(np.percentile(extra, 90))


def main():
    catalogue = load_catalogue()
    rates = reference_rates(catalogue)
    print(f"Catalogue: {len(catalogue)} buyable packs across "
          f"{catalogue['operator'].nunique()} operators")

    profiles = build_profiles(catalogue, rates)
    profiles.to_csv(PROFILES_PATH, index=False)
    print(f"Requests : {len(profiles)} synthetic -> {PROFILES_PATH}")
    print(f"           {profiles['offer_id'].nunique()} distinct packs ever "
          f"recommended (of {len(catalogue)})")

    operators = sorted(catalogue["operator"].unique())
    operator_index = {op: i for i, op in enumerate(operators)}

    X = np.column_stack([
        profiles["operator"].map(operator_index).to_numpy(),
        profiles["data_wanted_gb"].to_numpy(),
        profiles["minutes_wanted"].to_numpy(),
        profiles["sms_wanted"].to_numpy(),
        profiles["validity_days"].to_numpy(),
    ])
    y = profiles["offer_id"].to_numpy()

    # Stratifying keeps rare packs in both halves, but needs >=2 examples of
    # every class; fall back to a plain split if some pack was chosen once.
    counts = profiles["offer_id"].value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)

    trained, rows = {}, []
    for name, model in build_models().items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        masks = sellable_masks(model, catalogue, operator_index)
        top3 = ranked_ids(model, X_test, masks, top=3)
        top1 = top3[:, 0]
        accuracy = float(np.mean(top1 == y_test))
        top3_accuracy = float(np.mean([t in row for t, row in zip(y_test, top3)]))
        mean_lost, p90_lost = money_lost(catalogue, rates, operators,
                                         X_test, top1, y_test)

        trained[name] = model
        rows.append((name, accuracy, top3_accuracy, mean_lost, p90_lost,
                     fit_seconds))

    print(f"\nHow the three models compare, on {len(y_test):,} requests none "
          f"of them was trained on:\n")
    print(f"  {'Model':24} {'Top-1':>7} {'Top-3':>7} {'Mean BDT':>10} "
          f"{'90th BDT':>10} {'Fit (s)':>9}")
    for name, acc, top3_acc, mean_lost, p90_lost, seconds in rows:
        print(f"  {name:24} {acc:6.1%} {top3_acc:6.1%} {mean_lost:10.2f} "
              f"{p90_lost:10.2f} {seconds:9.1f}")
    print("\n  Top-1/Top-3 : how often the pick, or the top three, contain "
          "the pack the cost\n                model chose. This is "
          "agreement with that cost model -- the label\n                "
          "-- not real-world correctness.")
    print("  BDT columns : what the pick costs over the pack it should have "
          "chosen, for the\n                same request, averaged and at "
          "the 90th percentile. A correct\n                pick scores 0. "
          "Money is the honest measure: a wrong pack that\n                "
          "costs the same is not a bad answer.")

    forest = trained["Random forest"]
    print("\nFeature importance (random forest):")
    for name, importance in sorted(zip(FEATURES, forest.feature_importances_),
                                   key=lambda kv: -kv[1]):
        print(f"  {name:16} {importance:.3f}")

    metrics = {name: {"accuracy": acc, "top3_accuracy": top3_acc,
                      "mean_bdt_lost": mean_lost, "p90_bdt_lost": p90,
                      "fit_seconds": seconds}
               for name, acc, top3_acc, mean_lost, p90, seconds in rows}

    os.makedirs(MODEL_DIR, exist_ok=True)
    # The pack list each model's scores are in the order of. Stored here
    # rather than read back off the estimator, because a model pickled by one
    # scikit-learn and loaded by another can fail to rebuild the attribute it
    # keeps them in -- RidgeClassifier.classes_ is a property over an
    # internal label binariser, and 1.6 -> 1.9 breaks it. The order is the
    # same for all three: sorted, from the labels they were all trained on.
    joblib.dump({
        "models": trained,
        "classes": {name: [str(c) for c in np.asarray(model.classes_)]
                    for name, model in trained.items()},
        "sklearn_version": sklearn.__version__,
        "model": forest,          # the default, and what older scripts expect
        "operator_index": operator_index,
        "catalogue": catalogue,
        "rates": rates,
        "feature_names": FEATURES,
        "metrics": metrics,
        "test_size": int(len(y_test)),
    }, MODEL_PATH, compress=3)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"\nSaved all three -> {MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
