import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import build_profiles, score_offers  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "offer_recommender.joblib")

# Deliberately not train_model's seed, so every request here is unseen.
TEST_SEED = 2024
DEFAULT_N = 5000


def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found. Run: python3 scripts/train_model.py")
    return joblib.load(MODEL_PATH)


def resolve_operator(name, known):
    aliases = {"gp": "Grameenphone", "grameenphone": "Grameenphone",
               "bl": "Banglalink", "banglalink": "Banglalink",
               "tt": "Teletalk", "teletalk": "Teletalk",
               "robi": "Robi"}
    resolved = aliases.get(str(name).strip().lower())
    if resolved in known:
        return resolved
    raise SystemExit(f"Unknown operator {name!r}. Choose from: {', '.join(known)}")


def predict(bundle, operator, wants):
    """The model's pick for one request, restricted to that operator."""
    model, catalogue = bundle["model"], bundle["catalogue"]
    data_gb, minutes, sms, validity = wants

    features = np.array([[bundle["operator_index"][operator],
                          data_gb, minutes, sms, validity]])
    probabilities = model.predict_proba(features)[0]

    sellable = set(catalogue.loc[catalogue["operator"] == operator, "offer_id"])
    ranked = sorted(
        ((offer_id, p) for offer_id, p in zip(model.classes_, probabilities)
         if offer_id in sellable),
        key=lambda kv: -kv[1])
    return ranked


def optimal(bundle, operator, wants):
    """What the cost model itself would choose -- the label, i.e. the truth.

    Returns the ranking cost (with the hard coverage rule) and the money-terms
    cost used for reporting, which is comparable across packs.
    """
    catalogue = bundle["catalogue"]
    offers = catalogue[catalogue["operator"] == operator].reset_index(drop=True)
    rates = bundle["rates"][operator]
    cost = score_offers(offers, wants, rates)
    money = score_offers(offers, wants, rates, hard=False)
    return offers, cost, money, np.argsort(cost)


def line(row, cost=None):
    data = "Unlimited" if row["category"] == "Unlimited" else f"{row['data_gb']:g} GB"
    text = (f"{row['offer_name'][:44]:46} {data:>10} | "
            f"{row['minutes']:>6g} min | {row['sms']:>6g} SMS | "
            f"{row['validity_days']:>4g}d | BDT {row['price_bdt']:>6g} | "
            f"{row['ussd_code']}")
    return f"{text}   (cost {cost:.0f})" if cost is not None else text


def run_custom(bundle, operator, wants):
    data_gb, minutes, sms, validity = wants
    print(f"\nRequest: {operator} | {data_gb:g} GB | {minutes:g} min | "
          f"{sms:g} SMS | {validity:g} days\n")

    offers, _cost, money, order = optimal(bundle, operator, wants)
    best_id = offers.loc[order[0], "offer_id"]
    best_cost = float(money[order[0]])

    ranked = predict(bundle, operator, wants)
    picked_id, confidence = ranked[0]
    picked = offers.index[offers["offer_id"] == picked_id]
    picked_cost = float(money[picked[0]]) if len(picked) else float("nan")

    print("Model pick    :", line(offers.loc[picked[0]], picked_cost),
          f"  [{confidence:.0%} confident]")
    print("Cost-model best:", line(offers.loc[order[0]], best_cost))

    if picked_id == best_id:
        print("\n-> match: the model reproduced the optimal choice.")
    else:
        regret = (picked_cost - best_cost) / best_cost * 100
        print(f"\n-> differs from optimal by {regret:.1f}% cost "
              f"({picked_cost - best_cost:+.0f} BDT-equivalent).")
        in_top3 = best_id in [offer_id for offer_id, _ in ranked[:3]]
        print(f"   optimal pack {'is' if in_top3 else 'is NOT'} in the "
              f"model's top 3.")

    print("\nModel's top 3:")
    for rank, (offer_id, p) in enumerate(ranked[:3], start=1):
        idx = offers.index[offers["offer_id"] == offer_id][0]
        print(f"  {rank}. [{p:5.1%}] {line(offers.loc[idx], float(money[idx]))}")


def run_evaluation(bundle, n):
    catalogue = bundle["catalogue"]
    rates = bundle["rates"]
    model = bundle["model"]

    print(f"\nGenerating {n} unseen requests (seed {TEST_SEED})...")
    profiles = build_profiles(catalogue, rates, n=n, seed=TEST_SEED)

    X = np.column_stack([
        profiles["operator"].map(bundle["operator_index"]).to_numpy(),
        profiles["data_wanted_gb"].to_numpy(),
        profiles["minutes_wanted"].to_numpy(),
        profiles["sms_wanted"].to_numpy(),
        profiles["validity_days"].to_numpy(),
    ])
    truth = profiles["offer_id"].to_numpy()

    probabilities = model.predict_proba(X)
    predicted = model.classes_[np.argmax(probabilities, axis=1)]
    top3 = model.classes_[np.argsort(probabilities, axis=1)[:, -3:]]

    exact = predicted == truth
    in_top3 = np.array([t in row for t, row in zip(truth, top3)])

    # How much the disagreements actually cost, and whether the pick still
    # covers the request.
    by_operator = {op: catalogue[catalogue["operator"] == op].reset_index(drop=True)
                   for op in profiles["operator"].unique()}

    def covers(offers, j, row):
        repeats = max(1.0, np.ceil(row.validity_days /
                                   max(offers.loc[j, "validity_days"], 1)))
        return bool(
            offers.loc[j, "effective_gb"] * repeats >= row.data_wanted_gb
            and offers.loc[j, "minutes"] * repeats >= row.minutes_wanted
            and offers.loc[j, "sms"] * repeats >= row.sms_wanted)

    regrets, covered, covered_best, satisfiable = [], [], [], []
    for i, row in enumerate(profiles.itertuples(index=False)):
        offers = by_operator[row.operator]
        wants = (row.data_wanted_gb, row.minutes_wanted,
                 row.sms_wanted, row.validity_days)
        cost = score_offers(offers, wants, rates[row.operator])
        # Regret is measured in ordinary money terms; the hard coverage
        # constant is a ranking device and would swamp the comparison.
        money = score_offers(offers, wants, rates[row.operator], hard=False)

        best_j = int(np.argmin(cost))
        best_cost = float(money[best_j])
        covered_best.append(covers(offers, best_j, row))

        # Could ANY pack this operator sells have covered the request? If not,
        # a miss is the catalogue's limit, not the recommender's.
        satisfiable.append(any(covers(offers, j, row) for j in offers.index))

        picked = offers.index[offers["offer_id"] == predicted[i]]
        if len(picked) == 0:      # model named another operator's pack
            regrets.append(np.nan)
            covered.append(False)
            continue
        j = picked[0]
        regrets.append((float(money[j]) - best_cost) / max(best_cost, 1e-9))
        covered.append(covers(offers, j, row))

    regrets = np.array(regrets, dtype=float)
    covered = np.array(covered)
    covered_best = np.array(covered_best)
    satisfiable = np.array(satisfiable)
    wrong_operator = int(np.isnan(regrets).sum())

    print(f"\n{'':22}{'accuracy':>10}")
    print(f"  exact match{'':10}{exact.mean():>9.1%}")
    print(f"  top-3 match{'':10}{in_top3.mean():>9.1%}")

    print("\n  Cost regret (how much worse than optimal, when it differs):")
    differing = regrets[~exact & ~np.isnan(regrets)]
    if len(differing):
        print(f"    median {np.median(differing):.1%} | "
              f"mean {np.mean(differing):.1%} | "
              f"90th pct {np.percentile(differing, 90):.1%}")
        print(f"    within 5% of optimal : "
              f"{np.mean(differing <= 0.05):.1%} of disagreements")
    print(f"    overall mean regret  : {np.nanmean(regrets):.1%} "
          f"(includes exact matches at 0%)")

    print("\n  Coverage (does the pack meet the whole request?):")
    print(f"    model's pick        : {covered.mean():>6.1%}")
    print(f"    cost model's optimal: {covered_best.mean():>6.1%}")
    print(f"    any pack could have : {satisfiable.mean():>6.1%} "
          f"<- ceiling set by the catalogue")
    if satisfiable.any():
        print(f"    model, of the satisfiable ones: "
              f"{covered[satisfiable].mean():.1%}")
    if wrong_operator:
        print(f"  Predicted another operator's pack: {wrong_operator} "
              f"(recommend.py filters these out)")

    print("\n  Per operator:")
    for operator in sorted(profiles["operator"].unique()):
        mask = (profiles["operator"] == operator).to_numpy()
        print(f"    {operator:14} n={mask.sum():>5}  "
              f"exact {exact[mask].mean():>6.1%}  "
              f"top-3 {in_top3[mask].mean():>6.1%}  "
              f"covered {covered[mask].mean():>6.1%}")

    print("\n  Example disagreements:")
    shown = 0
    for i in np.where(~exact)[0]:
        if shown >= 3:
            break
        row = profiles.iloc[i]
        offers = by_operator[row["operator"]]
        got = offers.loc[offers["offer_id"] == predicted[i]]
        want = offers.loc[offers["offer_id"] == truth[i]]
        if got.empty or want.empty:
            continue
        print(f"    {row['operator']} | {row['data_wanted_gb']:g} GB, "
              f"{row['minutes_wanted']:g} min, {row['sms_wanted']:g} SMS, "
              f"{row['validity_days']:g}d  (regret {regrets[i]:.1%})")
        print(f"      model : {got.iloc[0]['offer_name'][:52]}")
        print(f"      truth : {want.iloc[0]['offer_name'][:52]}")
        shown += 1

    print("\nAll figures measure agreement with the cost model, "
          "not real-world correctness.")


def prompt_for_request(known):
    print("Enter the request (blank = 0).")
    operator = resolve_operator(
        input(f"  operator ({'/'.join(known)}): ") or "Grameenphone", known)

    def ask(label, default):
        raw = input(f"  {label}: ").strip()
        return float(raw) if raw else default

    return operator, (ask("data wanted (GB)", 0.0),
                      ask("minutes wanted", 0.0),
                      ask("SMS wanted", 0.0),
                      ask("validity wanted (days)", 30.0))


def main():
    parser = argparse.ArgumentParser(description="Test the recommender.")
    parser.add_argument("--evaluate", action="store_true",
                        help="score the model on unseen generated requests")
    parser.add_argument("-n", type=int, default=DEFAULT_N,
                        help=f"requests to evaluate (default {DEFAULT_N})")
    parser.add_argument("--operator")
    parser.add_argument("--data", type=float, default=0.0)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--sms", type=float, default=0.0)
    parser.add_argument("--validity", type=float, default=30.0)
    args = parser.parse_args()

    bundle = load_bundle()
    known = list(bundle["operator_index"])

    if args.evaluate:
        run_evaluation(bundle, args.n)
        return

    if args.operator:
        operator = resolve_operator(args.operator, known)
        wants = (args.data, args.minutes, args.sms, args.validity)
    else:
        operator, wants = prompt_for_request(known)

    run_custom(bundle, operator, wants)


if __name__ == "__main__":
    main()
