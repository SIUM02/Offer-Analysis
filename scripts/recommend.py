"""
Recommends a pack for one request, using the trained model.

    python3 scripts/recommend.py --operator Grameenphone \
        --data 10 --minutes 300 --sms 100 --validity 30

Prints the best match plus two runners-up, each with its data, minutes, SMS,
price and USSD code.

The model can only ever suggest a pack the chosen operator actually sells, so
its predictions are filtered to that operator before ranking. If the request
cannot be met exactly -- asking Robi for SMS, say, when Robi publishes no SMS
packs -- the closest pack is still returned and the gap is reported.

Requires models/offer_recommender.joblib (python3 scripts/train_model.py).
"""

import argparse
import math
import os

import joblib
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "offer_recommender.joblib")


def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found. Run: python3 scripts/train_model.py")
    return joblib.load(MODEL_PATH)


def resolve_operator(name, known):
    """Accept 'gp', 'robi', 'banglalink' etc. case-insensitively."""
    aliases = {"gp": "Grameenphone", "grameenphone": "Grameenphone",
               "bl": "Banglalink", "banglalink": "Banglalink",
               "tt": "Teletalk", "teletalk": "Teletalk",
               "robi": "Robi"}
    resolved = aliases.get(name.strip().lower())
    if resolved in known:
        return resolved
    raise SystemExit(f"Unknown operator {name!r}. Choose from: {', '.join(known)}")


def recommend(bundle, operator, data_gb, minutes, sms, validity, top=3):
    model = bundle["model"]
    catalogue = bundle["catalogue"]

    features = np.array([[
        bundle["operator_index"][operator], data_gb, minutes, sms, validity,
    ]])
    probabilities = model.predict_proba(features)[0]

    # Restrict to packs this operator actually sells, so the model can never
    # return another operator's pack even if its probabilities blur across.
    sellable = set(catalogue.loc[catalogue["operator"] == operator, "offer_id"])
    ranked = [(offer_id, p) for offer_id, p
              in zip(model.classes_, probabilities) if offer_id in sellable]
    ranked.sort(key=lambda kv: -kv[1])

    results = []
    for offer_id, confidence in ranked[:top]:
        row = catalogue.loc[catalogue["offer_id"] == offer_id].iloc[0]
        results.append((row, float(confidence)))
    return results


def describe(row, confidence, wants):
    data_gb, minutes, sms, validity = wants
    unlimited = row["category"] == "Unlimited"

    print(f"  {row['offer_name']}")
    print(f"    operator   : {row['operator']}")
    print(f"    data       : {'Unlimited' if unlimited else f'{row.data_gb:g} GB'}")
    print(f"    minutes    : {row['minutes']:g}")
    print(f"    SMS        : {row['sms']:g}")
    print(f"    validity   : {row['validity_days']:g} days")
    print(f"    price      : BDT {row['price_bdt']:g}")
    print(f"    USSD       : {row['ussd_code']}")
    print(f"    confidence : {confidence:.0%}")

    # A pack shorter than the requested period is meant to be re-bought, and
    # the model scored it that way, so report the totals over the whole
    # period rather than a single purchase.
    repeats = max(1, math.ceil(validity / max(row["validity_days"], 1)))
    if repeats > 1:
        print(f"    to cover {validity:g} days: buy {repeats}x -> "
              f"{'Unlimited' if unlimited else f'{row.data_gb * repeats:g} GB'}, "
              f"{row['minutes'] * repeats:g} min, {row['sms'] * repeats:g} SMS, "
              f"BDT {row['price_bdt'] * repeats:g} total")

    gaps = []
    if not unlimited and row["data_gb"] * repeats < data_gb:
        gaps.append(f"{data_gb - row['data_gb'] * repeats:g} GB short")
    if row["minutes"] * repeats < minutes:
        gaps.append(f"{minutes - row['minutes'] * repeats:g} min short")
    if row["sms"] * repeats < sms:
        gaps.append(f"{sms - row['sms'] * repeats:g} SMS short")
    if gaps:
        print(f"    note       : {', '.join(gaps)} of what you asked for")


def main():
    parser = argparse.ArgumentParser(description="Recommend a SIM pack.")
    parser.add_argument("--operator", required=True,
                        help="Grameenphone | Banglalink | Teletalk | Robi")
    parser.add_argument("--data", type=float, default=0.0, help="GB wanted")
    parser.add_argument("--minutes", type=float, default=0.0, help="minutes wanted")
    parser.add_argument("--sms", type=float, default=0.0, help="SMS wanted")
    parser.add_argument("--validity", type=float, default=30.0, help="days wanted")
    parser.add_argument("--top", type=int, default=3, help="how many to show")
    args = parser.parse_args()

    bundle = load_bundle()
    operator = resolve_operator(args.operator, bundle["operator_index"])
    wants = (args.data, args.minutes, args.sms, args.validity)

    print(f"\nRequest: {operator} | {args.data:g} GB | {args.minutes:g} min | "
          f"{args.sms:g} SMS | {args.validity:g} days\n")

    results = recommend(bundle, operator, *wants, top=args.top)
    for rank, (row, confidence) in enumerate(results, start=1):
        print("Best match:" if rank == 1 else f"Alternative {rank - 1}:")
        describe(row, confidence, wants)
        print()


if __name__ == "__main__":
    main()
