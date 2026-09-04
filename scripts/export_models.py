import gzip
import json
import os
import sys

import joblib
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(BASE, "models", "offer_recommender.joblib")
EXPORT = os.path.join(BASE, "models", "models.json.gz")

SAMPLE = 400


def export_tree(tree):
    inner = tree.tree_
    leaves = {}
    for node in range(inner.node_count):
        if inner.children_left[node] != -1:
            continue
        counts = inner.value[node][0]
        leaves[str(node)] = [[int(i), float(counts[i])]
                             for i in np.nonzero(counts)[0]]
    return {
        "feature": [int(f) for f in inner.feature],
        "threshold": [float(t) for t in inner.threshold],
        "left": [int(c) for c in inner.children_left],
        "right": [int(c) for c in inner.children_right],
        "leaves": leaves,
    }


def export_model(model, n_classes):
    if hasattr(model, "estimators_"):
        return {"kind": "forest", "n_classes": n_classes,
                "trees": [export_tree(t) for t in model.estimators_]}
    if hasattr(model, "tree_"):
        return {"kind": "forest", "n_classes": n_classes,
                "trees": [export_tree(model)]}

    scaler, poly, ridge = (model.named_steps["standardscaler"],
                           model.named_steps["polynomialfeatures"],
                           model.named_steps["ridgeclassifier"])
    return {
        "kind": "linear",
        "n_classes": n_classes,
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "powers": [[int(p) for p in row] for row in poly.powers_],
        "coef": [[float(c) for c in row] for row in np.atleast_2d(ridge.coef_)],
        "intercept": [float(v) for v in np.atleast_1d(ridge.intercept_)],
    }


def main():
    if not os.path.exists(BUNDLE):
        raise SystemExit(f"No {BUNDLE}. Run: python3 scripts/train_model.py")
    bundle = joblib.load(BUNDLE)

    export = {
        "classes": bundle["classes"],
        "operator_index": bundle["operator_index"],
        "metrics": bundle["metrics"],
        "sklearn_version": bundle.get("sklearn_version"),
        "models": {name: export_model(model, len(bundle["classes"][name]))
                   for name, model in bundle["models"].items()},
    }

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ml

    rng = np.random.default_rng(7)
    requests = np.column_stack([
        rng.integers(0, len(bundle["operator_index"]), SAMPLE),
        rng.uniform(0, 60, SAMPLE), rng.uniform(0, 1500, SAMPLE),
        rng.uniform(0, 500, SAMPLE), rng.choice([1, 3, 7, 15, 30, 60], SAMPLE),
    ])
    for name, model in bundle["models"].items():
        theirs = (model.predict_proba(requests) if hasattr(model, "predict_proba")
                  else model.decision_function(requests))
        mine = np.array([ml.raw_scores(export["models"][name], row)
                         for row in requests])
        worst = float(np.max(np.abs(theirs - mine)))
        agree = float(np.mean(np.argmax(theirs, 1) == np.argmax(mine, 1)))
        print(f"  {name:24} same pick on {agree:.1%} of {SAMPLE} requests, "
              f"largest score difference {worst:.2e}")
        if agree < 1.0 or worst > 1e-6:
            raise SystemExit(f"{name} does not survive the export -- not written")

    os.makedirs(os.path.dirname(EXPORT), exist_ok=True)
    with gzip.open(EXPORT, "wt", encoding="utf-8") as out:
        json.dump(export, out, separators=(",", ":"))
    print(f"\nWrote {EXPORT} ({os.path.getsize(EXPORT) / 1e6:.1f} MB), "
          f"readable with the standard library alone.")


if __name__ == "__main__":
    main()
