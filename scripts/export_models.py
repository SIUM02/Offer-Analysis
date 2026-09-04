"""Writes the trained models out as plain data, so anything can run them.

    python3 scripts/export_models.py     # after train_model.py

A joblib pickle needs scikit-learn to read, needs the *same* scikit-learn to
read safely, and drags numpy, scipy and pandas along with it. That is 250 MB
of dependencies and 400 MB of memory for a forest -- more than a serverless
function is given, and more than the page needs. Yet the models themselves
are small: a tree is thresholds and child indices, and a leaf holds two
classes on average, not the 156 that scikit-learn stores densely at every
one of them.

So they are written here as gzipped JSON, sparsely, losing nothing:

    models/models.json.gz

ml.py reads that with the standard library alone -- no scikit-learn, no
version to match, nothing to install. The predictions are identical, and
export_models.py checks that they are before it writes anything.
"""

import gzip
import json
import os
import sys

import joblib
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(BASE, "models", "offer_recommender.joblib")
EXPORT = os.path.join(BASE, "models", "models.json.gz")

SAMPLE = 400          # requests to check the export against scikit-learn on


def export_tree(tree):
    """One decision tree as flat arrays, with sparse leaves.

    Every node keeps the feature it splits on, the threshold, and where to go;
    a leaf keeps only the classes that actually reached it, as (class index,
    count) pairs. Normalising those counts gives back exactly the probability
    vector scikit-learn stores dense.
    """
    inner = tree.tree_
    leaves = {}
    for node in range(inner.node_count):
        if inner.children_left[node] != -1:
            continue
        counts = inner.value[node][0]
        # value is normalised in newer scikit-learn and raw counts in older;
        # either way only the ratios matter, and they are preserved.
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

    # The polynomial pipeline: standardise, expand to degree 2, then one
    # least-squares regression per pack. Exporting the powers rather than
    # assuming an order means the expansion cannot drift out of step.
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

    # Check before writing: the exported model must answer exactly as the one
    # it came from, or it is not the same model and should not be shipped.
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
