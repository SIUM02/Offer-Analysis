"""Ranking packs with the trained models, for anything that wants to.

    from ml import available, rank

`available()` says which models can be used right now, and why not when the
answer is none: the models live in models/offer_recommender.joblib, which
scripts/train_model.py writes, and reading it needs joblib and scikit-learn.
Neither is needed by the matcher, so nothing here is imported until it is
asked for -- the page still runs on a bare python3, with the matcher alone.

`rank(name, operator, wants, allowed, top)` returns the model's best packs
for one request, as [(offer_id, score)], highest first, restricted to packs
the operator actually sells. What the score means depends on the model, so
it comes back labelled: a probability for the two tree models, a normalised
regression output for the polynomial one, which is a ranking number and not
a probability however much it looks like one.
"""

import gzip
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "offer_recommender.joblib")

# The same models written out as plain data by scripts/export_models.py.
# Preferred over the pickle: it reads with the standard library, so the page
# needs no scikit-learn, matches no version, and installs nothing -- which is
# what lets the models run on a serverless host at all.
EXPORT_PATH = os.path.join(BASE, "models", "models.json.gz")

_export = None

MATCHER = "Cost matcher (no model)"

# Filled by _load() on first use: the bundle, or the reason there is none.
_bundle = None
_problem = None


def _load_export():
    """The exported models, read with the standard library. None if absent."""
    global _export
    if _export is None and os.path.exists(EXPORT_PATH):
        with gzip.open(EXPORT_PATH, "rt", encoding="utf-8") as handle:
            _export = json.load(handle)
    return _export


def _leaf(tree, features):
    """Walk one tree to the leaf this request lands in."""
    node = 0
    while tree["left"][node] != -1:
        feature = tree["feature"][node]
        node = (tree["left"][node] if features[feature] <= tree["threshold"][node]
                else tree["right"][node])
    return tree["leaves"][str(node)]


def raw_scores(spec, features):
    """One score per pack, from an exported model, in pure Python.

    A forest averages the class distribution of the leaf each tree drops the
    request into -- exactly what predict_proba does, over the sparse leaves
    the export keeps. The linear model standardises, expands to the stored
    powers and takes one dot product per pack, which is decision_function.
    """
    n_classes = spec["n_classes"]
    if spec["kind"] == "forest":
        totals = [0.0] * n_classes
        trees = spec["trees"]
        for tree in trees:
            pairs = _leaf(tree, features)
            weight = sum(count for _, count in pairs)
            for index, count in pairs:
                totals[index] += count / weight
        return [total / len(trees) for total in totals]

    standardised = [(value - mean) / scale for value, mean, scale
                    in zip(features, spec["mean"], spec["scale"])]
    expanded = []
    for powers in spec["powers"]:
        term = 1.0
        for value, power in zip(standardised, powers):
            if power:
                term *= value ** power
        expanded.append(term)
    return [sum(c * f for c, f in zip(row, expanded)) + bias
            for row, bias in zip(spec["coef"], spec["intercept"])]


def _load():
    global _bundle, _problem
    if _bundle is not None or _problem is not None:
        return _bundle

    if not os.path.exists(MODEL_PATH):
        _problem = ("no models/offer_recommender.joblib -- run "
                    "python3 scripts/train_model.py to build it")
        return None
    try:
        import joblib                                    # noqa: F401
        _bundle = joblib.load(MODEL_PATH)
    except ImportError as exc:
        _problem = (f"{exc.name} is not installed in this interpreter, so the "
                    "saved models cannot be read")
    except Exception as exc:                             # pragma: no cover
        _problem = f"the model file could not be read ({exc})"
    return _bundle


def available():
    """The model names that can be used now, and the reason if none can."""
    export = _load_export()
    if export is not None:
        return list(export["models"]), None

    bundle = _load()
    if bundle is None:
        return [], _problem
    return list(bundle["models"]), version_warning()


def version_warning():
    """A warning if these models were saved by a different scikit-learn.

    A model pickled by one version and loaded by another is not reliably the
    same model: attributes move, defaults change, and the failure is not
    always an exception -- it can be a silently different answer. The models
    still load, so this is said rather than enforced, but it is said.
    """
    bundle = _load()
    if bundle is None:
        return None
    saved = bundle.get("sklearn_version")
    try:
        import sklearn
    except ImportError:
        return None
    if saved and saved != sklearn.__version__:
        return (f"the models were trained with scikit-learn {saved} and this "
                f"interpreter has {sklearn.__version__}; retrain with "
                f"python3 scripts/train_model.py to be sure of them")
    return None


def class_list(bundle, name, model):
    """The packs a model scores, in the order its scores come back in.

    Taken from the bundle, which recorded them at training time; only if an
    older bundle lacks them is the estimator asked, which is exactly what
    breaks across scikit-learn versions.
    """
    stored = bundle.get("classes", {}).get(name)
    if stored is not None:
        return stored
    return [str(c) for c in model.classes_]


def metrics():
    """What each model scored when it was trained, or {} if unavailable."""
    export = _load_export()
    if export is not None:
        return dict(export["metrics"])
    bundle = _load()
    return dict(bundle["metrics"]) if bundle else {}


def rank(name, operator, wants, allowed, top=10):
    """One model's best packs for one request: [(offer_id, score)], best first.

    `allowed` is the set of offer_ids the operator sells; everything else is
    masked out before ranking, so a Robi request can never come back with a
    Grameenphone pack. Scores are comparable within one answer, not between
    models.
    """
    export = _load_export()
    if export is not None:
        if name not in export["models"]:
            raise RuntimeError(f"{name} is not one of the trained models")
        spec = export["models"][name]
        features = [export["operator_index"][operator], *wants]
        scores = raw_scores(spec, features)
        classes = export["classes"][name]

        if spec["kind"] == "forest":
            label = "Confidence"
        else:
            low, high = min(scores), max(scores)
            span = high - low
            scores = [(s - low) / span if span else 0.0 for s in scores]
            label = "Score"

        ranked = sorted(((offer_id, score) for offer_id, score
                         in zip(classes, scores) if offer_id in allowed),
                        key=lambda pair: -pair[1])[:top]
        return ranked, label

    import numpy as np

    bundle = _load()
    if bundle is None:
        raise RuntimeError(_problem)
    if name not in bundle["models"]:
        raise RuntimeError(f"{name} is not one of the trained models")
    model = bundle["models"][name]

    features = np.array([[bundle["operator_index"][operator], *wants]])
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[0]
        label = "Confidence"
    else:
        # A least-squares regression per pack: the numbers are unbounded and
        # often negative, so they are squashed into 0-1 across the packs on
        # offer. That makes them readable as a ranking; it does not make
        # them probabilities.
        raw = model.decision_function(features)[0]
        low, high = float(np.min(raw)), float(np.max(raw))
        scores = (raw - low) / (high - low) if high > low else np.zeros_like(raw)
        label = "Score"

    classes = np.asarray(class_list(bundle, name, model))
    keep = np.isin(classes, list(allowed))
    masked = np.where(keep, scores, -np.inf)
    order = np.argsort(masked)[::-1][:top]
    return [(str(classes[i]), float(scores[i])) for i in order
            if masked[i] != -np.inf], label
