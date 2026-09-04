import gzip
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "offer_recommender.joblib")

EXPORT_PATH = os.path.join(BASE, "models", "models.json.gz")

_export = None

MATCHER = "Cost matcher (no model)"

_bundle = None
_problem = None


def _load_export():
    global _export
    if _export is None and os.path.exists(EXPORT_PATH):
        with gzip.open(EXPORT_PATH, "rt", encoding="utf-8") as handle:
            _export = json.load(handle)
    return _export


def _leaf(tree, features):
    node = 0
    while tree["left"][node] != -1:
        feature = tree["feature"][node]
        node = (tree["left"][node] if features[feature] <= tree["threshold"][node]
                else tree["right"][node])
    return tree["leaves"][str(node)]


def raw_scores(spec, features):
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
        import joblib
        _bundle = joblib.load(MODEL_PATH)
    except ImportError as exc:
        _problem = (f"{exc.name} is not installed in this interpreter, so the "
                    "saved models cannot be read")
    except Exception as exc:
        _problem = f"the model file could not be read ({exc})"
    return _bundle


def available():
    export = _load_export()
    if export is not None:
        return list(export["models"]), None

    bundle = _load()
    if bundle is None:
        return [], _problem
    return list(bundle["models"]), version_warning()


def version_warning():
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
    stored = bundle.get("classes", {}).get(name)
    if stored is not None:
        return stored
    return [str(c) for c in model.classes_]


def metrics():
    export = _load_export()
    if export is not None:
        return dict(export["metrics"])
    bundle = _load()
    return dict(bundle["metrics"]) if bundle else {}


def rank(name, operator, wants, allowed, top=10):
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
