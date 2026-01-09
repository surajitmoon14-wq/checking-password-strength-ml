import joblib
from pathlib import Path
import sys
import types

# Provide a fallback `char_tokenizer` function for unpickling vectorizers
# that were serialized with a custom tokenizer defined in `__main__`.
def char_tokenizer(s):
    try:
        return list(s)
    except Exception:
        return [c for c in str(s)]

if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")
setattr(sys.modules["__main__"], "char_tokenizer", char_tokenizer)

# Resolve models directory: prefer the app's `models/` folder, but fall
# back to the repository root where some .pkl files are stored.
APP_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parents[2]
if (APP_DIR / "models").exists():
    MODELS_DIR = APP_DIR / "models"
else:
    MODELS_DIR = BASE_DIR


def _load_model(filename):
    # Try app/models first, then repo root. Return loaded object or raise.
    candidates = [APP_DIR / "models" / filename, BASE_DIR / filename]
    # prefer resaved variants when available
    resaved_candidates = [p.with_name(p.stem + "_resaved.pkl") for p in candidates]
    for p in resaved_candidates + candidates:
        try:
            if p.exists():
                return joblib.load(str(p))
        except Exception:
            # try next candidate
            continue
    raise FileNotFoundError(f"Model file '{filename}' not found in {candidates}")


def _find_best_vectorizer(model, candidate_paths):
    """Given a model and iterable of vectorizer file paths, return the loaded
    vectorizer that best matches the model's expected input feature count.
    Returns (vectorizer, path) or (None, None).
    """
    best = (None, None, None)  # (vec, path, diff)
    model_n = getattr(model, "n_features_in_", None)
    for p in candidate_paths:
        try:
            vec = joblib.load(str(p))
        except Exception:
            continue
        # determine feature count for vectorizer
        n_vec = None
        try:
            if hasattr(vec, "vocabulary_"):
                n_vec = len(vec.vocabulary_)
            else:
                # try feature names (scikit-learn >=1.0)
                n_vec = len(vec.get_feature_names_out())
        except Exception:
            try:
                # last resort: transform a sample and check shape
                n_vec = vec.transform(["x"]).shape[1]
            except Exception:
                n_vec = None

        if n_vec is None:
            continue

        if model_n is not None and n_vec == int(model_n):
            return vec, p

        diff = abs(n_vec - int(model_n)) if model_n is not None else None
        if best[0] is None or (diff is not None and (best[2] is None or diff < best[2])):
            best = (vec, p, diff)

    return (best[0], best[1]) if best[0] is not None else (None, None)

# ================= LOAD MODELS =================

# Strength prediction
strength_model = _load_model("enhanced_password_model_1.pkl")
strength_vectorizer = _load_model("tfidf_vectorizer_1.pkl")

# Password improvement
improvement_model = _load_model("password_improvement_model.pkl")

# Length recommendation
length_model = _load_model("length_recommender_model.pkl")
length_vectorizer = _load_model("length_vectorizer.pkl")

# New recommender
new_model = _load_model("new_recommender_model.pkl")
new_vectorizer = _load_model("new_vectorizer.pkl")


# ================= CORE ML PIPELINE =================
def run_all_ml(password):
    # Strength: try the primary vectorizer first; on feature-mismatch try other TF-IDF pickles
    def _attempt_strength_predict(vec):
        X = vec.transform([password])
        return strength_model.predict(X)[0]

    predicted_strength = None
    try:
        X_strength = strength_vectorizer.transform([password])
        predicted_strength = strength_model.predict(X_strength)[0]
    except Exception:
        # try to find the best-matching vectorizer by feature-count
        candidates = []
        app_models_dir = APP_DIR / "models"
        if app_models_dir.exists():
            candidates.extend(app_models_dir.glob("tfidf_vectorizer*.pkl"))
        candidates.extend(BASE_DIR.glob("tfidf_vectorizer*.pkl"))
        vec, path = _find_best_vectorizer(strength_model, candidates)
        if vec is not None:
            try:
                X_strength = vec.transform([password])
                predicted_strength = strength_model.predict(X_strength)[0]
                strength_vectorizer = vec
            except Exception:
                predicted_strength = "Unknown"
    if predicted_strength is None:
        predicted_strength = "Unknown"

    # Improvement (only if we have a transformed strength input)
    try:
        improved_password = improvement_model.predict(X_strength)[0]
    except Exception:
        try:
            # try to transform with length_vectorizer as a last resort
            X_tmp = length_vectorizer.transform([password])
            improved_password = improvement_model.predict(X_tmp)[0]
        except Exception:
            improved_password = ""

    # Length recommendation: robustly try vectorizers if feature-size mismatch occurs
    recommended_length = None
    try:
        X_length = length_vectorizer.transform([password])
        recommended_length = int(length_model.predict(X_length)[0])
    except Exception:
        candidates = []
        app_models_dir = APP_DIR / "models"
        if app_models_dir.exists():
            candidates.extend(app_models_dir.glob("length_vectorizer*.pkl"))
        candidates.extend(BASE_DIR.glob("length_vectorizer*.pkl"))
        candidates.extend(app_models_dir.glob("tfidf_vectorizer*.pkl") if app_models_dir.exists() else [])
        candidates.extend(BASE_DIR.glob("tfidf_vectorizer*.pkl"))
        vec, path = _find_best_vectorizer(length_model, candidates)
        if vec is not None:
            try:
                X_tmp = vec.transform([password])
                recommended_length = int(length_model.predict(X_tmp)[0])
                length_vectorizer = vec
            except Exception:
                recommended_length = 0
        else:
            recommended_length = 0

    # Extra ML suggestions
    try:
        X_new = new_vectorizer.transform([password])
        suggestions = new_model.predict(X_new)
    except Exception:
        # try to find the best-matching vectorizer for the new_model
        suggestions = []
        candidates = []
        app_models_dir = APP_DIR / "models"
        if app_models_dir.exists():
            candidates.extend(app_models_dir.glob("new_vectorizer*.pkl"))
        candidates.extend(BASE_DIR.glob("new_vectorizer*.pkl"))
        candidates.extend(app_models_dir.glob("tfidf_vectorizer*.pkl") if app_models_dir.exists() else [])
        candidates.extend(BASE_DIR.glob("tfidf_vectorizer*.pkl"))
        vec, path = _find_best_vectorizer(new_model, candidates)
        if vec is not None:
            try:
                X_tmp = vec.transform([password])
                suggestions = new_model.predict(X_tmp)
                new_vectorizer = vec
            except Exception:
                suggestions = []
    # ---- Rule-based checks (issues_found) ----
    issues = []
    if len(password) < 8:
        issues.append("Password is too short")
    if not any(c.isupper() for c in password):
        issues.append("No uppercase letter")
    if not any(c.islower() for c in password):
        issues.append("No lowercase letter")
    if not any(c.isdigit() for c in password):
        issues.append("No digit")
    if not any(not c.isalnum() for c in password):
        issues.append("No special character")

    # Map numeric labels to readable text when needed
    try:
        label_map = {0: "Weak", 1: "Medium", 2: "Strong"}
        pred_label = label_map.get(int(predicted_strength), str(predicted_strength))
    except Exception:
        pred_label = str(predicted_strength)

    # If issues were found and predicted label is Weak, indicate rule-based note
    if issues and pred_label == "Weak":
        pred_display = f"{pred_label} (Rule-based)"
    else:
        pred_display = pred_label

    # Suggestions: Prefer model-generated suggestions; fallback to improved password
    suggestions_list = []
    try:
        suggestions_list = list(suggestions)
    except Exception:
        suggestions_list = []

    if not suggestions_list:
        # fallback
        if improved_password:
            suggestions_list = [improved_password]
        else:
            suggestions_list = ["Could not generate strong password quickly"]

    return {
        "input_password": password,
        "predicted_strength": pred_display,
        "issues_found": issues,
        "suggestions": suggestions_list,
        "improved_password": improved_password,
        "recommended_length": recommended_length,
        "ml_suggestions": list(suggestions)
    }
