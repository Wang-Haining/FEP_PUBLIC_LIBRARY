"""
Bias Probing via Attributes Classification of LLM Outputs

This script probes whether large language model (LLM) outputs exhibit systematic
variation across demographic characteristics (e.g., sex, race/ethnicity, patron type).

It loads LLM-generated responses stored in seed-wise JSON files, then builds classifiers
(Logistic Regression, MLP, and XGBoost) to predict demographic labels based on two types
of linguistic cues:

1. Content words (TF-IDF weighted)
2. Function words / stopwords (normalized raw counts)

The script performs 5-fold cross-validation using fixed random seeds, reporting:
- Mean accuracy and 95% confidence interval
- Averaged feature weights across folds
- Statistical significance of features using statsmodels logistic regression
- Volcano plots to visualize coefficient strength vs. p-value

"""


import argparse
import json
import os
import string
import warnings

import nltk
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nltk.corpus import stopwords
from scipy.stats import t
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

nltk.download('stopwords')
stop_words_set = set(stopwords.words("english"))


def load_data(model_name: str,
              characteristic: str,
              input_dir: str = "outputs") -> pd.DataFrame:
    """
    Load model generation outputs and extract text responses and target characteristics.

    Parameters:
    - model_name: HF/OpenAI model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct', 'gpt-4o').
    - characteristic: One of 'sex', 'race_ethnicity', or 'patron_type'.
    - input_dir: Directory containing seed-wise output JSON files.

    Returns:
    - DataFrame with columns ['response', 'label', 'seed'].
    """
    assert characteristic in ['sex', 'race_ethnicity', 'patron_type'], \
        "Characteristic must be one of: sex, race_ethnicity, patron_type"

    tag = model_name.split('/')[-1].replace('-', '_').replace('/', '_')
    files = [f for f in os.listdir(input_dir) if f.startswith(f"{tag}_seed_") and f.endswith(".json")]

    rows = []
    for file in files:
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                rows.append({
                    "response": entry["response"],
                    "label": entry[characteristic],
                    "seed": entry["seed"]
                })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["response", "label"]).reset_index(drop=True)
    return df


def compute_ci(accs, confidence=0.95):
    mean = np.mean(accs)
    sem = np.std(accs, ddof=1) / np.sqrt(len(accs))
    h = sem * t.ppf((1 + confidence) / 2., len(accs) - 1)
    return mean, (mean - h, mean + h)


def get_feature_weights(clf, feature_names, model_type):
    if model_type == "logistic":
        weights = clf.coef_[0]
    elif model_type == "mlp":
        weights = clf.coefs_[0][:, 0]
    elif model_type == "xgboost":
        booster = clf.get_booster()
        importance = booster.get_score(importance_type="weight")
        return pd.DataFrame({
            "feature": list(importance.keys()),
            "weight": list(importance.values())
        }).sort_values(by="weight", ascending=False)
    else:
        raise ValueError("Unsupported model type")

    return pd.DataFrame({
        "feature": feature_names,
        "weight": weights
    }).sort_values(by="weight", ascending=False)


# def probe(df, mode="content", max_features=200):
#     """
#     Unified probing function for content vs stylistic cues.
#     Parameters:
#         - df: DataFrame with 'response', 'label', 'seed'
#         - mode: "content" or "stopwords"
#         - max_features: number of top features to use
#     Returns:
#         - Dictionary with model results and statsmodels output
#     """
#     assert mode in ["content", "stopwords"], "mode must be 'content' or 'stopwords'"
#     results = {}
#
#     # tokenize & vectorize
#     if mode == "content":
#         class ContentTokenizer:
#             def __init__(self):
#                 self.exclusion_set = set(stop_words_set).union({"mr", "ms", "mrs", "miss"})
#
#             def __call__(self, doc):
#                 tokens = [t.strip(string.punctuation).lower() for t in doc.split()]
#                 return [t for t in tokens if t and t not in self.exclusion_set]
#
#         vectorizer = TfidfVectorizer(
#             tokenizer=ContentTokenizer(),
#             token_pattern=None,
#             max_features=max_features
#         )
#         X = vectorizer.fit_transform(df["response"]).toarray()
#     else:
#         class StopwordTokenizer:
#             def __call__(self, doc):
#                 tokens = [t.strip(string.punctuation).lower() for t in doc.split()]
#                 return [t for t in tokens if t in stop_words_set]
#         vectorizer = CountVectorizer(
#             tokenizer=StopwordTokenizer(),
#             token_pattern=None,
#             max_features=max_features
#         )
#         X = vectorizer.fit_transform(df["response"]).toarray()
#         X = StandardScaler().fit_transform(X)
#
#     # prepare labels & splits
#     le = LabelEncoder()
#     y = le.fit_transform(df["label"])
#     feature_names = vectorizer.get_feature_names_out()
#     seeds = sorted(df["seed"].unique())
#     splits = [(df["seed"] != s, df["seed"] == s) for s in seeds]
#
#     # model definitions
#     model_defs = {
#         "logistic": lambda: LogisticRegression(
#             C=1.0, max_iter=1000, solver="liblinear", penalty="l2", random_state=42
#         ),
#         "mlp": lambda: MLPClassifier(
#             hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
#             alpha=1e-4, max_iter=2000, early_stopping=True, random_state=42
#         ),
#         "xgboost": lambda: XGBClassifier(
#             n_estimators=100, learning_rate=0.1, max_depth=4,
#             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
#             use_label_encoder=False, eval_metric="logloss", verbosity=0, random_state=42
#         )
#     }
#
#     # train & collect results
#     for name, constructor in model_defs.items():
#         accs, weights = [], []
#         for train_idx, test_idx in splits:
#             clf = constructor()
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 clf.fit(X[train_idx], y[train_idx])
#             preds = clf.predict(X[test_idx])
#             accs.append(accuracy_score(y[test_idx], preds))
#             weights.append(get_feature_weights(clf, feature_names, name))
#
#         # aggregate accuracy and feature importance
#         mean_acc, ci = compute_ci(accs)
#         avg_weights = (
#             pd.concat(weights)
#               .groupby("feature")
#               .mean()
#               .reset_index()
#               .sort_values("weight", ascending=False)
#         )
#
#         # map XGBoost f# names back to tokens
#         if name == "xgboost":
#             mapping = {f"f{i}": feature_names[i] for i in range(len(feature_names))}
#             avg_weights["feature"] = avg_weights["feature"].map(mapping)
#
#         results[name] = {"mean_acc": mean_acc, "ci": ci, "feature_weights": avg_weights}
#
#     # statsmodels analysis
#     X_const = sm.add_constant(X)
#     n_classes = len(np.unique(y))
#
#     if n_classes == 2:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             sm_model = sm.Logit(y, X_const).fit(disp=False, method='newton')
#         params, pvals = sm_model.params, sm_model.pvalues
#         feat_const = ['const'] + list(feature_names)
#         mask = ~np.isnan(params)
#         stats_df = pd.DataFrame({
#             'feature': [feat_const[i] for i in range(len(mask)) if mask[i]],
#             'class': '0',
#             'coef': params[mask],
#             'p_value': pvals[mask]
#         })
#     else:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             sm_model = sm.MNLogit(y, X_const).fit(disp=False, method='newton')
#         params, pvals = sm_model.params.flatten(), sm_model.pvalues.flatten()
#         feat_const = ['const'] + list(feature_names)
#         feats_exp, classes_exp = [], []
#         for i, feat in enumerate(feat_const):
#             for c in range(n_classes - 1):
#                 feats_exp.append(feat)
#                 classes_exp.append(str(c))
#         valid = ~np.isnan(params)
#         stats_df = pd.DataFrame({
#             'feature': [feats_exp[i] for i in range(len(valid)) if valid[i]],
#             'class': [classes_exp[i] for i in range(len(valid)) if valid[i]],
#             'coef': params[valid],
#             'p_value': pvals[valid]
#         })
#
#     stats_df = stats_df[stats_df.feature != 'const'].dropna(subset=['coef','p_value']).reset_index(drop=True)
#     stats_df = stats_df.loc[stats_df['coef'].abs().sort_values(ascending=False).index].reset_index(drop=True)
#     results["statsmodels"] = stats_df
#
#     return results

def probe(
    df: pd.DataFrame,
    mode: str = "content",
    max_features: int = 200,
    stats_top_k: int = 100
) -> dict:
    """
    Unified probing function for content vs. stylistic cues.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['response', 'label', 'seed'].
    mode : {"content", "stopwords"}
        "content": TF‑IDF on content words (excludes stopwords + honorifics).
        "stopwords": raw counts of function words (z‑scored).
    max_features : int
        Max vocabulary size for the vectorizer.
    stats_top_k : int
        In stopwords mode, number of top features (by absolute logistic weight)
        to include in the statsmodels test.

    Returns
    -------
    results : dict
      - "logistic", "mlp", "xgboost": each → {"mean_acc": float,
                                             "ci": (low, high),
                                             "feature_weights": DataFrame}
      - "statsmodels":    DataFrame(feature, class, coef, p_value)
    """
    import string
    import warnings
    from scipy.stats import t
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    import statsmodels.api as sm

    # 1. vectorize
    if mode == "content":
        class Tok:
            def __init__(self):
                stop = set(stopwords.words("english"))
                self.excl = stop.union({"mr","ms","mrs","miss"})
            def __call__(self, doc):
                toks = [w.strip(string.punctuation).lower() for w in doc.split()]
                return [w for w in toks if w and w not in self.excl]

        vec = TfidfVectorizer(tokenizer=Tok(), token_pattern=None,
                              max_features=max_features)
        X = vec.fit_transform(df["response"]).toarray()

    else:  # stopwords
        class Tok:
            def __init__(self):
                self.stop = set(stopwords.words("english"))
            def __call__(self, doc):
                toks = [w.strip(string.punctuation).lower() for w in doc.split()]
                return [w for w in toks if w and w in self.stop]

        vec = CountVectorizer(tokenizer=Tok(), token_pattern=None,
                              max_features=max_features)
        X = vec.fit_transform(df["response"]).toarray()
        X = StandardScaler().fit_transform(X)

    feature_names = vec.get_feature_names_out()

    # 2. labels & CV splits
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    seeds = sorted(df["seed"].unique())
    splits = [(df["seed"] != s, df["seed"] == s) for s in seeds]

    # helper for CI
    def compute_ci(accs, conf=0.95):
        m = np.mean(accs)
        se = np.std(accs, ddof=1) / np.sqrt(len(accs))
        h = se * t.ppf((1 + conf)/2., len(accs)-1)
        return m, (m-h, m+h)

    # helper for feature weights
    def get_weights(clf, names, kind):
        if kind=="logistic":
            w = clf.coef_[0]
        elif kind=="mlp":
            w = clf.coefs_[0][:,0]
        else:  # xgboost
            imp = clf.get_booster().get_score(importance_type="weight")
            return (pd.DataFrame.from_dict(imp, orient="index", columns=["weight"])
                    .reset_index().rename(columns={"index":"feature"}))
        return pd.DataFrame({"feature": names, "weight": w})

    # 3. train 3 probes
    models = {
      "logistic": lambda: LogisticRegression(C=1.0, solver="liblinear",
                                             penalty="l2", max_iter=1000,
                                             random_state=42),
      "mlp": lambda: MLPClassifier(hidden_layer_sizes=(128,64),
                                    alpha=1e-4, max_iter=2000,
                                    early_stopping=True, random_state=42),
      "xgboost": lambda: XGBClassifier(n_estimators=100, learning_rate=0.1,
                                       subsample=0.8, colsample_bytree=0.8,
                                       reg_alpha=0.1, reg_lambda=1.0,
                                       use_label_encoder=False,
                                       eval_metric="logloss",
                                       max_depth=4, random_state=42)
    }

    results = {}
    for name, ctor in models.items():
        accs, wts = [], []
        for tr, te in splits:
            clf = ctor()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X[tr], y[tr])
            preds = clf.predict(X[te])
            accs.append(accuracy_score(y[te], preds))
            wts.append(get_weights(clf, feature_names, name))

        mean_acc, ci = compute_ci(accs)
        avg_w = pd.concat(wts).groupby("feature", as_index=False).mean()
        results[name] = {
            "mean_acc": mean_acc,
            "ci": ci,
            "feature_weights": avg_w.sort_values("weight", ascending=False)
        }

    # 4. statsmodels significance testing
    X_const = sm.add_constant(X)
    n_classes = len(np.unique(y))

    # for stopwords mode, reduce dimensionality
    if mode == "stopwords":
        lw = results["logistic"]["feature_weights"].copy()
        lw["abs_w"] = lw["weight"].abs()
        chosen = lw.nlargest(stats_top_k, "abs_w")["feature"].tolist()
        feat_const = ["const"] + chosen
        # build reduced matrix with only selected features
        name_to_idx = {name: i+1 for i, name in enumerate(feature_names)}
        idx = [0] + [name_to_idx[f] for f in chosen]
        X_sub = X_const[:, idx]
    else:
        feat_const = ["const"] + list(feature_names)
        X_sub = X_const

    # now fit binary or multinomial logistic
    if n_classes == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_mod = sm.Logit(y, X_sub).fit(disp=False, method="newton")
        params, pvals = sm_mod.params, sm_mod.pvalues
        mask = ~np.isnan(params)
        stats_df = pd.DataFrame({
            "feature": [feat_const[i] for i in range(len(mask)) if mask[i]],
            "class":    ["0"] * mask.sum(),
            "coef":     params[mask],
            "p_value":  pvals[mask]
        })
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_mod = sm.MNLogit(y, X_sub).fit(disp=False, method="newton")
        # sm_mod.params and .pvalues are already ndarrays
        params = sm_mod.params.flatten()
        pvals  = sm_mod.pvalues.flatten()
        feats, classes = [], []
        # feature `i` appears for each of the (n_classes-1) logits
        for i, feat in enumerate(feat_const):
            for c in range(n_classes - 1):
                feats.append(feat)
                classes.append(str(c))
        valid = ~np.isnan(params)
        stats_df = pd.DataFrame({
            "feature":  [feats[i]   for i in range(len(valid)) if valid[i]],
            "class":    [classes[i] for i in range(len(valid)) if valid[i]],
            "coef":     params[valid],
            "p_value":  pvals[valid]
        })

    # drop constant, sort by absolute coefficient
    stats_df = (
        stats_df[stats_df.feature != "const"]
        .dropna(subset=["coef", "p_value"])
        .reset_index(drop=True)
    )
    stats_df = stats_df.loc[
        stats_df.coef.abs().sort_values(ascending=False).index
    ].reset_index(drop=True)

    results["statsmodels"] = stats_df


def print_top_features(results, top_n=10):
    for model in ["logistic", "mlp", "xgboost"]:
        if model in results:
            print(f"\n=== Top {top_n} features for {model.upper()} ===")
            print(results[model]["feature_weights"].head(top_n).to_string(index=False))
    if "statsmodels" in results:
        print(f"\n=== Top {top_n} features by STATS MODELS Logistic Regression (with p-values) ===")
        print(results["statsmodels"].head(top_n).to_string(index=False))


def serialize_for_json(results):
    def convert(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.ndarray, list)):
            return [convert(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        else:
            return obj
    return convert(results)


# def main():
#     """
#     Main driver for probing LLM outputs by demographic attributes.
#     Loads data, runs probes for all model-characteristic-mode combinations,
#     and serializes to probe.json.
#     """
#     model_names = [
#         "meta-llama/Llama-3.1-8B-Instruct",
#         "mistralai/Ministral-8B-Instruct-2410",
#         "google/gemma-2-9b-it"
#     ]
#     characteristics = ["sex", "race_ethnicity", "patron_type"]
#     modes = ["content", "stopwords"]
#
#     all_results = {}
#     total = len(model_names) * len(characteristics) * len(modes)
#     progress = tqdm(total=total, desc="Running probes")
#
#     for model in model_names:
#         all_results[model] = {}
#         for char in characteristics:
#             df = load_data(model, char)
#             all_results[model][char] = {}
#             for mode in modes:
#                 results = probe(df, mode=mode, max_features=200)
#                 all_results[model][char][mode] = results
#                 progress.update(1)
#
#     progress.close()
#
#     with open("probe.json", "w") as f:
#         json.dump(serialize_for_json(all_results), f, indent=2)
#     print("\nAll experiments completed and results saved to 'probe.json'.")

def main():
    """
    Main driver for probing LLM outputs by demographic attributes.
    With --debug, only runs a single probe for quick inspection.
    Otherwise, runs the full grid of models × characteristics × modes.
    """
    parser = argparse.ArgumentParser(description="Run attribute‐probing suite")
    parser.add_argument(
        "--debug", action="store_true",
        help="only run Llama-3.1 patron_type stopwords probe and print statsmodels"
    )
    args = parser.parse_args()

    if args.debug:
        # pick exactly one combination to sanity‐check
        model = "meta-llama/Llama-3.1-8B-Instruct"
        characteristic = "patron_type"
        mode = "stopwords"
        print(f"DEBUG: running single probe for {model} / {characteristic} / {mode}")
        df = load_data(model, characteristic)
        results = probe(df, mode=mode, max_features=200)
        print("\nDEBUG: statsmodels output:\n")
        print(results["statsmodels"])
        return   # bail out immediately

    else:
        # --- normal mode: full sweep ---
        model_names = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Ministral-8B-Instruct-2410",
            "google/gemma-2-9b-it"
        ]
        characteristics = ["sex", "race_ethnicity", "patron_type"]
        modes = ["content", "stopwords"]

        all_results = {}
        total = len(model_names) * len(characteristics) * len(modes)
        progress = tqdm(total=total, desc="Running probes")

        for model in model_names:
            all_results[model] = {}
            for char in characteristics:
                df = load_data(model, char)
                all_results[model][char] = {}
                for mode in modes:
                    results = probe(df, mode=mode, max_features=200)
                    all_results[model][char][mode] = results
                    progress.update(1)

        progress.close()

        with open("probe.json", "w") as f:
            json.dump(serialize_for_json(all_results), f, indent=2)
        print("\nAll experiments completed and results saved to 'probe.json'.")


if __name__ == "__main__":
    main()
