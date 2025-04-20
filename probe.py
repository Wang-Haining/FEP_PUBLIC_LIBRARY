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


import json
import os
import string
import warnings

import matplotlib.pyplot as plt
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
from xgboost import XGBClassifier

nltk.download('stopwords')
stop_words_set = set(stopwords.words("english"))


def load_data(model_name: str,
              characteristic: str,
              input_dir: str = "/content/equity_across_difference/outputs") -> pd.DataFrame:
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


def probe(df, mode="content", max_features=200):
    """
    Unified probing function for content vs stylistic cues.
    Parameters:
        - df: DataFrame with 'response', 'label', 'seed'
        - mode: "content" or "stopwords"
        - max_features: number of top features to use
    Returns:
        - Dictionary with model results and statsmodels output
    """
    assert mode in ["content", "stopwords"], "mode must be 'content' or 'stopwords'"
    results = {}

    if mode == "content":
        class ContentTokenizer:
            def __call__(self, doc):
                tokens = [t.strip(string.punctuation).lower() for t in doc.split()]
                return [t for t in tokens if t and t not in stop_words_set]
        vectorizer = TfidfVectorizer(
            tokenizer=ContentTokenizer(),
            token_pattern=None,
            max_features=max_features
        )
        X = vectorizer.fit_transform(df["response"]).toarray()

    else:  # stopwords mode
        class StopwordTokenizer:
            def __call__(self, doc):
                tokens = [t.strip(string.punctuation).lower() for t in doc.split()]
                return [t for t in tokens if t in stop_words_set]
        vectorizer = CountVectorizer(
            tokenizer=StopwordTokenizer(),
            token_pattern=None,
            max_features=max_features
        )
        X = vectorizer.fit_transform(df["response"]).toarray()
        X = StandardScaler().fit_transform(X)

    # encode labels & prepare 5-fold “seed” splits
    y = LabelEncoder().fit_transform(df["label"])
    feature_names = vectorizer.get_feature_names_out()
    seeds = sorted(df["seed"].unique())
    splits = [(df["seed"] != s, df["seed"] == s) for s in seeds]

    # three classifiers
    model_defs = {
        "logistic": lambda: LogisticRegression(C=1.0, max_iter=1000),
        "mlp":      lambda: MLPClassifier(
                         hidden_layer_sizes=(50,),
                         max_iter=2000,
                         random_state=0
                     ),
        "xgboost":  lambda: XGBClassifier(
                         use_label_encoder=False,
                         eval_metric="logloss",
                         verbosity=0
                     )
    }

    for name, constructor in model_defs.items():
        accs, weights = [], []
        for train_idx, test_idx in splits:
            clf = constructor()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], preds))
            weights.append(get_feature_weights(clf, feature_names, name))

        mean_acc, ci = compute_ci(accs)
        avg_weights = (
            pd.concat(weights)
              .groupby("feature")
              .mean()
              .reset_index()
              .sort_values("weight", ascending=False)
        )
        results[name] = {
            "mean_acc": mean_acc,
            "ci": ci,
            "feature_weights": avg_weights
        }

    X_const = sm.add_constant(X)

    # suppress convergence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # always use MNLogit (binary Logit is just a special case)
        sm_model = sm.MNLogit(y, X_const).fit(disp=False)

    # params & p‑values come back as DataFrames: (n_exog × n_classes)
    params = sm_model.params
    pvals  = sm_model.pvalues

    # turn into long form
    stats_df = (
        params
        .reset_index()  # index (const & features) → column named 'index'
        .melt(
            id_vars="index",
            var_name="class",
            value_name="coef"
        )
    )
    pval_df = (
        pvals
        .reset_index()
        .melt(
            id_vars="index",
            var_name="class",
            value_name="p_value"
        )
    )

    # merge coefficients + p‑values, rename feature column, drop intercept
    stats_df = (
        stats_df
        .merge(pval_df, on=["index", "class"])
        .rename(columns={"index": "feature"})
    )
    stats_df = stats_df[stats_df.feature != "const"].reset_index(drop=True)

    results["statsmodels"] = stats_df

    return results



def print_top_features(results, top_n=10):
    for model in ["logistic", "mlp", "xgboost"]:
        if model in results:
            print(f"\n=== Top {top_n} features for {model.upper()} ===")
            print(results[model]["feature_weights"].head(top_n).to_string(index=False))
    if "statsmodels" in results:
        print(f"\n=== Top {top_n} features by STATS MODELS Logistic Regression (with p-values) ===")
        print(results["statsmodels"].head(top_n).to_string(index=False))


def plot_volcano(statsmodels_df, top_n_labels=20, title="Volcano Plot"):
    df = statsmodels_df.copy()
    df['log_p'] = -np.log10(df['p_value'].clip(lower=1e-10))
    plt.figure(figsize=(10, 6))
    plt.scatter(df['coef'], df['log_p'], alpha=0.6, s=30)
    plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    top = df.sort_values('log_p', ascending=False).head(top_n_labels)
    for _, row in top.iterrows():
        plt.text(row['coef'], row['log_p'], row['feature'], fontsize=9, ha='center', va='bottom')
    plt.xlabel("Coefficient (log-odds)")
    plt.ylabel("-log10(p-value)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    characteristic = "sex"

    df = load_data(model_name, characteristic)
    print(df.head())

    results = probe(df, mode="content", max_features=200)
    print("Logistic Acc & CI:", results["logistic"]["mean_acc"], results["logistic"]["ci"])
    print_top_features(results, top_n=10)

    results_stop = probe(df, mode="stopwords", max_features=200)
    print_top_features(results_stop, top_n=10)
    plot_volcano(results_stop["statsmodels"], top_n_labels=20)
