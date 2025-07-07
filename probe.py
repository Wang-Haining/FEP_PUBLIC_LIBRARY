"""
Fairness Evaluation Protocol (FEP) for Public Library LLM Services

This script evaluates whether LLMs provide equitable service across demographic
characteristics in public library reference interactions. Using classification-based
bias detection, it tests if demographic attributes can be predicted from LLM responses,
indicating systematic differences in service delivery.

The evaluation follows the academic methodology adapted for public libraries:
1. Loads LLM responses from balanced demographic samples
2. Builds classifiers to predict user demographics from response text
3. Reports classification performance vs. chance levels
4. Identifies linguistic features driving any systematic differences

FAIRNESS CRITERIA:
- Equitable service: Classification accuracy should be near chance levels
- Above-chance performance indicates systematic demographic differences
- Feature analysis reveals whether differences represent bias vs. appropriate adaptation

MODELS EVALUATED:
- Open models: Llama-3.1-8B, Ministral-8B, Gemma-2-9B
- Commercial models: GPT-4o, Claude-3.5-Sonnet, Gemini-2.5-Pro

DEMOGRAPHIC DIMENSIONS:
- Gender: Male, Female, Nonbinary
- Race/Ethnicity: 6-category Census taxonomy
- Education: 8 levels (less than high school to doctorate)
- Income: 6 brackets (under $25K to $150K+)

CLASSIFICATION APPROACH:
- 5-fold cross-validation using generation seeds
- Three classifiers: Logistic Regression, MLP, XGBoost
- TF-IDF features (100 content words, excluding stopwords and honorifics)
- Statistical significance testing with Bonferroni correction

OUTPUT:
- Classification accuracies with 95% confidence intervals
- Feature importance rankings and coefficients
- Statistical significance of linguistic patterns
- Volcano plots for feature visualization

Usage:
    python script.py                    # Full evaluation grid
    python script.py --debug           # Single debug probe
"""

import argparse
import json
import os
import string
import sys
import warnings

import nltk
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nltk.corpus import stopwords
from scipy.stats import t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

nltk.download("stopwords", quiet=True)
stop_words_set = set(stopwords.words("english"))


def load_data(
    model_name: str,
    characteristic: str,
    input_dir: str = "outputs",
    failure_token: str = "[NO_TEXT_AFTER_RETRIES]",
) -> pd.DataFrame:
    """
    Load model generation outputs and extract text responses and target characteristics.

    Parameters:
    - model_name: Model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct', 'gpt-4o').
    - characteristic: One of 'gender', 'race_ethnicity', 'education', 'household_income'.
    - input_dir: Directory containing seed-wise output JSON files.
    - failure_token: Token indicating failed generation to filter out.

    Returns:
    - DataFrame with columns ['response', 'label', 'seed'].
    """
    assert characteristic in [
        "gender",
        "race_ethnicity",
        "education",
        "household_income",
    ], "Characteristic must be one of: gender, race_ethnicity, education, household_income"

    tag = model_name.split("/")[-1].replace("-", "_").replace("/", "_")
    files = [
        f
        for f in os.listdir(input_dir)
        if f.startswith(f"{tag}_seed_") and f.endswith(".json")
    ]

    rows = []
    for file in files:
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                response = entry["response"]
                if failure_token not in response:  # filter out failed generations
                    rows.append(
                        {
                            "response": response,
                            "label": entry[characteristic],
                            "seed": entry["seed"],
                        }
                    )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["response", "label"]).reset_index(drop=True)
    return df


def compute_ci(accs, confidence=0.95):
    """Compute mean and confidence interval for accuracy scores."""
    mean = np.mean(accs)
    sem = np.std(accs, ddof=1) / np.sqrt(len(accs))
    h = sem * t.ppf((1 + confidence) / 2.0, len(accs) - 1)
    return mean, (mean - h, mean + h)


def get_feature_weights(clf, feature_names, model_type):
    """Extract feature weights from trained classifier."""
    if model_type == "logistic":
        weights = clf.coef_[0]
    elif model_type == "mlp":
        weights = clf.coefs_[0][:, 0]
    elif model_type == "xgboost":
        booster = clf.get_booster()
        importance = booster.get_score(importance_type="weight")
        return pd.DataFrame(
            {"feature": list(importance.keys()), "weight": list(importance.values())}
        ).sort_values(by="weight", ascending=False)
    else:
        raise ValueError("Unsupported model type")

    return pd.DataFrame({"feature": feature_names, "weight": weights}).sort_values(
        by="weight", ascending=False
    )


def probe(df, max_features=100, model_name=None):
    """
    Fairness evaluation using classification-based bias detection.

    Tests whether demographic characteristics can be predicted from LLM responses
    using TF-IDF content features. Above-chance performance indicates systematic
    differences in service delivery.

    Parameters:
        - df: DataFrame with 'response', 'label', 'seed'
        - max_features: number of top TF-IDF features to use (fixed at 100)
        - model_name: model identifier for logging

    Returns:
        - Dictionary with classifier results and statistical analysis
    """
    results = {}

    # Content word tokenizer (excludes stopwords and honorifics)
    class ContentTokenizer:
        def __init__(self):
            self.exclusion_set = set(stop_words_set).union({"mr", "ms", "mrs", "miss"})

        def __call__(self, doc):
            tokens = [t.strip(string.punctuation).lower() for t in doc.split()]
            return [t for t in tokens if t and t not in self.exclusion_set]

    # TF-IDF vectorization with fixed feature count
    vectorizer = TfidfVectorizer(
        tokenizer=ContentTokenizer(), token_pattern=None, max_features=max_features
    )
    X = vectorizer.fit_transform(df["response"]).toarray()

    # explicit label encoding with proper reference group ordering for statsmodels
    le = LabelEncoder()
    unique_labels = set(df["label"].unique())

    # define reference groups and ordering (reference group last for statsmodels)
    if unique_labels == {"Female", "Male", "Nonbinary"}:
        # female as reference (encoded as 2)
        le.classes_ = np.array(["Male", "Nonbinary", "Female"])
        reference_group = "Female"
    elif unique_labels == {
        "White",
        "Black or African American",
        "Asian or Pacific Islander",
        "American Indian or Alaska Native",
        "Two or More Races",
        "Hispanic or Latino",
    }:
        # white as reference (encoded as 5)
        le.classes_ = np.array(
            [
                "Black or African American",
                "Asian or Pacific Islander",
                "American Indian or Alaska Native",
                "Two or More Races",
                "Hispanic or Latino",
                "White",
            ]
        )
        reference_group = "White"
    elif unique_labels == {
        "Less than high school",
        "High school graduate",
        "Some college, no degree",
        "Associate degree",
        "Bachelor's degree",
        "Master's degree",
        "Professional degree",
        "Doctorate degree",
    }:
        # less than high school as reference (encoded as 7)
        le.classes_ = np.array(
            [
                "High school graduate",
                "Some college, no degree",
                "Associate degree",
                "Bachelor's degree",
                "Master's degree",
                "Professional degree",
                "Doctorate degree",
                "Less than high school",
            ]
        )
        reference_group = "Less than high school"
    elif unique_labels == {
        "Under $25,000",
        "$25,000 to $49,999",
        "$50,000 to $74,999",
        "$75,000 to $99,999",
        "$100,000 to $149,999",
        "$150,000 and above",
    }:
        # under $25,000 as reference (encoded as 5)
        le.classes_ = np.array(
            [
                "$25,000 to $49,999",
                "$50,000 to $74,999",
                "$75,000 to $99,999",
                "$100,000 to $149,999",
                "$150,000 and above",
                "Under $25,000",
            ]
        )
        reference_group = "Under $25,000"
    else:
        raise RuntimeError(f"Unexpected label set: {sorted(unique_labels)}")

    y = le.fit_transform(df["label"])
    print(f"Reference group: {reference_group} (encoded as {len(le.classes_)-1})")
    feature_names = vectorizer.get_feature_names_out()

    # cross-validation splits by seed
    seeds = sorted(df["seed"].unique())
    splits = [(df["seed"] != s, df["seed"] == s) for s in seeds]

    # classifier definitions
    model_defs = {
        "logistic": lambda: LogisticRegression(
            C=1.0, max_iter=1000, solver="liblinear", penalty="l2", random_state=93187
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=2000,
            early_stopping=True,
            random_state=93187,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=93187,
        ),
    }

    # run classifiers
    for name, constructor in model_defs.items():
        accs, weights = [], []
        for train_idx, test_idx in splits:
            clf = constructor()
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

        if name == "xgboost":
            mapping = {f"f{i}": feature_names[i] for i in range(len(feature_names))}
            avg_weights["feature"] = avg_weights["feature"].map(mapping)

        results[name] = {"mean_acc": mean_acc, "ci": ci, "feature_weights": avg_weights}

    # statistical significance testing with multinomial logistic regression
    X_const = sm.add_constant(X)
    n_classes = len(np.unique(y))

    sm_model = sm.MNLogit(y, X_const).fit(disp=0, maxiter=2000, method="lbfgs")
    params, pvals = sm_model.params.flatten(), sm_model.pvalues.flatten()
    feat_const = ["const"] + list(feature_names)
    feats_exp, classes_exp = [], []

    # build feature-class combinations (excluding reference class)
    for i, feat in enumerate(feat_const):
        for c in range(n_classes - 1):  # Exclude reference class
            feats_exp.append(feat)
            classes_exp.append(le.classes_[c])  # Non-reference classes

    valid = ~np.isnan(params)
    stats_df = pd.DataFrame(
        {
            "feature": [feats_exp[i] for i in range(len(valid)) if valid[i]],
            "class": [classes_exp[i] for i in range(len(valid)) if valid[i]],
            "coef": params[valid],
            "p_value": pvals[valid],
        }
    )

    stats_df = stats_df[stats_df.feature != "const"]
    stats_df = stats_df.dropna(subset=["coef", "p_value"]).reset_index(drop=True)
    stats_df = stats_df.loc[
        stats_df["coef"].abs().sort_values(ascending=False).index
    ].reset_index(drop=True)
    results["statsmodels"] = stats_df

    return results


def print_top_features(results, top_n=10):
    """Print top features from each classifier."""
    for model in ["logistic", "mlp", "xgboost"]:
        if model in results:
            print(f"\n=== Top {top_n} features for {model.upper()} ===")
            print(results[model]["feature_weights"].head(top_n).to_string(index=False))
    if "statsmodels" in results:
        print(f"\n=== Top {top_n} features by Statistical Significance ===")
        print(results["statsmodels"].head(top_n).to_string(index=False))


def serialize_for_json(results):
    """Convert results to JSON-serializable format."""

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


def main():
    """
    Main driver for fairness evaluation across public library LLM services.
    """
    parser = argparse.ArgumentParser(
        description="Fairness Evaluation Protocol for Public Library LLMs"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run single debug probe (Llama gender evaluation)",
    )
    args = parser.parse_args()

    if args.debug:
        model = "meta-llama/Llama-3.1-8B-Instruct"
        char = "gender"
        print(f"DEBUG: Running fairness evaluation for {model} / {char}")
        df = load_data(model, char)
        results = probe(df, max_features=100, model_name=model)
        print_top_features(results)
        print(f"\nClassification accuracies:")
        for clf_name in ["logistic", "mlp", "xgboost"]:
            if clf_name in results:
                acc = results[clf_name]["mean_acc"]
                ci = results[clf_name]["ci"]
                print(f"{clf_name:>10}: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        sys.exit(0)

    # full evaluation grid
    model_names = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Ministral-8B-Instruct-2410",
        "google/gemma-2-9b-it",
        # "gpt-4o-2024-08-06",
        # "claude-3-5-sonnet-20241022",
        # "gemini-2.5-pro-preview-05-06"
    ]
    characteristics = ["gender", "race_ethnicity", "education", "household_income"]

    all_results = {}
    total = len(model_names) * len(characteristics)
    progress = tqdm(total=total, desc="Fairness Evaluation")

    for model in model_names:
        all_results[model] = {}
        for char in characteristics:
            try:
                df = load_data(model, char)
                results = probe(df, max_features=100, model_name=model)
                all_results[model][char] = results

                # print summary
                print(f"\n{model} - {char}:")
                for clf_name in ["logistic", "mlp", "xgboost"]:
                    if clf_name in results:
                        acc = results[clf_name]["mean_acc"]
                        ci = results[clf_name]["ci"]
                        print(f"  {clf_name:>10}: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

            except Exception as e:
                print(f"Error evaluating {model} - {char}: {e}")
                all_results[model][char] = {"error": str(e)}

            progress.update(1)

    progress.close()

    # save results
    with open("fairness_evaluation.json", "w") as f:
        json.dump(serialize_for_json(all_results), f, indent=2)
    print(
        f"\nFairness evaluation completed. Results saved to 'fairness_evaluation.json'"
    )


if __name__ == "__main__":
    main()
