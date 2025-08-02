# Fairness Evaluation of Large Language Models in Public Libraries 🔍

This repository supports the paper **Fairness Evaluation of Large Language Models in Public Libraries**.
Our project presents an explainable diagnostic framework for auditing the fairness of large language models (LLMs) in
a range of realistic public library scenarios.

In this repo, we provide code, data, and results for analyzing whether LLM-generated outputs differ by
user attributes such as **gender, race/ethnicity, education, and income**.

---

## 🧪 What's in This Repository?

* ✅ **[Fairness Evaluation Protocol (FEP)](probe.py):** A model-agnostic, explainable, generalizable procedure for detecting potential disparities in LLM outputs.
* 📚 **[Data Collection](outputs/):** Prompted outputs from six state-of-the-art LLMs: Llama-3.1 (8B), Gemma-2 (9B), Mistral (8B), GPT-4o, Claude-3.5 Sonnet, and Gemini-2.5 Pro across diverse simulated patron profiles.
* 🦜 **[Patron-LLM Interaction Simulation](run.py):** Script for simulating virtual reference interactions between LLMs and library users with different demographic and socioeconomic profiles. Used to generate outputs for fairness probing.

---

## 🚀 How to Run

1. Install dependencies

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Run the diagnostic classifiers

```bash
python probe.py
```

---

## 📄 License

[MIT License](LICENSE)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue before submitting a pull request.

---