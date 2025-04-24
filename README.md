# Evaluating LLM Fairness in Library Reference Services ⚖

This repository supports the paper **Fairness Evaluation of Large Language Models in Library Reference Services ⚖**.
Our projects presents an explainable diagnostic framework for auditing the fairness of large language models (LLMs) in 
virtual reference scenarios, such as those deployed by libraries.

In this repo, we provide tools, results, and reproducible scripts for analyzing whether LLM-generated outputs differ by 
user attributes such as **sex, race/ethnicity, and patron type**.


---

## 🧪 What’s in This Repository?

- ✅ **[Fairness Probing Framework (FPF)](probe.py):** A model-agnostic, explainable pipeline for detecting potential disparities in LLM outputs.
- 📚 **[Data Collection](outputs/):** Prompted outputs from Llama-3.1 (8B), Gemma-2 (9B), and Ministral (8B) across different user groups.
- 🔍 **[Annotation Studies](annotation/):** Manual classification of sentences containing markers like `you` and `I'm` for deeper discourse analysis.
- 🦜 **[Patron-LMM Interaction Simulation](run.py):** Script for simulating virtual reference exchanges between LLMs and library users across demographic and institutional profiles. Used to generate outputs for fairness probing.

---

## 🚀 How to Run

1. Install dependencies
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Run the probing script
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
