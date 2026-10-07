Big Five personality prediction from long texts (Longformer)

Capstone project, Postgraduate Micro-credential in Machine Learning & NLP, University of Antwerp (2025).

Question

Can a long-context transformer predict a writer's Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from the full body of what they write, rather than from single short posts?

Approach
Data: the PANDORA research corpus (Reddit authors with Big Five scores). Not included in this repo; request access from its authors.
Preprocessing: texts aggregated per author and chunked to 4,096 tokens; each trait binned into low / medium / high.
Baseline: XGBoost on engineered text features.
Model: Longformer (allenai/longformer-base-4096) fine-tuned as a multi-output classifier, with MBTI-derived features injected as an extra signal; compared with RoBERTa baselines.
Training on an A100 GPU in Google Colab; experiments tracked in Weights & Biases.
Results (macro F1)
Trait	Validation	Test
Openness	0.55	0.32
Conscientiousness	0.57	0.24
Extraversion	0.55	0.37
Agreeableness	0.51	0.21
Neuroticism	0.38	0.19
Macro average	~0.51	0.27

The model learns real signal on validation data but generalises weakly to the held-out test set. The gap points to overfitting on a small number of authors (1,568 after filtering) and to noisy self-reported labels. Next steps would be author-level cross-validation, more authors, and calibration.

Files
Training Longformer_v1.ipynb: data preparation, chunking, training
Evaluation_Longformer_HF.ipynb: evaluation of the published model
Longformer-Based Personality Trait Modeling_0530.pdf: project paper / poster
Model

Published on Hugging Face: https://huggingface.co/Zvoni/longformer-big5-mbti

Ethics

Personality prediction from text is sensitive. This is a research exercise on a public academic corpus, not a tool for profiling individuals. Notebook outputs showing author names and texts have been removed.
