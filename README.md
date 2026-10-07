

Readme · MD
Big Five personality prediction from long texts (Longformer)
Capstone project, Postgraduate Micro-credential in Machine Learning & NLP, University of Antwerp (2025).

Question
Can a long-context transformer predict a writer's Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from the full body of what they write, rather than from single short posts?

Approach
Data: the PANDORA research corpus (Reddit authors with Big Five scores and MBTI types). Not included in this repo; request access from its authors.
Labels: each trait binned into low / medium / high (quantile binning).
Split: author-level 90/10 train/validation split, so no author appears in both sets.
Input: each author's text, with the MBTI type prepended as a short text prefix (e.g. "MBTI: E n F j."), tokenised to 2,048 tokens.
Model: Longformer (allenai/longformer-base-4096) encoder with one linear head producing 5 traits x 3 classes from the CLS token; class-weighted cross-entropy averaged over the five traits.
Training: lr 2e-5, warmup 10%, weight decay 0.01, fp16, early stopping on validation macro F1; A100 GPU in Google Colab, tracked in Weights & Biases.
Earlier iterations compared RoBERTa and an XGBoost baseline (see notebooks).
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
training_script_v11.ipynb: final training script
Training Longformer_v1.ipynb: earlier data preparation, chunking and training run
Evaluation_Longformer_HF.ipynb: evaluation of the published model
Longformer-Based Personality Trait Modeling_0530.pdf: project paper / poster
Model
Published on Hugging Face: https://huggingface.co/Zvoni/longformer-big5-mbti

Ethics
Personality prediction from text is sensitive. This is a research exercise on a public academic corpus, not a tool for profiling individuals. Notebook outputs showing author names and texts have been removed.

Acknowledgements
Prof. Walter Daelemans, Prof. Luna De Bruyne and Jens Van Nooten (University of Antwerp).


