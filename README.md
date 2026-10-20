# Movie Review Sentiment Classification with DistilBERT

A course project that fine-tunes DistilBERT to classify movie reviews as positive or negative, then explores how the classifier handles new sentences, negation, and sarcasm.

**Author:** Jianing Fan · **Status:** course project with saved training and prediction outputs

[Open notebook](sentiment_prediction_system.ipynb) · [Run in Google Colab](https://colab.research.google.com/github/brucefjn/ML-Sentiment-Analysis-/blob/main/sentiment_prediction_system.ipynb) · [Portfolio](https://brucefjn.github.io/)

## What I built

- A preprocessing workflow that removes missing reviews and labels, encodes sentiment labels, and creates a stratified 80/20 split.
- A fine-tuning pipeline using Hugging Face Transformers and Datasets, PyTorch, and the `distilbert-base-uncased` checkpoint.
- Evaluation with accuracy and macro-F1, plus sentence-level predictions with class probabilities and an interactive text prompt.
- A small error-analysis exercise examining comparative sentiment, negation, sarcasm, and contrastive phrasing.

Reviews are tokenized to a maximum of 256 tokens. Training uses two epochs, a learning rate of `2e-5`, and batches of 16. The notebook records a Google Colab T4 GPU environment.

## Saved results

The committed notebook contains a completed training run on 19,990 reviews and evaluation on 4,998 reviews, from 24,988 labeled rows after preprocessing.

| Metric | Saved evaluation result |
| --- | ---: |
| Accuracy | 91.38% |
| Macro-F1 | 91.38% |

These values come from the notebook's saved outputs; they have not been independently reproduced for this README. The evaluation split also guides model selection during training, so these are validation results rather than an untouched test-set benchmark.

Saved inference examples include:

| Input | Prediction |
| --- | --- |
| “This movie is not as good.” | Negative |
| “Absolutely fantastic — I loved it.” | Positive |

## Run the notebook

1. Open the Colab link above and select a GPU runtime.
2. Run the cells in order. The notebook downloads a CSV from its Google Drive link and installs `transformers`, `datasets`, `accelerate`, `torch`, and `scikit-learn`.
3. Complete preprocessing and training before running the prediction examples or entering a review at the interactive prompt. Trained model weights are not included in this repository.

The input CSV must contain `Review` text and `Emotion` labels (`positive` or `negative`). The notebook does not document the dataset's original source or license; the download link should not be treated as a complete data citation or permission to redistribute it. If the link is unavailable, supply a compatible CSV and update the loading cell.

## Interpretation and next steps

Sentiment classification works well on many direct examples, but the saved stress tests show confidently incorrect predictions for sarcastic reversals such as “Great movie... said no one ever.” Class probabilities should not be interpreted as calibrated certainty.

The recorded run uses Transformers 5.0.0 and includes checkpoint-loading warnings. Useful next steps are pinning dependencies, checking model save/reload consistency, documenting dataset provenance, and separating training, validation, and final testing.
