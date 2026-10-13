# Spam Detector

An email spam classifier built as a scikit-learn pipeline, trained on the Apache SpamAssassin public corpus (2,500 ham and 500 spam emails, included under `datasets/spam/`). This is my solution to exercise 4 in chapter 3 of Aurélien Géron's *Hands-On Machine Learning* (3rd ed.).

## How it works

`spam_detector.ipynb` builds a three-step pipeline:

1. **`EmailToWordCounterTransformer`** (custom): parses each email, converts HTML parts to plain text, optionally strips headers, lowercases, replaces URLs and numbers with placeholder tokens, removes punctuation, stems words with NLTK's Porter stemmer, and counts words.
2. **`WordCounterToVectorTransformer`** (custom): keeps the 1,000 most common words seen in training and turns each email's counts into a sparse vector.
3. **`LogisticRegression`** (`max_iter=1000`). I started with Naive Bayes and switched to logistic regression after it scored better.

The fitted pipeline is saved to `spam_detector_model.pkl` with joblib.

## Results

Re-run in October 2026 with the notebook's 80/20 split (`random_state=42`, 600 test emails):

| Metric | Value |
|---|---|
| Precision (spam) | 96.88% |
| Recall (spam) | 97.89% |
| 3-fold CV accuracy on the training set | 98.5% |

Only 1 in 6 emails is spam, so precision and recall say more than accuracy here. These are "easy ham" emails from 2002, so expect lower numbers on modern mail.

## Run it

```bash
pip install scikit-learn nltk urlextract scipy numpy pandas joblib jupyter
jupyter notebook spam_detector.ipynb
```

The dataset is already in the repo. If `datasets/spam/` is missing, the first cell downloads it from spamassassin.apache.org.
