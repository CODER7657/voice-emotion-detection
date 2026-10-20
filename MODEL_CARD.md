# Model card: WavLM (frozen) + logistic regression, 8 emotions

The best configuration evaluated in this repository. It is a recipe that the notebook
retrains on every run, not a released checkpoint: no model weights are distributed.

## Model

| | |
|---|---|
| Encoder | [`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus), frozen, never trained here |
| Embedding | Hidden states averaged over time, then over all layers: one 768-value vector per clip |
| Classifier | Standardisation + class-balanced logistic regression (C = 0.5) |
| Input | 16 kHz mono, 3 s window starting 0.5 s into the clip |
| Output | One of: neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Code | `src/vemo/ssl.py`, `src/vemo/models.py`, `src/vemo/experiment.py` |

## Data

[RAVDESS](https://zenodo.org/records/1188976) speech: 1,440 clips from 24 professional
actors (12 female, 12 male), two fixed English sentences, each emotion at two
intensities. Neutral has 96 clips; every other emotion has 192. Licensed CC BY-NC-SA 4.0.

## Evaluation

6-fold cross-validation grouped by actor: each fold tests 4 actors (2 female, 2 male)
that the model never heard, and every clip is tested exactly once.

| Metric | Value |
|---|---|
| Macro-F1 | **0.766 ± 0.052** across folds (range 0.714 to 0.860) |
| Accuracy | 0.767 ± 0.051 |
| Majority-class floor | 0.029 macro-F1 |

Recall per emotion, from all 1,440 held-out predictions:

| angry | calm | disgust | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| 0.83 | 0.77 | 0.77 | 0.83 | 0.76 | 0.80 | **0.58** | 0.82 |

Sad is the weakest class: 22 of its 192 clips are predicted as calm and 24 as fearful.

The same configuration scored 0.875 under the original random split, which puts every
actor in both train and test. The 0.11 gap is how much speaker leakage inflates this model.

Source: `results/compact.json`, one Colab T4 run, seed 0.

## Intended use

Learning and research on speech emotion recognition, and as an honest baseline to
compare other models against.

## Out of scope

- Decisions about people: hiring, education, insurance, policing, border control, or
  any assessment of an individual's mental health.
- Claims about real, spontaneous emotion. Every clip is acted.

## Limitations

- **Acted, scripted speech.** 24 North American actors reading two sentences. Accuracy
  on spontaneous speech, other languages and accents, children, or noisy phone audio
  has not been measured and should be expected to be lower.
- **Not measured by gender.** Folds are gender-balanced, but scores were not broken
  down by speaker gender.
- **Small test folds.** With 4 voices per fold, scores move by about ±0.05 depending on
  which actors are held out.
- **One seed.** Results come from a single run; run-to-run variation is not reported.
