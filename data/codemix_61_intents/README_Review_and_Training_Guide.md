# CodeMix Support AI — 61-Intent Starter Dataset

## What this is
A synthetic, human-review-required starter dataset for a customer-support intent classifier. It contains 61 intent labels and 549 message rows:
- 61 intents
- 3 scenarios per intent
- Each scenario represented in English, Tanglish, and Hinglish
- 183 rows per language
- 2 scenario groups per intent marked `train_candidate` and 1 marked `validation_candidate`

## Important limitations
This is NOT a perfect, production-ready or independently validated dataset. The messages were authored as candidate examples and must be reviewed by fluent Tanglish and Hinglish speakers before training. Code-mixed spelling varies by person and region; some messages may be unnatural, ambiguous, or overlap with other intents. Do not describe this dataset as real customer data.

## Leakage prevention
`scenario_group` links translations/paraphrases of the same scenario across languages. Keep every row with the same `scenario_group` in the same split. Do not randomly split individual rows, because translations of the same scenario could leak across training and validation. For a final test set, create new, independently authored examples that are not derived from these scenarios.

## Recommended workflow
1. Review labels and phrasing with fluent speakers.
2. Correct or remove unnatural examples; add authentic spelling variation and short informal messages.
3. Add realistic multi-intent messages and hard negatives only with clear annotation rules.
4. Preserve existing app labels or create a documented mapping; don't silently rename labels.
5. Train a baseline and compare it against the current model on the same held-out test set.
6. Report macro-F1, per-intent precision/recall/F1, confusion matrix, and separate English/Tanglish/Hinglish results.
7. Use a human-review fallback for low-confidence or sensitive cases.

## Columns
- `id`: unique row ID
- `message`: customer message
- `language`: English, Tanglish, or Hinglish
- `expected_intent`: target label
- `intent_category`: broad grouping for analysis only; not a prediction label
- `scenario_group`: keeps equivalent scenario versions together for splitting
- `split_candidate`: suggested development split; not a final evaluation split
- `source`: synthetic authored candidate, not real customer data
- `review_status`: review required before training
- `review_notes`: reviewer notes

## Intent-label note
This starter taxonomy includes 61 intents. Some are new relative to the project's original 27 labels. Before integration, agree on a label mapping and update the app, model, reply templates, and tests consistently. Similar labels such as `order_status_inquiry` and `track_order`, or `payment_failed` and `payment_deducted_order_failed`, need explicit annotation guidance.
