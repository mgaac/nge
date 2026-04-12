# Audit Protocol

## Scope

Use this protocol to decide whether a deep learning result supports the claimed capability or is better explained by leakage, shortcuts, confounds, bugs, or overfit evaluation.

## Procedure

### 1. Define the unit of generalization

Identify what must be new at evaluation time:

- sample
- patient/user/site/source
- graph family or generator seed regime
- prompt template or benchmark family
- time period
- modality pairing

If the split does not isolate that unit, the claim is overstated.

### 2. Reconstruct the data lineage

Trace raw source -> preprocessing -> caching -> split assignment -> batching -> augmentation -> evaluation. Look for:

- duplicates or near-duplicates across splits
- transforms fit on all data before splitting
- labels, future information, filenames, IDs, or metadata entering features
- reused templates, generators, or seeds that collapse the effective split

### 3. Inspect the evaluation contract

Verify:

- model-selection data is separate from final reporting data
- early stopping does not peek at test results
- hyperparameters were not tuned on the reported test split
- metrics match the scientific claim
- reported aggregation is not hiding high variance or subgroup failure

### 4. Search for trivial baselines

Ask what a cheap model or heuristic could exploit:

- class priors
- sequence length
- graph size or node ordering
- token formatting
- acquisition site
- source identity
- background texture or watermark

If a trivial baseline can explain most of the gain, the main claim is weak.

### 5. Run falsification tests

Prioritize tests that isolate the suspected shortcut:

| Risk | Test |
| --- | --- |
| duplicate leakage | hash, embedding, or metadata dedup across splits |
| group leakage | group/site/patient/source holdout |
| temporal leakage | chronological split and future-feature audit |
| label leakage | drop proxy columns, metadata, or post-treatment variables |
| shortcut learning | counterfactual edits, occlusion, crop, masking, or feature ablation |
| test-set overfitting | untouched final split or fresh resample |
| instability | multi-seed reruns with spread reported |

### 6. Grade the claim

Use one of these outcomes:

- supported: core claim survives the strongest plausible invalidators
- partially supported: some capability is real, but the original claim is too broad
- unsupported: evidence is insufficient or contaminated
- contradicted: targeted tests falsify the original explanation

## Reporting Standard

For each finding, include:

- the exact claim threatened
- the mechanism of failure
- the artifact or code path that supports the concern
- the decisive next test
- the likely impact on the headline result

## Common anti-patterns

- reporting only the best seed
- citing ablations that preserve the confound
- using the test set repeatedly during debugging
- trusting synthetic-data splits without checking generator reuse
- interpreting attention maps or saliency as proof of causal use
