# POPGym RepeatPrevious Backbone Ablation: Final Local Report

Generated on 2026-05-10 from local checkpoints and metrics.

## Experiment Scope

Environment family: POPGym RepeatPrevious.

Tasks:

- Easy: shorter memory demand.
- Medium: intermediate memory demand.
- Hard: longer memory demand.

Backbones compared:

- GRU: DreamerV3 RSSM baseline, 5 seeds per difficulty.
- S3M / S4D: diagonal SSM replacement for the deterministic state, 5 seeds per difficulty.
- S5: complex-diagonal SSM replacement, 5 seeds per difficulty.
- Transformer-XL: relative-position attention with segment cache, 1 seed per difficulty.

All reported scores are raw POPGym returns where higher is better. In RepeatPrevious, returns are non-positive, so values closer to 0 are better.

## Generated Artifacts

Report folder:

`checkpoints/repeat_previous_reduced_20260504_004958/figures_for_report_4backbones_20260510/`

Key figures:

- `final_eval_by_task.png`: deterministic 20-episode evaluation return.
- `final_train_by_task.png`: final training episode score.
- `learning_curve_Easy.png`, `learning_curve_Medium.png`, `learning_curve_Hard.png`: smoothed training curves.
- `sample_efficiency_auc.png`: AUC of smoothed learning curve over 1M environment steps.
- `compute_efficiency_fps.png`: steady-state training FPS.

Key tables:

- `coverage_table.md`: completed and evaluated seed coverage.
- `final_eval_table.md`: final deterministic evaluation.
- `final_train_table.md`: final training score.
- `aggregate_backbone_table.md`: macro-average across difficulties.
- `sample_efficiency_auc.md`: sample efficiency.
- `compute_efficiency.md`: throughput.
- `pairwise_vs_gru.md`: empirical comparison vs GRU on final training score.

## Headline Results

Final deterministic eval, 20 episodes per checkpoint:

| Task | Best Backbone | Main Read |
|---|---:|---|
| Easy | GRU, -0.3388 | GRU wins clearly on the short-memory setting. |
| Medium | Transformer-XL, -0.4861, n=1 | Treat as suggestive only. Among 5-seed models, GRU, S3M, and S5 are close. |
| Hard | S3M / S4D, -0.4963 | S3M is best by eval mean; GRU is close; S5 is slightly worse. |

Macro-average deterministic eval:

| Backbone | Eval Mean | Train Mean | Median FPS | Seeds |
|---|---:|---:|---:|---:|
| GRU | -0.4462 | -0.4353 | 75.2 | 15 |
| S3M / S4D | -0.5003 | -0.4663 | 57.8 | 15 |
| S5 | -0.5062 | -0.4585 | 89.7 | 15 |
| Transformer-XL | -0.5029 | -0.4954 | 11.8 | 3 |

Pairwise final-training comparison vs GRU:

| Task | S3M vs GRU | S5 vs GRU | Transformer vs GRU |
|---|---:|---:|---:|
| Easy | -0.2167, P=0.080 | -0.1667, P=0.160 | -0.1417, P=0.200 |
| Medium | +0.0500, P=0.760 | +0.0278, P=0.640 | -0.0167, P=0.400 |
| Hard | +0.0739, P=0.640 | +0.0696, P=0.680 | -0.0217, P=0.400 |

## Interpretation

The cleanest defensible result is difficulty-dependent behavior:

- On Easy, the GRU baseline is strongest. This is expected when the memory horizon is short enough that the simple recurrent inductive bias is enough and the larger sequence models do not get much benefit.
- On Medium and Hard, S3M and S5 become more competitive in training. Their final-training scores beat GRU on average, and the pairwise probability of improvement vs GRU is above 0.5.
- Deterministic evaluation is more conservative than final training. On Hard, S3M has the best eval mean, while S5 does not clearly beat GRU in eval despite stronger final-training scores.
- Transformer-XL is computationally expensive in this implementation, around 11.8 FPS versus 57.8 to 89.7 FPS for the other backbones. Because Transformer has only one seed per difficulty, present it as a pilot result, not a statistically comparable result.

## Presentation Structure

1. Motivation: DreamerV3 uses a GRU/RSSM world model; this project asks whether long-memory sequence backbones improve partially observable control.
2. Implementation: shared Dreamer harness with selectable backbones: GRU, S3M/S4D, S5, Transformer-XL.
3. Experiment matrix: RepeatPrevious Easy/Medium/Hard; 1M env steps; 5 seeds for GRU/S3M/S5; 1 seed for Transformer-XL.
4. Main score plot: show `final_eval_by_task.png`.
5. Learning dynamics: show `learning_curve_Easy.png` and `learning_curve_Hard.png` to highlight the Easy-vs-Hard crossover.
6. Sample efficiency: show `sample_efficiency_auc.png`.
7. Compute cost: show `compute_efficiency_fps.png`.
8. Takeaway: GRU is strongest on short-memory Easy; SSM backbones become competitive on harder memory settings; Transformer-XL needs more seeds and optimization before strong claims.
9. Limitations: one task family, Transformer n=1, raw POPGym returns only, no Mamba teammate results included yet.

## Report Wording

Suggested abstract-style result sentence:

"Across POPGym RepeatPrevious, the GRU baseline achieved the strongest deterministic evaluation on the Easy setting, while SSM-style backbones became competitive as the memory difficulty increased. S3M/S4D achieved the best Hard-setting deterministic eval mean, and both S3M/S4D and S5 improved over GRU in final-training score on Medium and Hard. The Transformer-XL pilot did not yet show a clear advantage and was substantially slower, so we treat it as an implementation/compute baseline rather than a final statistical comparison."

