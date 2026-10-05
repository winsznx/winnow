# Winnow

Winnow learns which clips an editor keeps from their own early picks, then ranks the rest of a shoot the way they would, locally, with open models (SigLIP embeddings plus TabPFN v2). It is a Python CLI. It exports ranked markers for Resolve, Final Cut and Premiere, and it never hides or deletes a clip.

## Status: tonight's numbers are from SumMe users, not the friend

The videographer this was built for could not share footage in time. The results below come from SumMe (Gygli et al., ECCV 2014), 25 public videos with 15 to 18 people's keep/skip summaries each. Each of those people stands in for one editor. The friend's own run comes next, through the `winnow run` path below, which has only run on small fixtures and synthetic clips so far.

## Results (SumMe, run 20261005T021310Z, from results/20261005T021310Z/results.json)

Correction: the first run (20261005T020454Z) compared against a crowd arm that used other people's labels on the held-out segments themselves, which no deployed tool has. That arm is kept below as crowd_oracle (not deployable), and matched arms were added (generic, hybrid). Every arm from the first run reproduced exactly. The first run's two NULL verdicts are unchanged and are repeated here.

Setup: 18 videos gave usable pairs. 390 editor-video pairs existed, 156 were used and 234 were skipped (225 with fewer than 3 positives in the seed block, 9 with no positives in the held-out block). Each editor trains on the first 25% of 2-second segments of a video and is scored on recall of their own picks in the rest. Cells are mean recall with a 95% bootstrap CI over pairs.

Arms in plain terms: tabpfn is the personal model (trained only on this editor's seed). generic is logistic regression trained on every user's labels from the other 24 videos, with no labels from this video. TabPFN's 10k-row limit rules out TabPFN for that stacked set of about 33k rows, so it is logistic regression. hybrid is TabPFN on this editor's seed plus the generic score as an extra feature. crowd_oracle uses other users' labels on the held-out segments and is an upper bound, not a tool anyone could ship.

Primary split (time-ordered seed):

| Arm | Recall @10% | Recall @20% | Recall @30% |
| --- | --- | --- | --- |
| chronological | 0.16 [0.13, 0.19] | 0.27 [0.23, 0.31] | 0.35 [0.31, 0.40] |
| crowd_oracle (not deployable) | 0.39 [0.35, 0.43] | 0.56 [0.52, 0.61] | 0.67 [0.63, 0.71] |
| duration_only | 0.10 [0.08, 0.12] | 0.22 [0.19, 0.25] | 0.31 [0.29, 0.34] |
| generic | 0.16 [0.13, 0.19] | 0.30 [0.27, 0.34] | 0.41 [0.38, 0.45] |
| hybrid | 0.17 [0.14, 0.20] | 0.27 [0.24, 0.31] | 0.38 [0.34, 0.42] |
| knn | 0.10 [0.08, 0.12] | 0.18 [0.15, 0.22] | 0.27 [0.24, 0.31] |
| logreg | 0.18 [0.14, 0.21] | 0.30 [0.26, 0.34] | 0.38 [0.34, 0.42] |
| sharpness_stability | 0.12 [0.09, 0.14] | 0.21 [0.18, 0.25] | 0.30 [0.27, 0.34] |
| shuffled_tabpfn | 0.13 [0.10, 0.16] | 0.21 [0.18, 0.25] | 0.31 [0.27, 0.34] |
| tabpfn | 0.16 [0.13, 0.19] | 0.26 [0.23, 0.30] | 0.38 [0.34, 0.42] |
| tech_only_tabpfn | 0.12 [0.10, 0.14] | 0.25 [0.21, 0.28] | 0.35 [0.32, 0.39] |

Verdicts at 20% review depth, pre-registered in results/20261005T021310Z/preregistration.json before the run:

- learns_personal_taste: **NULL** (tabpfn minus crowd_oracle (not deployable) at 20% depth, 95% CI lower bound > 0; difference -0.299, 95% CI [-0.351, -0.247], by-video CI [-0.353, -0.234])
- beats_similarity: **NULL** (tabpfn minus knn at 20% depth, mean >= 0.10; difference +0.080, 95% CI [+0.042, +0.119], by-video CI [+0.026, +0.126])
- personal_adds_value: **NULL** (hybrid minus generic at 20% depth, 95% bootstrap CI lower bound > 0; difference -0.033, 95% CI [-0.075, +0.011], by-video CI [-0.085, +0.017])

Generic versus chronological at 20%: +0.032, 95% CI [-0.019, +0.081].

What this says:

- All three verdicts on the primary split are NULL. Adding the editor's own seed labels to a generic model did not help (0.27 hybrid against 0.30 generic).
- The generic model found 0.30 of an editor's picks at 20% depth against 0.27 for chronological order. The CI includes zero, so generic taste from other videos is not clearly better than watching in order.
- The crowd_oracle number (0.56) shows how much of a person's picks agree with other people's picks on the same segments. It does not measure anything a tool could do.
- TabPFN is within noise of its shuffled-label control (0.21) and of logistic regression (0.30) on the primary split.

Secondary split (random 25% seed, not time-ordered, 139 pairs from 20 videos, 251 skipped):

| Arm | Recall @10% | Recall @20% | Recall @30% |
| --- | --- | --- | --- |
| chronological | 0.15 [0.13, 0.18] | 0.28 [0.24, 0.32] | 0.43 [0.38, 0.47] |
| crowd_oracle (not deployable) | 0.35 [0.32, 0.39] | 0.52 [0.49, 0.56] | 0.65 [0.61, 0.69] |
| duration_only | 0.11 [0.09, 0.13] | 0.21 [0.19, 0.24] | 0.31 [0.29, 0.34] |
| generic | 0.15 [0.13, 0.18] | 0.27 [0.24, 0.30] | 0.37 [0.34, 0.41] |
| hybrid | 0.29 [0.26, 0.32] | 0.45 [0.41, 0.48] | 0.56 [0.52, 0.59] |
| knn | 0.20 [0.17, 0.23] | 0.31 [0.28, 0.35] | 0.43 [0.39, 0.47] |
| logreg | 0.31 [0.28, 0.34] | 0.46 [0.42, 0.49] | 0.57 [0.53, 0.61] |
| sharpness_stability | 0.12 [0.10, 0.14] | 0.20 [0.18, 0.23] | 0.30 [0.26, 0.33] |
| shuffled_tabpfn | 0.13 [0.11, 0.16] | 0.24 [0.21, 0.28] | 0.33 [0.30, 0.37] |
| tabpfn | 0.29 [0.26, 0.32] | 0.45 [0.41, 0.49] | 0.56 [0.52, 0.60] |
| tech_only_tabpfn | 0.22 [0.19, 0.25] | 0.34 [0.31, 0.37] | 0.45 [0.42, 0.49] |

- learns_personal_taste: **NULL** (tabpfn minus crowd_oracle (not deployable) at 20% depth, 95% CI lower bound > 0; difference -0.072, 95% CI [-0.118, -0.024], by-video CI [-0.143, -0.004])
- beats_similarity: **PASS** (tabpfn minus knn at 20% depth, mean >= 0.10; difference +0.138, 95% CI [+0.099, +0.178], by-video CI [+0.076, +0.206])
- personal_adds_value: **PASS** (hybrid minus generic at 20% depth, 95% bootstrap CI lower bound > 0; difference +0.177, 95% CI [+0.127, +0.226], by-video CI [+0.120, +0.229])

Read this secondary table with care. A random seed puts neighbouring segments of the same shot on both sides of the split, so a model can score well by recognising near-duplicates of a segment it saw labelled. That is the leakage the time-ordered split exists to block. kNN, logistic regression and TabPFN all jump on this split (0.31, 0.46, 0.45), which fits that explanation and does not show learned taste. The primary split is the headline.

## Why it runs locally with open models

Footage never leaves the machine. Rerunning on another hour of footage costs nothing. The ranker is a file the editor keeps, can inspect and can retrain. Embeddings come from SigLIP through open_clip and the ranker is TabPFN v2 (Apache 2.0 weights). There are no network calls after the model download.

## Reproduce

```
uv venv --python 3.11 && uv pip install -e .
# download and unzip SumMe into ./data (see license below)
winnow bench summe --data ./data
```

For a real shoot:

```
winnow run --clips ./footage --timeline ./final.fcpxml --seed-fraction 0.25
```

`run` writes out/markers.csv, out/markers.fcpxml, out/model.pkl and out/report.md, plus a results.json under runs/. The timeline can be FCPXML, Premiere XML or a CMX3600 EDL. The run stops if more than 10% of timeline sources cannot be matched to clip files.

## Dataset license

SumMe: Videos: not owned by the authors, provided for non-commercial research purposes only. Annotations: free to use for research purposes. Cite Gygli et al., ECCV 2014. Source: https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip. The official project page is offline, so these terms come from the dataset's own README.txt and an archived copy of the page (Wayback, April 2023). The videos are not redistributed here and are gitignored.

## Limitations

- SumMe summaries are made by viewers shown the whole video with a 15% length cap, which is not the same as an editor's final cut of a shoot.
- Short videos give seeds with too few positives, so about 60% of editor-video pairs were skipped on the primary split. Results cover the longer videos.
- PCA uses at most 64 components and is capped at seed size minus one, so seeds under 65 segments get fewer.
- Pairs from the same video share clips, so the pair-level CIs are narrower than the video-level CIs, which are printed alongside in results.json.
- The generic arm is a single logistic regression with default settings and was not tuned. A stronger generic model could change the hybrid comparison.
- The real-user path matches timeline sources to clip files by filename stem, then reel name. It has not run on real footage yet.
- The marker CSV and FCPXML have not been import-tested in Resolve, Final Cut or Premiere.
- One dataset, category-level evidence, no claim about any individual editor.

## Deadline note

The Hacktoberfest Weekend Challenge closes at 06:59 UTC on Oct 5, 2026. Any commit made after that time was made after the deadline.
