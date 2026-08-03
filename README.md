# no_caps_bro
An ML based business proposal for optimizing HR retention strategies: A data-driven HR analytics project leveraging Machine Learning to predict employee attrition and formulate retention strategies for an IT consulting firm
# HR Attrition Prediction & Retention ROI - Company I

This project builds a machine learning pipeline that predicts employee attrition and turns the model's output into a retention program with a projected ROI. It was built as a consulting-style deliverable for a fictional client, "Company I," an IT services firm dealing with high turnover.

## Why this project

Company I loses employees at a rate that costs it real money: lost billable hours, disrupted client work, and a slow replacement search every time someone leaves. The brief was to act as a new associate at a consulting firm and turn Company I's HR records into something leadership could actually use, not just a model with a good accuracy score.

That meant the project had to go further than training a classifier. It needed a model that flags which employees are at risk and explains why in terms an HR person can act on, a decision threshold chosen for the actual business trade-off instead of the default 0.5, and a retention program costed out well enough that a CFO could poke holes in it and re-run the numbers themselves.

## What's in the notebook

- Exploratory analysis comparing employees who stayed against those who left, on income, age, stress, and workload, before any model is trained.
- Feature engineering based on how HR actually reasons about risk: tenure buckets, the gap between self-reported and organizational stress, income normalized by experience, a satisfaction composite, and an overtime-by-stress interaction.
- A preprocessing pipeline that's fit only on the training split, so nothing about the test set leaks into imputation, scaling, or the one-hot categories.
- A Logistic Regression baseline first, so the extra complexity of LightGBM has to earn its place.
- Hyperparameter tuning with Optuna, seeded so the same run gives the same parameters, threshold, and ROI numbers every time.
- A decision threshold picked from the Precision-Recall curve on out-of-fold predictions, since the default 0.5 cutoff is a bad fit for a dataset where only about 16% of employees leave.
- SHAP explanations at both the global level and per employee, with the per-employee values shown in their real units (job level, stress rating, income) instead of the scaled numbers the model actually trains on.
- Risk tiers built around the model's own threshold rather than arbitrary round numbers, so nobody the model actually flags ends up sitting in the "Low Risk" bucket.
- A financial simulation for a targeted retention program, including a sensitivity table across a few effectiveness assumptions and a break-even point, rather than one number presented as fact.

## Results

The exact numbers move a little each time Optuna re-tunes the model (though the run is seeded, so repeat runs on the same data reproduce the same result). Rather than freeze a number here that could go stale, treat the "Executive Recap" cell at the end of Section 10 as the source of truth. Copy from there into any slide deck or report.

What stays consistent run to run: LightGBM beats the Logistic Regression baseline, the tuned threshold sits well below 0.5 given the class imbalance, and the retention program's Net ROI is comfortably positive even under the more conservative effectiveness assumptions in the sensitivity table.

## Repository layout

```
.
├── no_caps_bro.ipynb   main notebook, run this top to bottom
├── dataset/
│   └── data.csv                   HR dataset, not tracked in this repo
├── artifacts/                     figures generated on each run
└── README.md
```

`data.csv` isn't committed here since it's course/client data. Drop your own copy at `dataset/data.csv` before running; the notebook checks that exact path and will tell you clearly if it's missing.

## Setup

Needs Python 3.10 or newer and Jupyter.

```
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm optuna shap jupyter
```

Or with a requirements file:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
optuna
shap
jupyter
```

## Running it

1. Clone the repo.
2. Put `data.csv` in `dataset/`.
3. Open the notebook: `jupyter notebook hr_attrition_prediction.ipynb`
4. Restart the kernel and run all cells, in order. There's no hidden state and no Colab-specific paths, so this works the same on any machine.
5. It takes a few minutes. The Optuna search is capped on purpose (`OPTUNA_N_TRIALS = 30`, `OPTUNA_TIMEOUT_SECONDS = 180` in Section 1) so the notebook finishes quickly instead of chasing marginal gains.
6. Everything is seeded (`RANDOM_STATE = 42`, including Optuna's sampler), so re-running against the same `data.csv` gives the same result every time.
7. Take the numbers from the "Executive Recap" print block at the end of Section 10 and copy them into the summary at the top of the notebook, or into a slide deck.

## How the notebook is organized

| Section | Content |
|---|---|
| 1 | Setup, logging, seeding, and every tunable assumption in one place |
| 2 | Load the data, encode the target, drop constant/identifier columns, build a data dictionary |
| 3 | Exploratory analysis: class balance, income/age/stress by attrition, top correlations |
| 4 | Feature engineering and the leak-safe preprocessing pipeline |
| 5 | Logistic Regression baseline |
| 6 | LightGBM with Optuna tuning |
| 7 | Threshold selection from the Precision-Recall curve |
| 8 | SHAP: global importance and per-employee explanations |
| 9 | Risk tiers anchored to the model's threshold |
| 10 | ROI simulation, with a sensitivity table and break-even point |
| 11 | Conclusion and recommendations for Company I |

A few cells are marked as business insight prompts and left partly open on purpose. They're meant to be filled in after looking at that run's actual plots and numbers, not pre-written.

## Reading the output

Start with Section 3 before looking at the model at all; the correlation list and comparison plots set up the hypotheses the model either confirms or complicates. In Section 6/7, the baseline-vs-tuned comparison shows whether LightGBM's extra complexity was worth it, and the threshold plot shows where precision and recall actually cross, which is a more useful way to pick a cutoff than defaulting to 0.5.

The SHAP section is meant to be read almost like notes before an HR conversation: the summary plot ranks what matters across all employees, and the waterfall plot for the highest-risk employee breaks down their specific top five factors in real units.

In Section 9, the crosstab of predicted label against risk tier is worth checking directly. It confirms that no one the model actually flags as likely to leave has ended up in the Low Risk bucket, which would otherwise undercut the whole point of the tiers.

For the ROI section, don't quote the single Net ROI figure on its own. The sensitivity table right below it shows how that number changes if the program turns out less effective than assumed, and the break-even line marks the point below which the program stops making financial sense.

## Assumptions worth knowing before presenting this externally

The dataset is a modified version of the public IBM HR Analytics set, not Company I's actual records, so this is a proof of concept, not a production forecast. The replacement cost multiplier, program cost per employee, and assumed effectiveness in Section 1 are estimates based on general industry figures, not Company I's real financials; they're kept as named constants at the top specifically so someone can challenge them and re-run with different numbers in seconds. And the ROI simulation is a projection from the model's probabilities, not a measured outcome from an actual program, which is why it's reported as a range with a break-even point rather than one confident total.

## Deliverables for this assignment

This notebook covers the technical deliverable. Still to produce: the executive presentation (up to 15 pages, PDF), built from this notebook's EDA figures, model metrics, and the Section 10 ROI breakdown, and the reference list for market research sources and any AI chat history used along the way.
