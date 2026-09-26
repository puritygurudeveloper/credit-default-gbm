# creditrisk — cost-sensitive consumer default modelling

A scorecard and a gradient booster, compared the way a credit shop actually
compares them: out of time, calibrated, and scored in dollars rather than in
Gini points.

The headline result is not which model wins. It is that **the decision rule
matters more than the model**. Swapping a logistic scorecard for XGBoost buys
about 0.04% of portfolio profit. Swapping a single portfolio-wide PD cut-off
for a per-loan break-even rule buys roughly ten times that, for either model.

```
                model    auc   gini     ks  brier  obs/pred    psi   approve  bad rate        profit
        boosted_trees 0.7949 0.5899 0.4440 0.0573    0.7934 0.0002    96.74%     5.75%  $31,012,821
            scorecard 0.7915 0.5829 0.4419 0.0577    0.7966 0.0005    96.62%     5.84%  $30,948,740
```

*100,000 loans, 24 monthly vintages, the last 6 held out. Figures are the
per-loan break-even policy on the out-of-time book; reproduce with
`python -m creditrisk.cli run --rows 100000`.*

## Why this problem is not a classification problem

At an 8% base rate, declining every applicant scores 92% accurate and
originates nothing. Accuracy, F1 and Youden's J are all silent on the only
question that matters: *is the marginal approval worth making?*

That question has an answer in dollars. A loan that performs earns an
origination fee plus net interest margin over its expected life; a loan that
defaults loses `LGD x EAD x principal` and keeps the fee. Setting those equal
gives a break-even PD — and because the lender prices in coarse grade bands,
that break-even is wildly different across the book:

| grade | mean APR | break-even PD | realised default rate |
|-------|---------|---------------|----------------------|
| A | 7.3% | 6.4% | 1.3% |
| C | 11.8% | 13.7% | 3.4% |
| E | 17.5% | 21.8% | 8.6% |
| G | 26.4% | 32.3% | 23.0% |

A single cut-off cannot express this. Any threshold tight enough to decline the
uneconomic tail of grade A throws away most of a profitable grade G book, and
any threshold loose enough to keep grade G approves grade A loans that lose
money. **5.2% of the book is uneconomic at its own price**, and a global
threshold cannot isolate it.

So the project ships two policies and prices both:

```
  boosted_trees  global_cutoff        | PD <= 0.3814 | approve 96.97% | bad rate 5.79% | profit $30,899,234
  boosted_trees  risk_adjusted        | per-loan break-even | approve 96.74% | bad rate 5.75% | profit $31,012,821
```

Note the per-loan rule does **not** simply approve less. It reshapes the book:
it declines well-rated loans whose thin margin cannot absorb their risk, and
approves poorly-rated loans whose price can. A policy judged on bad rate alone
would reject it.

## What the data generator is for

Bureau data cannot be redistributed, so the repo generates its own. The
generator is the substantive part of the project, because a synthetic book that
is too clean makes every downstream conclusion vacuous. It reproduces four
properties that make credit data hard:

- **A latent factor.** Every observable is a noisy read on one unobserved
  creditworthiness axis, so features are strongly correlated and a naive
  "drop collinear columns" step destroys signal.
- **Informative missingness.** Thin-file applicants default at 25.7% against
  7.2% for the rest. Mean-imputing the bureau score erases that and the model
  silently gets worse.
- **Coarse pricing.** Grades come off a legacy rule that sees only the bureau
  score. Within a single priced grade the residual PD dispersion is large and
  entirely unpriced — which is the only reason a model earns its keep.
- **A macro cycle.** Vintages carry a sinusoidal log-odds shift, so the
  out-of-time sample is genuinely a different population.

Calibrated against these, a perfectly specified model reaches **AUC 0.804**.
Both fitted models land at 0.79, which is the right amount of headroom: enough
to be a real problem, not so much that the pipeline is obviously broken.

An earlier version of the generator priced risk almost perfectly from the
latent factor. Every loan then looked profitable, the optimal policy approved
98% of applicants, and the whole decision layer was decorative. The grade-band
rewrite is what made the problem real — that history is in the commit log
because it is the most useful thing in the repository.

## Two models, two representations

The comparison is deliberately asymmetric, because "logistic versus trees" is
not the choice anyone actually faces.

**`Scorecard`** sees weight-of-evidence bins: monotone, missing-as-its-own-bin,
and printable as a points table a credit committee can sign. It carries two
pieces of machinery that exist purely to keep it reviewable:

- *Correlation pruning.* WOE columns above 0.80 correlation are dropped, weaker
  IV first. Collinearity is the standard way a coefficient acquires the wrong
  sign.
- *Direction enforcement.* WOE is signed high-means-safe, so every coefficient
  must be negative. Any feature that fits positive — "safer applicants default
  more" — is dropped and the model refits. On this book `loan_amount` is cut
  every time; its raw IV is 0.18, but conditional on income and price it
  inverts. That is a suppression artefact, not a finding.

**`BoostedTrees`** sees the raw block, missing values included, and early-stops
on the tail of the development window rather than a random slice.

## Where the leaks were

Two look-ahead bugs were caught by tests and are worth naming, because both are
the kind that make a backtest quietly unreproducible:

1. **The threshold saw the answers.** The global cut-off was originally chosen
   by maximising profit on the out-of-time sample and then reported on that
   same sample — so it got to see which loans defaulted. That flattered it by
   enough to reverse the comparison against the per-loan rule. The cut-off is
   now chosen on development data and applied out of time.
2. **A feature was its own input.** `payment_to_income` was computed by backing
   income out of `debt_to_income`, so it reduced algebraically to exactly
   `debt_to_income`. It looked like a strong new feature and carried zero
   information. There is now a test asserting the two are not collinear.

Calibration is fitted on held-back late vintages, never on the data the model
was fitted to — otherwise the Platt slope is 1.0 by construction. The model
still over-predicts out of time (`obs/pred` ≈ 0.79) because the holdout sits on
a different part of the macro cycle. That is the honest number, and it is why
the reliability table ships alongside the AUC.

## Layout

```
src/creditrisk/
  data.py         synthetic book: latent factor, grade pricing, macro cycle
  features.py     WOE binning, information value, correlation pruning
  models.py       Scorecard (WOE + logistic + points) and BoostedTrees
  calibration.py  Platt and isotonic, Brier, ECE, reliability tables
  evaluate.py     AUC / Gini / KS / PR-AUC / lift / PSI, hand-rolled
  economics.py    break-even PD, profit curves, policies, swap sets
  explain.py      tree SHAP, global importance, Reg-B adverse action codes
  pipeline.py     the out-of-time experiment
  cli.py          generate | run
```

Metrics are written out rather than imported. Gini, KS and PSI are the three
numbers in every monitoring pack and libraries disagree on the conventions
(tie handling, which denominator, whether the banding is re-fitted). They are
checked against hand-counted cases and cross-checked against scikit-learn.

## Running it

```bash
pip install -e ".[dev,explain]"

python -m creditrisk.cli run --rows 100000      # the full comparison
python -m creditrisk.cli generate --rows 50000  # just the book, to CSV

pytest -m "not slow"    # ~6s, no model fitting
pytest                  # 78 tests, 93% coverage
```

CI runs lint, mypy, the suite with an 85% coverage floor, and a separate
**model validation** job that performs an out-of-time run on every merge and
uploads the leaderboard. A merge that degrades the model, or breaks the
economic conclusion the credit policy rests on, fails the build rather than
being discovered in a monitoring pack a quarter later.

## Things this deliberately does not do

- **Reject inference.** The book is loans that were already approved under the
  legacy rule, so the model's job here is marginal declines on an
  already-screened population. A 3% incremental decline rate is the realistic
  magnitude; it is not an application-level approval rate.
- **Fairness testing.** No protected attributes are generated, so any
  disparate-impact analysis would be theatre. On a real book this would be a
  gating requirement, not an appendix.
- **Survival modelling.** PD is treated as a fixed-horizon binary outcome.
  Time-to-default with censoring is the better formulation and a larger project.
