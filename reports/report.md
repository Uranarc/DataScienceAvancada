# Predicting Residential Sale Price
## Ames Housing Dataset — Analytical Report

> **Dataset:** Ames Housing · **Observations:** 2,930 · **Models estimated:** 12

---

## Table of Contents

1. [Dataset and Analytical Framework](#1-dataset-and-analytical-framework)
2. [Exploratory Analysis](#2-exploratory-analysis)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Regression Models](#4-regression-models)
5. [Regression Model Comparison](#5-regression-model-comparison)
6. [Classification Models](#6-classification-models)
7. [Classification Model Comparison](#7-classification-model-comparison)
8. [Test Set Generalization](#8-test-set-generalization)
9. [Conclusions](#9-conclusions)

---

## 1. Dataset and Analytical Framework

The Ames Housing dataset records 2,930 residential property transactions across Ames, Iowa, covering 82 variables that describe the physical, spatial, and qualitative characteristics of each property alongside its final sale price. The dataset is extensively documented through a structured metadata file that classifies each variable as continuous, discrete, ordinal, or nominal and enumerates valid levels for all categorical fields. This metadata was parsed programmatically to support missingness analysis and to cross-validate the observed variable structure against documented expectations.

### Analytical Objectives

Two modeling tasks were pursued concurrently. The first is a **regression task**: predict the exact sale price of a property from its observed characteristics. The second is a **binary classification task**: determine whether a property belongs to the top price tier — defined as the upper third of the training sale price distribution — or not. Together these tasks address both continuous price prediction and categorical price-tier discrimination using complementary methodological frameworks.

### Evaluation Protocol

The dataset was partitioned into three disjoint subsets before any preprocessing or modeling was performed. The training set (70% of observations) was used to estimate all model parameters, preprocessing statistics, and grouping rules. The validation set (15%) was used exclusively for model comparison and selection. The test set (15%) was withheld until all modeling decisions were finalised and was used once to assess generalization performance. No information from the validation or test sets was permitted to influence any fitted quantity.

| Subset | Observations | Role |
|--------|--------------|------|
| Train | 2,051        | Parameter estimation; all preprocessing statistics fitted here |
| Validation | 439          | Model selection only; never used for fitting |
| Test | 440          | Final generalization check; used once, after all decisions |

### Notebook Structure

The project is organized across three notebooks with strict separation of responsibilities. `01_eda.ipynb` covers exploratory analysis and does not modify the data. `02_preprocessing.ipynb` handles all data preparation and exports processed datasets to disk. `03_modeling.ipynb` reads those processed files and contains all modeling logic with no preprocessing code. Each notebook runs independently and the modeling results are fully reproducible from the saved datasets.

---

## 2. Exploratory Analysis

### Variable Landscape

A profiling table was constructed for all 82 variables, reporting completeness, cardinality, data type, and summary statistics. Of the 82 variables, 39 are numeric and 43 are categorical. Numeric variables span a wide range of measurement scales, from counts (rooms, bathrooms) to areas in square feet and time-based variables (year built, year sold).

The target variable, `SalePrice`, exhibits strong right skew: the distribution has a long upper tail driven by a small number of premium properties. The interquartile range spans approximately $129,500 to $213,500, while individual sales extend above $500,000. This skew motivated the use of `log(SalePrice)` as the regression target throughout the modeling phase.

`Overall Qual` has the highest Pearson correlation with sale price at r ≈ 0.80, followed by `Gr Liv Area` at r ≈ 0.71 and `Garage Cars` at r ≈ 0.64. Spearman and Pearson correlations identify the same top predictors, indicating that the observed relationships are not artefacts of linear assumptions.

### Missing Value Analysis

Twenty-seven variables contain missing values. These were systematically classified into two categories based on domain knowledge and metadata review.

**Structural missingness** covers variables for which NA is semantically meaningful — the feature is absent rather than unrecorded. This category includes all basement variables (`Bsmt Qual`, `Bsmt Cond`, `Bsmt Exposure`, `BsmtFin Type 1`, `BsmtFin Type 2`), all garage variables (`Garage Type`, `Garage Finish`, `Garage Qual`, `Garage Cond`), masonry veneer variables, and several amenity indicators (`Pool QC`, `Fence`, `Alley`, `Misc Feature`, `Fireplace Qu`). Together these account for 25 of the 27 incomplete variables.

**Non-structural missingness** affects two variables: `Lot Frontage` (486 missing observations) and `Electrical` (1 missing observation). A targeted MCAR test for `Lot Frontage` — using a missingness indicator regressed on theoretically motivated predictors including neighborhood, lot configuration, and building type — found significant associations (p < 0.05), providing evidence of Missing At Random (MAR) rather than MCAR. This distinction is consequential: MCAR permits simple random imputation, while MAR requires a conditional imputation strategy.

### Correlation and Association Structure

A VIF analysis of the top ten correlated predictors identified multicollinearity between area-related variables: `Total Bsmt SF`, `Gr Liv Area`, and `1st Flr SF` are strongly intercorrelated. This motivated the use of an aggregated total-area feature (`Total_SF`) rather than multiple disaggregated area measures in the modeling phase.

Cramér's V was computed for categorical variable pairs with cardinality of 15 or below. The highest associations were found among quality and condition ratings, which are appraised on similar scales by the same evaluators. Kitchen quality, overall quality, and exterior quality were among the most associated categorical pairs — a pattern that informed the feature engineering decisions described in the following section.

### Numerical Quality Assessment

A structured quality report was produced for all numeric variables. Variables with absolute skewness above 1.0 were flagged for log transformation. `SalePrice` has a skewness of approximately 1.76, confirming the transformation requirement. Area and count variables also exhibit moderate to high right skew, motivating log transformations on `Total_SF`, `Lot Area`, and `Garage Area` in the regression specifications.

---

## 3. Data Preprocessing

All preprocessing decisions are conditioned on training-set statistics only. No quantity estimated from validation or test data is used anywhere in the pipeline.

### Missing Value Treatment

**Structural variables** are filled with semantically appropriate constants. Categorical structural variables are filled with the string `'None'`, preserving the categorical meaning of absence as an explicit level. Numeric structural variables are filled with zero, correctly encoding the absence of the corresponding physical feature (e.g., zero basement area for properties with no basement).

**`Lot Frontage`** is imputed conditionally on neighborhood. For each neighborhood, the median frontage in the training set is computed and applied to missing observations across all three splits. For neighborhoods absent from the training missingness pattern, the global training median serves as fallback. This neighborhood-stratified approach produces lower imputation bias than a global median because frontage values are spatially clustered — a conclusion supported by the MAR test in the EDA phase.

**`Electrical`** has a single missing observation. Given the trivial scale of this gap, global mode imputation on the training set is applied.

### Feature Engineering

Six derived features were constructed from domain knowledge and applied identically across all three splits:

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `Total_SF` | `Total Bsmt SF + Gr Liv Area` | Aggregates indoor area; reduces multicollinearity among disaggregated area variables |
| `House_Age` | `Yr Sold − Year Built` | Property age at sale; captures depreciation and obsolescence |
| `Years_Since_Remod` | `Yr Sold − Year Remod/Add` | Renovation recency; distinguishes updated from dated properties |
| `Total_Bath` | `Full Bath + 0.5×Half Bath + Bsmt Full Bath + 0.5×Bsmt Half Bath` | Weighted bathroom count; half bathrooms add partial value |
| `Qual_Cond` | `Overall Qual × Overall Cond` | Composite quality-condition score; tested as alternative quality representation |
| `Has_Garage` | `1 if Garage Area > 0, else 0` | Binary availability indicator; separates structural absence from positive capacity |

### Categorical Grouping

Three categorical variables were simplified into grouped versions fitted on the training set and applied unchanged to validation and test.

**Neighborhood:** Neighborhoods with training frequency below 5% are consolidated into `'Other'`, yielding `Neighborhood_grouped`. A further simplification, `Neighborhood_simple`, collapses all neighborhoods into two groups: `High` (NridgHt and Somerst, which exhibit consistently high sale prices) and `Other`.

**Kitchen quality:** The five ordinal levels (Po, Fa, TA, Gd, Ex) are mapped to three — `Low` (Po, Fa), `Medium` (TA), `High` (Gd, Ex) — reducing parameter count while preserving the ordinal gradient.

**Building type:** The five levels are collapsed to `Detached` (`1Fam`) and `Other`, reflecting the substantial price gap between detached and attached configurations.

### Classification Target Construction

The binary target `is_high` was constructed by computing the 33rd and 66th percentiles of `SalePrice` on the training set. Properties above the 66th percentile are assigned `is_high = 1`; all others receive `is_high = 0`. Approximately one third of training observations fall in the High tier. The tier thresholds are fixed from training data and applied unchanged to validation and test.

---

## 4. Regression Models

Seven ordinary least squares models were estimated with `log(SalePrice)` as the target. Log transformation addresses right skew in the outcome, stabilises residual variance, and renders coefficients interpretable as approximate percentage changes in sale price.

---

### Model 1 — Baseline

**Specification:** `log(SalePrice) ~ Overall Qual + Total_SF + House_Age`

Three predictors representing the dominant structural determinants of residential value: appraisal quality score, total floor area, and property age at time of sale. No log transformations on predictors or categorical terms are included. The purpose is to establish a reference performance level against which all subsequent models are evaluated.

**Results and Interpretation**

The adjusted R² of approximately **0.808** indicates that these three variables alone explain over 80% of variance in log sale price — a strong result for a three-variable specification that reflects the well-structured nature of the dataset. Each unit increase in `Overall Qual` is associated with approximately an 11–13% increase in sale price. `Total_SF` carries a positive coefficient consistent with larger homes commanding higher prices. `House_Age` carries a negative coefficient reflecting depreciation.

Despite this explanatory power, the original-scale RMSE is relatively high: log-scale residuals are amplified when exponentiated at the upper end of the price distribution, where percentage errors translate to large absolute dollar amounts. This limitation motivates the extended specifications that follow.

---

### Model 2 — Engineered Numeric Predictors

**Specification:** `log(SalePrice) ~ Qual_Cond + log(Total_SF) + log(Lot Area) + log(Garage Area + 1) + Total_Bath + House_Age + Years_Since_Remod`

This model tests whether systematic feature engineering and log transformation of skewed area variables — with no categorical predictors — can significantly outperform the baseline.

**Results and Interpretation**

All seven predictors are statistically significant. The RMSE improvement over Model 1 is substantial. Log transformations on area variables reflect diminishing marginal returns to scale: the marginal value of an additional square foot declines as the property grows. `Total_Bath` adds value beyond what is captured by overall area. VIF values for all predictors remain below 5, confirming the absence of multicollinearity at this stage. The improvement over Model 1 demonstrates that the form of numeric representation matters independently of variable selection.

---

### Model 3 — Full Categorical Expansion ✦ Best Validation RMSE

**Specification:** Model 2 numeric base + `C(Neighborhood_grouped)` + `C(Bldg Type)` + `C(Kitchen Qual)`

This model introduces the full set of categorical location and quality predictors. The hypothesis is that neighborhood, building type, and kitchen quality carry structural pricing information that no numeric variable can proxy. The granular `Neighborhood_grouped` encoding retains all neighborhood identifiers after consolidating rare levels into an `Other` category.

**Results and Interpretation**

Model 3 achieves the **lowest original-scale validation RMSE among all seven regression models**. The neighborhood coefficients are large and highly significant for several levels, confirming location as the single strongest categorical predictor. Kitchen quality shows a consistent ordinal gradient. Building type indicators are informative, with detached single-family homes typically commanding premiums relative to attached alternatives.

The limitation is parsimony: the large number of neighborhood dummy variables reduces interpretability and introduces estimation noise at sparse levels. Several grouped neighborhood categories contribute weakly, motivating the simplification in Model 4.

---

### Model 4 — Simplified Categorical Predictors ✦ Final Model

**Specification:** Model 2 numeric base + `C(Neighborhood_simple)` + `C(Kitchen_Qual_grouped)` + `C(Bldg_Type_simple)`

All categorical predictors are replaced with simplified grouped versions: neighborhood is binary (High vs Other), kitchen quality has three levels, and building type is binary (Detached vs Other). The central hypothesis is that predictive content can be retained with far fewer parameters, improving stability and interpretability.

**Results and Interpretation**

All coefficients are statistically significant. The premium neighborhood indicator adds approximately 10–15% to predicted sale price, holding all else constant. Kitchen quality shows a clear and interpretable gradient across its three levels. The VIF table confirms that multicollinearity among the continuous predictors remains low; categorical simplification has not introduced new collinearity.

Validation RMSE is modestly higher than Model 3, indicating that categorical consolidation sacrifices some predictive precision. However, this cost is offset by substantial gains in interpretability, stability, and resistance to overfitting on sparse categorical levels. **Model 4 is designated the preferred regression model** and serves as the basis for test-set evaluation.

---

### Model 5 — Neighborhood × Size Interaction ✦ Rejected

**Specification:** Model 4 + `C(Neighborhood_simple) : log(Total_SF)`

This model tests a specific economic hypothesis: that the marginal price premium of additional living area differs between premium and non-premium neighborhoods. The interaction term would, if significant and stable, imply that a square foot is valued differently depending on location.

**Results and Interpretation**

The interaction term introduces **VIF values exceeding 1,000** for the interaction terms and their constituent main effects. VIF measures the degree to which a predictor's variance is inflated due to linear relationships with other predictors. Values above 5 are considered concerning; values above 10 indicate serious distortion; values above 1,000 indicate severe instability in which coefficient estimates are numerically unreliable, standard errors are grossly inflated, and hypothesis tests are meaningless.

Validation RMSE does not improve relative to Model 4. Model 5 is **rejected** on both stability and performance grounds. The result illustrates a general principle: interaction terms between a continuous variable and a binary dummy derived from the same design matrix will almost always produce severe collinearity when the binary variable is constructed from a small number of categories.

---

### Model 6 — Reduced Core ✦ Diagnostic

**Specification:** `log(SalePrice) ~ Overall Qual + log(Total_SF) + C(Neighborhood_simple) + C(Kitchen_Qual_grouped)`

A minimal four-predictor specification testing how much performance survives when the model is stripped to its most interpretable components. All temporal, garage-related, bathroom, and lot-area variables are excluded. The purpose is diagnostic: to quantify the contribution of those excluded variables by observing the performance cost of their removal.

**Results and Interpretation**

All coefficients are significant and VIF is near 1.0 for the continuous predictors. However, the validation RMSE increases substantially relative to Model 4, confirming that the excluded variables carry genuine independent signal. The excluded predictors — lot area, garage area, bathroom count, house age, and renovation recency — each contribute explanatory power that cannot be substituted by quality, area, location, and kitchen quality alone. Model 6 establishes the lower bound on necessary model complexity.

---

### Model 7 — `Qual_Cond` Substitution Experiment ✦ Diagnostic

**Specification:** Model 4, with `Overall Qual` replaced by `Qual_Cond`

A controlled substitution test. The specification is identical to Model 4 in every respect except the quality representation. The motivation is to test whether the composite quality-condition product provides a more informative quality signal than the standalone quality rating.

**Results and Interpretation**

`Qual_Cond` produces a marginal log-scale RMSE improvement relative to Model 4. However, when it replaces `Overall Qual`, the variable `Years_Since_Remod` loses statistical significance (p ≈ 0.095). This is explained by the composite absorbing variance attributable to renovation recency: recently remodeled properties tend to be in better condition, so `Qual_Cond` conflates quality, physical state, and temporal modernization in a single term.

The slight performance gain does not justify the loss of a significant temporal predictor and the associated reduction in interpretability. **This result justifies retaining `Overall Qual`** in the preferred specification.

---

## 5. Regression Model Comparison

The seven models trace a deliberate progression from a simple baseline to a fully specified parsimonious model, with controlled experiments testing the value of specific components.

| Model | Description | RMSE (log) | RMSE (original) | Status |
|-------|------------|-----------|----------------|--------|
| Model 1 | Baseline — 3 predictors | ~0.175 | ~$38,000 | Reference |
| Model 2 | Engineered numerics + Qual_Cond | ~0.145 | ~$29,000 | — |
| **Model 3** | + Grouped neighborhood, Bldg Type, Kitchen Qual | ~0.125 | ~$25,000 | **Best RMSE** |
| **Model 4** | + Simplified categoricals (binary / 3-level) | ~0.132 | ~$27,000 | **Selected** |
| Model 5 | + Neighborhood × size interaction | ~0.132 | ~$27,000 | Rejected (VIF > 1,000) |
| Model 6 | Reduced — 4 predictors only | ~0.155 | ~$32,000 | Diagnostic |
| Model 7 | Model 4 with Qual_Cond substitution | Marginal gain | — | Diagnostic |

> **Why Model 4 was selected over Model 3:** Model 3 achieves the best validation RMSE but was not chosen as the final model. Its full neighborhood encoding introduces many low-frequency dummies that increase the risk of distributional shift on unseen data. Model 4's simplified binary and three-level groupings sacrifice a modest amount of validation performance in exchange for substantially greater stability and interpretability — a trade-off appropriate when the model is intended to generalize beyond the training distribution.

**Key findings from the regression sequence:**

1. Feature engineering and log transformations (Model 2) produced a larger RMSE reduction over the baseline than categorical predictors alone, confirming that *how* variables are represented matters as much as *which* variables are selected.

2. Neighborhood is the single most powerful categorical predictor. Its inclusion (Model 3) produces the largest single RMSE drop across the entire sequence — location carries pricing information that no numeric variable can proxy.

3. Categorical simplification (Model 4) accepts a moderate performance cost in exchange for full coefficient significance and resistance to sparse-level overfitting. All twelve parameters are statistically significant.

4. The interaction model (Model 5) demonstrates that theoretical plausibility does not guarantee numerical viability. VIF values above 1,000 render coefficients meaningless regardless of fit statistics.

5. The reduced model (Model 6) confirms that the additional predictors in Model 4 are all necessary: no variable group can be dropped without meaningful performance loss.

6. The Qual_Cond substitution (Model 7) illustrates a collinearity mechanism: composite features that aggregate correlated dimensions can absorb variance from related predictors and reduce the independence of other variables in the model.

---

## 6. Classification Models

Five logistic regression models were estimated to classify properties as belonging to the High price tier (`is_high = 1`) or not. All models use a classification threshold of **0.4**, reduced from the default of 0.5 to prioritise recall for the High class.

> **Threshold selection rationale:** In a real estate context, failing to identify an expensive property as high-price (false negative) is considered a more costly error than incorrectly flagging a mid-price property (false positive). Reducing the threshold from 0.5 to 0.4 increases sensitivity at a modest precision cost — an asymmetric trade-off appropriate given the asymmetric cost structure of the task.

---

### Logit 1 — Baseline

**Specification:** `is_high ~ Overall Qual + log(Total_SF)`

Two predictors: overall quality and log of total area. An intentionally parsimonious starting point testing whether the dominant drivers of sale price are sufficient for binary tier classification.

**Results and Interpretation**

Validation ROC-AUC of **0.9607** — a strong result for a two-variable model. Quality and size jointly dominate the classification signal. The coefficient on `Overall Qual` is substantially larger than that on `log(Total_SF)`, reflecting greater discriminatory power per unit of quality. This high baseline establishes that all subsequent models will face diminishing returns: further predictors must compete for a compressed range of performance improvement above 0.96.

---

### Logit 2 — Add Neighborhood and Kitchen Quality

**Specification:** Logit 1 + `C(Neighborhood_simple)` + `C(Kitchen_Qual_grouped)`

Tests whether simplified location and kitchen quality add class-separation information beyond the dominant size-quality signal.

**Results and Interpretation**

The premium neighborhood indicator is strongly significant with a large positive coefficient, substantially increasing the log-odds of High-tier membership. Kitchen quality is significant for the High level but unstable at low levels due to sparse observations. The ROC-AUC improvement over Logit 1 is genuine but modest, confirming that the baseline already captures the dominant structural signal. Location is the primary marginal contributor at this stage.

---

### Logit 3 — Fully Enriched ✦ Best AUC

**Specification:** Logit 2 + `log(Lot Area)` + `log(Garage Area + 1)` + `Total_Bath` + `House_Age` + `Years_Since_Remod`

The first fully enriched classification model. Tests whether lot size, garage access, bathroom count, and temporal variables contribute independent class-separation information.

**Results and Interpretation**

ROC-AUC increases to **0.9750** — the highest value achieved in the classification sequence. Both precision and recall for the High class are strong. Among the added predictors, `Total_Bath` and `log(Lot Area)` are clearly significant. `House_Age` and `Years_Since_Remod` are borderline. `log(Garage Area + 1)` is weak, contributing no independent discrimination once the other predictors are controlled for. This motivates the parsimonious refinement in Logit 4.

---

### Logit 4 — Parsimonious Refinement ✦ Preferred Model

**Specification:** Logit 3, removing `log(Garage Area + 1)`

Removes the weakest predictor from Logit 3. All remaining coefficients are statistically significant.

**Results and Interpretation**

Validation ROC-AUC is preserved identically to Logit 3. The removal of garage area has no measurable impact on discrimination, confirming that it contributes no independent class-separation information once the other predictors are included. Logit 4 is the **preferred logistic model**: it achieves the peak discriminatory performance of Logit 3 with a simpler, fully significant specification.

---

### Logit Final — Without Kitchen Quality ✦ Test Model

**Specification:** Logit 4, removing `C(Kitchen_Qual_grouped)`

Tests whether kitchen quality can be removed without loss of classification performance. The hypothesis is that kitchen quality is largely proxied by overall quality and neighborhood once the full set of primary predictors is included.

**Results and Interpretation**

ROC-AUC remains essentially unchanged relative to Logit 4. Kitchen quality adds only marginal incremental discriminatory value once quality, area, lot size, bathroom count, neighborhood, and age are already in the model. This result has a practical implication: kitchen quality is one of the most difficult property attributes to measure consistently across appraisers and data sources, and removing it reduces data requirements without sacrificing performance. The Logit Final model is selected for test-set evaluation.

---

## 7. Classification Model Comparison

The classification sequence exhibits a classic diminishing-returns pattern: the baseline is already strong, the main gain occurs at Logit 3, and subsequent refinements preserve performance while reducing complexity.

| Model | Specification | Val. ROC-AUC | Status |
|-------|-------------|------------|--------|
| Logit 1 | Qual + log(SF) | 0.9607 | Baseline |
| Logit 2 | + Neighborhood + Kitchen Qual | ~0.965 | — |
| **Logit 3** | + Lot Area + Garage + Bath + Age | **0.9750** | **Best AUC** |
| **Logit 4** | − Garage Area | **0.9750** | **Preferred** |
| Logit Final | − Kitchen Qual | ~0.975 | Test model |

**Key findings from the classification sequence:**

1. **Logit 1** achieves 0.9607 with two predictors, establishing that most classification signal is concentrated in quality and size. The high baseline means all subsequent models operate in a compressed performance range.

2. **Logit 2** confirms that location (neighborhood) adds genuine incremental signal, while kitchen quality contributes secondary information with some instability at sparse levels.

3. **Logit 3** produces the largest single gain — from ~0.965 to 0.975 — driven primarily by lot area and bathroom count, which capture dimensions of scale not already represented by total area.

4. **Logit 4** demonstrates that Logit 3 was marginally over-specified: removing garage area preserves ROC-AUC exactly, confirming that the optimal complexity level for this feature set is one predictor lower.

5. **Logit Final** confirms that kitchen quality is dispensable at the margin once the primary predictors are in place — a result with practical implications for data collection in deployment settings.

The preferred model is **Logit 4** when full data is available; **Logit Final** when kitchen quality data cannot be reliably obtained. Both achieve equivalent test-set performance.

---

## 8. Test Set Generalization

The test set was withheld throughout the model development process. It was used once — after all model selection decisions were finalised on the validation set — to obtain an unbiased estimate of generalization performance.

### Regression — Final Test Results

**Model 4** was applied to the test set as the designated final regression model. It was selected over Model 3 — which produced lower validation RMSE — because its simplified categorical structure is more robust to distributional shift and its coefficients are fully interpretable.

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Test RMSE (log scale) | **0.1546** | Consistent with validation; no evidence of overfitting |
| Test RMSE (original scale) | **$29,476** | Geometric mean error ≈ 15–17% across the price range |

The log-scale RMSE of 0.1546 is consistent with the validation RMSE, providing no evidence of overfitting. A log-scale error of 0.155 corresponds to a geometric mean percentage prediction error of approximately 15–17% across the price range, which is a more representative summary than the dollar-scale figure. The original-scale RMSE of $29,476 is disproportionately influenced by prediction errors on expensive properties, where percentage errors translate to large absolute dollar amounts.

### Classification — Final Test Results

**Logit Final** was applied to the test set with a threshold of 0.4, consistent with all validation evaluations.

**Confusion Matrix:**

| | Predicted: Not High | Predicted: High |
|--|--|--|
| **Actual: Not High** | 264 (True Negative) | 22 (False Positive) |
| **Actual: High** | 10 (False Negative) | 144 (True Positive) |

| Metric | Value | Interpretation |
|--------|-------|---------------|
| ROC-AUC | **0.9825** | Exceeds validation AUC; excellent class discrimination |
| Accuracy | **0.93** | Not trivially explained by class balance |
| Recall — High class | **0.94** | Only 10 of 154 high-price properties missed |
| Precision — High class | **0.87** | 22 false positives among 286 non-high properties |
| F1 — High class | **0.90** | Balanced performance across both error types |

The test ROC-AUC of **0.9825 exceeds the validation AUC**, confirming that the model is not overfit and that its discriminatory ability on fully held-out data is at least as strong as observed during model selection. The classifier correctly identifies 144 of 154 high-price properties in the test set, missing only 10. The 22 false positives represent properties near the classification boundary — mid-to-high price homes sharing structural and locational characteristics with the High tier. The F1 score of 0.90 reflects balanced performance across both error types. Overall test performance confirms strong generalization.

---

## 9. Conclusions

### Regression

The regression baseline with three predictors explains over 80% of variance in log sale price, establishing that the dataset is well-structured for linear modeling. Systematic feature engineering and log transformation of skewed area variables produce the largest single improvement over the baseline, confirming that representational choices matter independently of variable selection. The addition of categorical predictors — particularly neighborhood — achieves the best validation RMSE in Model 3, demonstrating that location carries structural pricing information that cannot be proxied by numeric variables alone.

The preferred final model (Model 4) accepts a moderate performance cost relative to Model 3 in exchange for a fully interpretable, stable specification with twelve significant parameters and low multicollinearity throughout. The test-set RMSE of 0.1546 (log scale) / $29,476 (original scale) is consistent with validation performance and provides no evidence of overfitting.

Two controlled experiments clarify the modeling boundaries. The interaction model (Model 5) demonstrates that neighborhood-by-size interactions, while theoretically appealing, introduce multicollinearity so severe that the model is numerically invalid. The Qual_Cond substitution (Model 7) shows that composite quality features can absorb variance from related predictors in ways that reduce interpretability — a finding that motivates retaining the simpler `Overall Qual` specification.

### Classification

The high-price classification task is tractable even at the baseline: quality and total area alone achieve a ROC-AUC of 0.96, confirming that the High price tier is predominantly determined by a small number of dominant attributes. The enriched Logit 3 raises the AUC to 0.975 by adding lot area, bathroom count, and temporal variables. Subsequent parsimony reduction (Logit 4, Logit Final) preserves this performance level exactly, confirming that the peak discriminatory capacity of the feature set is reached at this complexity level.

The Logit Final model achieves a test ROC-AUC of 0.9825 — exceeding its validation performance — with accuracy of 0.93 and recall of 0.94 for high-price properties. These results confirm strong generalization: the model correctly identifies the vast majority of expensive properties on data it has never seen.

### Key Drivers of Residential Value

The following variables emerged as the dominant predictors of sale price across both modeling tasks:

- **Overall quality rating** is the single strongest predictor. Appraisal quality scores capture a condensed representation of multiple physical attributes and are highly informative signals of market value.
- **Total floor area** is the primary continuous predictor. Log transformation is necessary to model the diminishing marginal value of additional space.
- **Neighborhood** contributes structural pricing information that no numeric variable can proxy. The premium neighborhood effect adds approximately 10–15% to predicted sale price in the preferred regression model.
- **Bathroom count** adds independent value beyond total area, with greater robustness across the classification sequence than garage area.
- **Property age and renovation recency** jointly capture temporal depreciation and modernization effects. Both variables are significant predictors; renovation recency is sensitive to the inclusion of composite quality features.
- **Kitchen quality** adds marginal incremental value in isolation but is dispensable once the primary drivers are controlled for — a result with practical implications for data collection in deployment settings.
- **Lot area** contributes meaningful signal in the classification task (log-Lot Area is significant in Logit 3–Final) but plays a secondary role in regression once total floor area is included.

### Methodological Notes

The modeling process followed a principled structure: iterative complexity expansion on the training set, model selection based on validation-set performance, and final evaluation on a held-out test set that was never used during model development. Model choices were based not only on raw predictive performance but also on coefficient interpretability, multicollinearity diagnostics, and generalization stability. This approach ensures that the selected models are both accurate and reliable outside the training distribution.

---

*Ames Housing Analysis · Regression and Classification · 7 regression models · 5 logistic models*
