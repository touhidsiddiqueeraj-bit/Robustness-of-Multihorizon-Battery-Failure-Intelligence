**Abstract.** 
Multihorizon hazard learning has emerged as a promising framework for operational battery reliability estimation, producing calibrated failure probabilities that enable risk-aware dispatch decisions. However, existing validation is performed exclusively on clean laboratory cycling data, leaving the crucial question unanswered: does calibration survive when the model encounters realistic operating conditions? This paper presents the first systematic investigation of calibration robustness under operational distribution shift. Starting from the published XGBoost hazard model trained on NASA 18650 lithium-ion battery data, we generate synthetic perturbed profiles at four severity levels by truncating raw discharge curves to simulate partial cycling at varying depths of discharge, injecting temperature measurement noise, and randomizing rest-period effects. The fixed model's calibrated failure probabilities degrade substantially: Expected Calibration Error (ECE) rises from 0.01–0.03 on clean data to 0.27–0.38 across all severity levels, while AUC drops from 0.985–0.998 to 0.61–0.77. We further show that recalibration on a small sample (10%) of operational data reduces ECE to 0.05–0.09, largely recovering trustworthiness. Our results establish an operational boundary map for multihorizon hazard models, demonstrating that while calibration is not inherently robust to operational shifts, a minimal recalibration strategy suffices to maintain reliability in the field.

## Introduction**

Battery energy storage systems (BESSs) are increasingly dispatched to provide flexibility services in renewable-dominated power systems. Operators must determine not only how much energy a battery can deliver, but whether it can reliably complete a scheduled service within a predefined duration. Conventional battery prognostics focus on state-of-health (SOH) and remaining useful life (RUL), which describe long-term degradation rather than short-term operational safety [shikdar2026learning].

The multihorizon hazard learning framework [shikdar2026learning] reformulates battery prognostics as an operational reliability estimation problem. Instead of predicting lifetime, the model estimates the probability of failure within predefined service horizons and applies isotonic regression calibration to obtain trustworthy risk metrics. When integrated into a reliability-aware dispatch policy, this framework reduced operational failure rates from 10.3% to 2.95% in cross-validated laboratory experiments.

However, a critical gap separates this laboratory validation from field deployment. The model was trained and evaluated exclusively on clean, controlled laboratory cycling data: full charge/discharge cycles at fixed C-rates with consistent rest periods and controlled temperatures. Real BESS operation involves partial charges at varying depths of discharge (DoD), variable and irregular loads, random rest periods, and temperature fluctuations. The calibration quality under this distribution shift is completely unknown. No operator can trust a model whose reliability degrades in unknown ways.

In this paper, we address this gap by systematically testing the robustness of multihorizon failure intelligence to realistic operating conditions. Our contributions are:

-  The first operational boundary map for multihorizon hazard models, characterizing how calibration degrades as operating conditions deviate from laboratory baselines.

-  A synthetic perturbation methodology that generates realistic operational profiles from existing laboratory cycling data, requiring no field data collection.

-  A demonstration that recalibration on a small sample (10%) of operational data largely recovers calibration quality, providing a practical path to deployment.

## Background

### Multihorizon Hazard Learning

The original framework [shikdar2026learning] defines operational failure as the inability of a battery to complete a service commitment within horizon \(H \in \10, 20, 30, 50\\) cycles. For a battery observed at cycle \(t\), a binary operational event is defined as:

$$
y_t,H = \begincases
1, & \textif failure occurs within  (t, t+H], 

0, & \textotherwise.
\endcases
$$

A gradient-boosted tree ensemble (XGBoost) learns the mapping from per-cycle measurements to failure probability:

$$
f_\theta: \mathbfx_t \longrightarrow P_\textfail(t, H)
$$

where \(\mathbfx_t = \\textSOH_t, V_\textavg, I_\textavg, T_\textavg, V_\textmin, \textduration, t\\). The model is trained on 37 NASA 18650 lithium-ion cells using leave-battery-out cross-validation. After training, isotonic regression is applied to calibrate raw probabilities into empirically consistent risk estimates.

### The Calibration Requirement

Probability calibration is essential for operational decision-making. A well-calibrated model satisfies:

$$
P(\textfailure \mid \hatP_\textcal = p) \approx p.
$$

Without calibration, a high AUC score—which only measures rank-ordering—is insufficient for dispatch decisions. When the model is used to gate flexibility participation via a risk threshold \(\tau\), the accuracy of the probability magnitude directly determines operational safety.

## Methodology

### Experimental Design

We evaluate the robustness of a fixed multihorizon hazard model (trained on clean laboratory data) when applied to perturbed operational data. The experiment proceeds in three stages:

#### Model Training

Following the original paper, we train an XGBoost classifier with isotonic calibration on the full NASA dataset (37 cells, 1,028 valid cycles after SOH filtering). An 80/20 cell-level holdout split is used for calibrator fitting. The model remains unchanged throughout the perturbation experiment, serving as the fixed inference engine.

#### Synthetic Perturbation Generation

Raw discharge curves from the NASA dataset (voltage, current, temperature, and time vectors per cycle) are extracted from the .mat source files. For each cycle, we generate a perturbed version simulating non-laboratory operation:

**Partial cycling: The discharge time-series is truncated at a randomly sampled depth of discharge. For each severity level, DoD is sampled uniformly from progressively lower quartiles (75–100% for mild, 55–75% for moderate, 35–55% for severe, and 15–35% for aggressive). Features are recomputed from the truncated curve: duration decreases proportionally, average and minimum voltages increase (the discharge tail is cut), and average temperature drops slightly due to reduced joule heating.

**Temperature noise: Gaussian noise \(\mathcalN(0, \sigma_s)\) is added to raw temperature measurements before averaging, with \(\sigma_s = \0.5, 1.0, 2.0, 3.0\^\circ\)C for severity levels 1–4.

**Rest randomization: Proportional Gaussian noise \(\mathcal{N}(0, \rho_s)\) is applied to duration and average temperature, with \(\rho_s = \{0.01, 0.02, 0.03, 0.05\}\) for severity levels 1–4.

The true battery health (SOH trajectory and failure labels) is preserved from the clean data, isolating the effect of observational distribution shift from degradation model uncertainty. The perturbation cascade is described in Table 1.

#### Evaluation Protocol

For each severity level, we generate five independent perturbed datasets (seeds 42, 123, 456, 789, 101112) for statistical robustness. Each dataset is evaluated using:

-  **Expected Calibration Error (ECE): The mean absolute difference between predicted probability and observed frequency across 10 bins.

-  **Area Under the ROC Curve (AUC): Discrimination between safe and unsafe states.

-  **Brier score: Mean squared probability error.

We then test a recalibration strategy: retraining the isotonic regression on a random 10% sample of the perturbed data and re-measuring ECE.

## Results

### Clean Baseline

On clean NASA data, the trained model achieves strong calibration and discrimination:

-  H=10: ECE=0.010, AUC=0.994, Brier=0.013

-  H=20: ECE=0.031, AUC=0.985, Brier=0.030

-  H=30: ECE=0.013, AUC=0.994, Brier=0.014

-  H=50: ECE=0.023, AUC=0.998, Brier=0.020

These results confirm that the model is well-calibrated on its training distribution, consistent with the original paper's findings.

### Calibration Degradation Under Perturbation

Table  summarizes the primary result at the 20-cycle horizon. The complete dataset covers horizons 10, 20, 30, and 50.

The degradation is substantial and consistent across all severity levels. Expected Calibration Error increases by approximately an order of magnitude (from \(\sim\)0.03 to \(\sim\)0.28), and AUC drops from 0.985 to 0.65–0.74. The severity level has a relatively weak effect once partial cycling is introduced: even mild perturbation (75–100% DoD, severity 1) causes near-maximal degradation.

This saturation occurs because the model relies heavily on the minimum voltage feature, which shifts dramatically even at the mildest perturbation level. In a full discharge, minimum voltage reaches approximately 2.3 V; at 75% DoD truncation, the observed minimum voltage rises to approximately 3.2 V. This shift alters the feature distribution in a way that the fixed calibrator cannot accommodate.

### Recalibration Recovery

Recalibration on a 10% sample of operational data recovers most of the calibration quality. ECE drops from \(\sim\)0.28 to 0.05–0.09 across all severity levels. While ECE does not return to the clean baseline, the recalibrated model produces substantially more trustworthy probability estimates.

The recalibration strategy is practical: isotonic regression requires no labeled training of the base classifier and can be implemented with a small number of operational cycles. This finding suggests that field deployment is feasible if accompanied by a lightweight recalibration step during commissioning.

## Discussion

### Operational Boundary Map

Our results provide the first quantitative characterization of multihorizon hazard model robustness to operational distribution shift. Key findings:

-  **Calibration is fragile. Even minimal partial cycling (75–100% DoD) degrades ECE from 0.03 to 0.29. The fixed isotonic calibrator cannot adapt to shifts in the feature distribution, particularly in voltage-derived features.

-  **Discrimination degrades moderately. AUC drops from 0.985 to 0.74 at mild perturbation and further to 0.65 at aggressive perturbation, indicating that the model's rank-ordering ability is more robust than its probability estimation.

-  **Recalibration is effective. A 10% operational sample recovers ECE to 0.05–0.09, suggesting a practical deployment pathway: deploy with fixed model + recalibrate on initial operational data.

### Implications for Deployment

The results support two deployment scenarios. In the first, operators deploy the published model directly, accepting degraded calibration but using recalibration as a corrective step after collecting modest operational data. In the second, operators retrain the base model on domain-specific data, though this requires more extensive labeled data.

The recalibration strategy is particularly attractive because it requires no modification to the base classifier, no additional feature engineering, and no access to the original training data. A grid operator commissioning a BESS could collect 50–100 operational cycles, recalibrate the isotonic regression, and immediately obtain improved reliability estimates.

### Limitations

Several limitations should be acknowledged. First, our perturbation methodology preserves the true SOH trajectory, which isolates observational shift but does not capture the coupled effect of changed degradation dynamics under realistic operation. A battery undergoing partial cycling would experience different capacity fade, which could further affect model performance. Second, our evaluation uses a single battery chemistry (LCO) under laboratory conditions; field validation across chemistries and use cases is needed. Third, the recalibration test uses a simple 10% random sample; optimal sample size and selection strategies warrant further investigation.

## Conclusion

This paper presents the first systematic robustness evaluation of multihorizon battery failure intelligence under realistic operating conditions. Using synthetic perturbation of laboratory cycling data at four severity levels, we demonstrate that calibration degrades substantially under partial cycling, temperature noise, and rest irregularity: ECE increases from 0.01–0.03 to 0.27–0.38, and AUC drops from 0.985–0.998 to 0.61–0.77. However, we also show that recalibration on a 10% sample of operational data recovers ECE to 0.05–0.09, providing a practical path to field deployment.

Our findings establish that multihorizon hazard models, while not inherently robust to operational distribution shift, can be deployed with a lightweight recalibration strategy. The operational boundary map we provide enables grid operators to understand when the model can be trusted and what corrective measures are needed.