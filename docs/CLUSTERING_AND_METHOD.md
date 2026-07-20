# Clustering audit and benchmark rationale

## Question

Does the 2024 Finger Lakes SPARCS cohort contain stable, clinically useful
unsupervised clusters that should replace case-mix benchmarking?

## Audit

The cleaned 139,690-record cohort was encoded from outcome-free clinical fields
(age, gender, admission type, diagnosis, severity, mortality risk, MDC, DRG,
medical/surgical class, and ED indicator). A 30-component truncated SVD embedding
was clustered with MiniBatch K-means for k=2 through k=8. Separation was evaluated
on a fixed 12,000-record sample; stability was checked using adjusted Rand index
across two random starts.

| k | Silhouette | Davies-Bouldin | Seed stability ARI |
|---:|---:|---:|---:|
| 2 | 0.194 | 2.046 | 1.000 |
| 3 | 0.121 | 2.612 | 0.979 |
| 4 | 0.139 | 2.423 | 0.357 |
| 5 | 0.150 | 2.173 | 0.576 |
| 6 | 0.175 | 2.114 | 0.340 |
| 7 | 0.148 | 2.220 | 0.495 |
| 8 | 0.160 | 2.187 | 0.518 |

The only stable solution was a weak two-group split: emergency/older/medical
care (66.4%) versus elective/maternity/newborn care (33.6%). It primarily
rediscovered known administrative categories rather than novel phenotypes.
Clustering actual log-LOS and log-cost produced a clearer two-tier utilization
split (silhouette 0.491), but this is circular and cannot support prediction.

## Decision

Opaque cluster labels are not used in the product. The clustering result instead
motivates seven mutually exclusive, outcome-free service lines. Expectations are
generated separately within each service line, facility is excluded from the
model, and candidate facility×service-line and facility×DRG findings must meet:

- at least 100 cases;
- a positive 95% uncertainty signal;
- Benjamini-Hochberg FDR q ≤ 0.05;
- conservative ranking by the smaller of the confidence-bound and 1%-trimmed
  robust cost-gap estimates.

Top-10% residual concentration labels each finding as broad-based, mixed, or
outlier-concentrated. These are investigation signals, not estimates of causal
waste or realizable savings.

## Related work

- [SPARCS LightGBM LOS prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC9448550/)
- [Hospital latent-class patient segmentation](https://pubmed.ncbi.nlm.nih.gov/35612177/)
- [Clustering persistent high utilizers](https://pubmed.ncbi.nlm.nih.gov/34592712/)
- [Observed-predicted LOS as an efficiency indicator](https://pubmed.ncbi.nlm.nih.gov/15102334/)
- [Impact of LOS outliers](https://pmc.ncbi.nlm.nih.gov/articles/PMC8427900/)

The current audit is exploratory and limited to one year. A defensible external
validation requires another year and, ideally, structural hospital covariates.
