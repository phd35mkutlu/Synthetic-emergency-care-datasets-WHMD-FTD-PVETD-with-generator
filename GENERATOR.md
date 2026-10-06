# Generator

## What produced the released data

```bash
python ierda_realistic_generator.py
```

Run with no arguments. The defaults are the released configuration: 25 000
WHMD, 20 000 FTD and 10 000 PVETD records, seed 42, and `--ierad_path` unset.
The script is self-contained; it imports only pandas, numpy and the standard
library.

Because `--ierad_path` was not supplied, no record of the Integrated Emergency
Response Dataset was read, and the `linked_incident_id` column is empty in all
three released files.

## The other files

`whmd_schema.py`, `ftd_schema.py` and `pvetd_schema.py` define the field
semantics and the region and vehicle taxonomies. They are documentation: the
generator does not import them.

`ierda_integration_schema.py` documents the intended mapping to the Integrated
Emergency Response Dataset.

`ierda_data_generator.py` is an earlier pipeline that reads that dataset through
`kagglehub` and links records to specific incidents. **It did not produce the
released files** and is included only for completeness.

## Labelling rules, as implemented

**FTD.** `maintenance_probability` is computed from the vehicle profile, age,
usage, fuel and battery terms, retained as a column, and the label drawn as
`true_maintenance_needed ~ Bernoulli(maintenance_probability)`. The column
`detected_maintenance_needed` is then drawn from the realised label with
sensitivity 0.80 and a false-positive rate of 0.10, so it is a consequence of
the outcome rather than a measurement of its cause.

**WHMD.** The label is drawn from a latent probability built from an unobserved
health-risk term, age and user type, which is **not** retained in the release.
The column `emergency_probability` is a separate noisy vital-sign anomaly score
and is not the probability the label came from. Vital signs, fall detection,
the SOS button and the ECG flag are each generated conditional on the realised
label.

**PVETD.** A continuous `severity_score` is computed from the crash physics and
thresholded into `true_severity` (Minor below 25, Moderate below 50, Severe
above). `detected_severity` is the same score plus detection noise, thresholded
the same way. `recommended_response` is then drawn **conditional on
`detected_severity` alone**, from fixed probabilities:

| detected_severity | None | Check | Ambulance | Fire_Rescue |
| --- | --- | --- | --- | --- |
| Minor | 0.20 | 0.60 | 0.20 | — |
| Moderate | — | 0.10 | 0.70 | 0.20 |
| Severe | — | — | 0.40 | 0.60 |

The dependence is stochastic, not an identity. Conditioning on
`detected_severity` therefore bounds accuracy at the probability-weighted sum
of the row maxima, not at 1. Note that the literal string `None` is read as a
missing value by pandas; it denotes no dispatch.

Airbag deployment, vehicle operability, fire and fuel-leak detection are each
drawn conditional on `true_severity`, so they are consequences of the outcome.

## Why these rules are stated here

The article argues that whoever publishes a synthetic dataset should release
the labelling rule and retain the latent quantity as a column, because that is
what makes a downstream user's performance figure checkable. These files are
released so that the same standard applies to this work.
