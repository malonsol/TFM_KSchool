# Questions to answer:
  - Instead of obtaining the Delay Regressor:
    - Aim at a Delay Binary Classifier (>15min).
    - Aim at predicting the **cause of the delay** (probably a dataset with more features would be required):
      - 'CARRIER_DELAY'
      - 'WEATHER_DELAY'
      - 'NAS_DELAY'
      - 'SECURITY_DELAY'
      - 'LATE_AIRCRAFT_DELAY'
    
# Model
- [Target encoding done the right way](https://maxhalford.github.io/blog/target-encoding/):
  - The problem of target encoding has a name: **over-fitting**.
  - There are various ways to handle this:
    - A popular way is to use **cross-validation** and compute the means in each out-of-fold dataset.
    - Another approach which I much prefer is to use **additive smoothing**.
- Which metric is the best to score the Delay Classifier (or Regressor)?
