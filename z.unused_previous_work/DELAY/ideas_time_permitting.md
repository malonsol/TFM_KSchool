# Questions to answer:
  - Instead of obtaining the Delay Regressor:
    - Aim at a Delay Binary Classifier (>15min).
    - Aim at predicting the **cause of the delay** (probably a dataset with more features would be required):
      - 'CARRIER_DELAY'
      - 'WEATHER_DELAY'
      - 'NAS_DELAY'
      - 'SECURITY_DELAY'
      - 'LATE_AIRCRAFT_DELAY'

# Visualization:
- [US domestic flight delays by airport with Tableau](https://www.tableau.com/solutions/workbook/big-data-more-common-now-ever?utm_campaign_id=2017049&utm_campaign=Prospecting-ALL-ALL-ALL-ALL-ALL&utm_medium=Paid+Search&utm_source=Google+Search&utm_language=EN&utm_country=USCA&kw=&adgroup=CTX-Trial-Solutions-DSA&adused=DSA&matchtype=b&placement=&gclid=EAIaIQobChMIha-S96Kh7gIVh7HtCh3D5wYREAAYASAAEgIsBfD_BwE&gclsrc=aw.ds)


# Model
- [Target encoding done the right way](https://maxhalford.github.io/blog/target-encoding/):
  - The problem of target encoding has a name: **over-fitting**.
  - There are various ways to handle this:
    - A popular way is to use **cross-validation** and compute the means in each out-of-fold dataset.
    - Another approach which I much prefer is to use **additive smoothing**.
- Which metric is the best to score the Delay Classifier (or Regressor)?
  - [Specifying multiple metrics for evaluation](https://scikit-learn.org/stable/modules/grid_search.html#specifying-multiple-metrics-for-evaluation)
- Use [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearch#sklearn.model_selection.GridSearchCV)? And/or  [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn-model-selection-randomizedsearchcv)?
  - In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter.
- Play with Threshold values to penalize classifying a Delay as On-Time
