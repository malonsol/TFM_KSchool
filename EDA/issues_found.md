## Airline_On_Time_Performance:
  - Very useful information with Flights as the aggregation level. However, for since I'm interested in a **Fuel Consumption Regressor** and **Delay Classifier**, I'd also need data about each flight's fuel.

## B43_inventory:
  - Too many MANUFACTURER values, many of which are essentially the same: compliacated cleaning.
  - Same issue with MODEL values, but with a whole lot more different values.
  
    **QUESTION:** Is it worth it to spend much time cleaning it? Should I stick with most used models (e.g. B737, A320, etc.) and evaluate carrriers which only use those?
    
  - If I stick with these two maufacturers (i.e. Airbus and Boeing), the dataset is reduced from 100494 to 57794 rows only.
    - On the one hand, I'd be dropping half the dataset, thus losing A/C models variability.
      - For example, most turboprops (and even turbojets) would be excluded from this analysis, since Airbus and Boeing mainly manufacture turbofan-engined aircraft. Manufacturers like Embraer
    - On the other hand, the analysis would be much clearer as the assessment would focus on the two worldwide major manufacturers only.
