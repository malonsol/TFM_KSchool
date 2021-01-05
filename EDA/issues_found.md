# General:
- I would like to study two cornerstones of commercial aviation industry, ultimately resulting in a **Fuel Consumption Regressor** and a **Delay Classifier**.
- However, the most interesting data is restricted to each Airline (hereafter *A/L*). For a while now, data is worth its weight in gold, hence the reluctance of A/L to divulge their operating data.
  - The best data I've been able to gather is, curiously, that coming from the US Department of Transportation's BTS (Bureau of Transportation Statistics). Unfortunately, most relevant data has been extracted from one of the databases we used during the Master: *Airline On-Time Performance*. **Would this be deemed as unworthy for being repetitive or lacking creativity?**
  - Even disregarding the previous potential issue, I'd still **lack data about fuel consumption PER FLIGHT**. Looking on the bright side, the silver lining is that I can retrieve through another database from the same provier (BTS), relevant fuel consumption data at a higher aggregation level, according to a particular carrier and time (year/quarter/month). → [Schedule P-12(a) Fuel](https://www.transtats.bts.gov/Tables.asp?DB_ID=135&DB_Name=Air%20Carrier%20Financial%20Reports%20%28Form%2041%20Financial%20Data%29)


## Work already done:
- [ALL Airlines Fuel Cost and Consumption, January 2000 - October 2020](https://www.transtats.bts.gov/fuel.asp) → Aggregation level: each record represents a year with the all-carriers domestic/intenational/total fuel cost and consumption
- [Schedule B-43 Inventory](https://www.transtats.bts.gov/Tables.asp?DB_ID=135&DB_Name=Air%20Carrier%20Financial%20Reports%20%28Form%2041%20Financial%20Data%29) → Aggregation level: each record represents an A/C. At the moment:
  - It has been reduced to a dataset containing all the A/C operating currently
  - The data has been cleaned so the two major manufacturers worldwide (i.e. Airbus and Boeing) can be easily filtered.
  - On top of it, an additional column has been created to identify the two main commercial airlifter families (i.e. A320 and B737). This could be used later on to quickly group results discriminating by these families.
- [Schedule P-12(a) Fuel](https://www.transtats.bts.gov/Tables.asp?DB_ID=135&DB_Name=Air%20Carrier%20Financial%20Reports%20%28Form%2041%20Financial%20Data%29) → Aggregation level: each record represents an A/L reported fuel consumption/cost by time (year/quarter/month). It contains a lot of different columns concerning fuel, so it has been narrowed down to totals corresponding to scheduled/non-scheduled, domestic/international fuel cost/consumptions. Two sets of graphs are provided to better visualize the data:
  - **Evolution over the years**
  - **Top10 carriers by total fuel consumption in 2019**


## Future lines of work:
- **Short-term milestones:**
  - Inner-joined database with A/L as the aggregation level, and considering year 2019. It would consist of:
    - A/L On-Time Performance → corresponding to 2019 and grouped by Carrier, showing several aggregation functions according to the type of data and later use: 
        ```python
          .agg({'ORIGIN_CITY_MARKET_ID' : ['count'],
              'ORIGIN' : ['count'],
              'DEST_CITY_MARKET_ID' : ['count'],
              'DEST' : ['count'],
              'DEP_DELAY' : ['sum', 'mean', 'min', 'max', 'median'],
              'TAXI_OUT' : ['sum', 'mean', 'median'],
              'TAXI_IN' : ['sum', 'mean', 'median'],
              'ARR_DELAY' : ['sum', 'mean', 'min', 'max', 'median'],
              'CANCELLED' : ['sum', ('mode', mode)],
              'DIVERTED' : ['sum', ('mode', mode)],
              'ACTUAL-CRS_ELAPSED_TIME' : ['sum', 'mean', 'min', 'max', 'median'],
              'AIR_TIME' : ['sum', 'mean', 'median'],
              'FLIGHTS' : [('mode', mode)],
              'DISTANCE' : ['sum', 'mean', 'median']
        ```
    - Schedule P-12(a) Fuel → corresponding to 2019 (and already aggregated at A/L level from the start.
    - In case of having fuel consumption data at FLIGHT LEVEL, a third dataset would be merged (***Schedule B-43 Inventory***) in order to allocate a type of aircraft to the Tail Number (present in the first dataset).
      ### ***This additional feature could greatly contribute to improving the model.***
  - Visualize some interesting data coming from the above merged dataset.
- **Medium-term milestones:**
  - Apply Machine Learning / Deep Learning techniques in order to obtain relevant models which aim at predicting the initially proposed objectives:
    - **Fuel Consumption Regressor**
    - **Delay Classifier**
- **Long-term milestones:**
  - Identify relevant insights and conclusions to include later on in an eye-catchy, user-friendly frontend (Streamlit?)
  - Include interactive maps like [these ones with Altair](https://altair-viz.github.io/gallery/airport_connections.html)
- Further exploration:
  - [P52_op_expenses_US_certified](https://www.transtats.bts.gov/Fields.asp?Table_ID=297&SYS_Table_Name=T_F41SCHEDULE_P52&User_Table_Name=Schedule%20P-5.2&Year_Info=1&First_Year=1990&Last_Year=2020&Rate_Info=0&Frequency=Quarterly&Data_Frequency=Annual,Quarterly) → Aggregation level: each record represents an A/C
    - Direct Operating Expense: Flying operations, Direct Maintenance, Depreciation, Amortization, ...
  - [P7_op_expenses_by_groupings_US_certified](https://www.transtats.bts.gov/Fields.asp?Table_ID=278&SYS_Table_Name=T_F41SCHEDULE_P7&User_Table_Name=Schedule%20P-7&Year_Info=1&First_Year=1990&Last_Year=2020&Rate_Info=0&Frequency=Quarterly&Data_Frequency=Annual,Quarterly) → Aggregation level: each record represents an A/C
    - Direct Operating Expense: Totals only
    - Indirect Operating Expense: Passenger Service, Aircraft Servicing, Traffic Servicing, Reservation And Sales, Advertising And Publicity, ...




## DOUBTS:
#### B43_inventory:
  - Too many MANUFACTURER values, many of which are essentially the same: compliacated cleaning.
  - Same issue with MODEL values, but with a whole lot more different values.
  
    **QUESTION:** Is it worth it to spend much time cleaning it? Should I stick with most used models (e.g. B737, A320, etc.) and evaluate carrriers which only use those?
    
  - If I stick with these two maufacturers (i.e. Airbus and Boeing), the dataset is reduced from 100494 to 57794 rows only.
    - On the one hand, I'd be dropping half the dataset, thus losing A/C models variability.
      - For example, most turboprops (and even turbojets) would be excluded from this analysis, since Airbus and Boeing mainly manufacture turbofan-engined aircraft. Manufacturers like Embraer
    - On the other hand, the analysis would be much clearer as the assessment would focus on the two worldwide major manufacturers only.
