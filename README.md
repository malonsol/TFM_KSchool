# Predicting Flight Delays in Commercial Aviation

*:arrow_right_hook: (Instructions on how to run the code for the project can be found at the end of this README)*
&nbsp;

&nbsp;


## :books: Abstract

Airborne transportation is widely accepted as a cornerstone in mobility worldwide. Its meaningful impact on the global community is remarkable from a social and economic viewpoint. Societies all across the globe greatly benefit from this industrial activity. However, in order to do so, flight tickets must be fair. Considering the various costs airlines have to face, margins are undoubtedly tight-fitting; so, in order to make it possible for air travelers to fly at a reasonable price, air carriers require minimizing operational costs. There are several challenging tasks an operator has to tackle to maximize its profit, such as optimizing maintenance activities or route planning to minimize fuel consumption.  Among all these tasks, this project will focus on predicting flights delays; in accordance with the definition convened by the Federal Aviation Administration (FAA), a flight is considered to be delayed when it is 15 minutes behind its schedule time. According to the FAA, the total cost of delays from 2016 to 2019 has steadily increased from $23.7 to $33 billion ($5.6 to $8.3 billion corresponding to air carriers) [1]. By addressing the punctuality issue, operators can better understand what the main reasons are behind delays, and set up corrective measures accordingly in advance. Since meteorological predictions are quite accurate nowadays, the proposed model takes these valuable data into consideration in order to generate actionable insights. A comprehensive analysis is undertaken accounting for various predictors besides weather data. In this paper, predictive modelling approaches are applied based on machine learning techniques using publicly available flight and meteorology datasets from the US for year 2019. A binary classifier is built on such data, and optimized using recall as the primary metrics, achieving a value for delayed flights of 0.65, together with 0.70 for on-time flights.

*\[1] Michael Lukacs, FAA - APO-100 Deputy Division Manager, \[2019], Cost of Delay Estimates*
&nbsp;

&nbsp;


## :white_check_mark: Objectives

1. Help airlines predict potential delays in advance so as to minimize costs incurred from unpunctuality (increased operating expenses comprising extra crew, fuel, or maintenance among others).
2. Provide carriers with causality suggestions to better understand the reason behind these delays. This information is gathered through various methods, such as feature importance or SHAP values.
&nbsp;

&nbsp;



## :bar_chart: Relevant figures

### Confusion matrix and Classification report


<center>

|                |     precision    |     recall    |     f1-score    |
|----------------|------------------|---------------|-----------------|
|     on-time    |     0.89         |     0.70      |     0.78        |
|     delayed    |     0.34         |     0.65      |     0.44        |

</center>
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/71399207/117335574-97826080-ae9b-11eb-8db9-fe170161a2e3.png">
</p>

### ROC curve analysis

<p align="center">
  <img src="https://user-images.githubusercontent.com/71399207/117335798-d4e6ee00-ae9b-11eb-8821-2643de56633d.png">
</p>
&nbsp;

&nbsp;



## :bookmark_tabs: Instructions

### 1. Create an environment with the required libraries

This project has been built upon [`Anaconda`](https://www.anaconda.com/distribution/) with Python version 3.8 . However, a superior version of the Anaconda distribution should also work.

Once Anaconda is up and running, the recommended procedure is to create a Conda environment based on the repo's `requirements.txt` file:

To do so, just run the following command on your terminal:

```bash
conda create --name <env> --file src/requirements-conda.txt
```

*(An utterly useful Conda official cheatsheet can be found [here](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf))*

---

### 2. Download the data
The whole project has been zipped and uploaded to Google Drive. Here is the [link](https://drive.google.com/drive/folders/11dHZbCN1WCdrYVgMQM8FwV2x2KzdewXI)  
  
:warning: **RELATIVE PATHS WARNING** :warning:  

Relative paths have been used throughout all the notebooks in order to ensure complete reproducibility. DO NOT change directory arborescence !!!

*NOTE: At certain points in the notebooks, computation times are quite long. Therefore, these are already pre-run so the reader can review their contents without having to run them.* 

---

### 3. Running the Jupyter Notebooks
The proper way to fully reproduce the project is to run the notebooks in the corresponding order. To ease this process, these have been numbered:
- [1_otp_preprocessing](notebooks/1_otp_preprocessing.ipynb)
- [2_wban_iata](notebooks/2_wban_iata.ipynb)
- [3_lcd_datapreparation](notebooks/3_lcd_datapreparation.ipynb)
- [4_otp_lcd](notebooks/4_otp_lcd.ipynb)
- [5.1_model_various_classifiers](notebooks/5.1_model_various_classifiers.ipynb)
- [5.2_model_xgboost](notebooks/5.2_model_xgboost.ipynb)
- [6_shap_values](notebooks/6_shap_values.ipynb)
- [7_frontend](https://github.com/malonsol/TFM_KSchool/tree/main/frontend) → *It has its own self-contained directory within this repo, so as to better organize the workflow*



