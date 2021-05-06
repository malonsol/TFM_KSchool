# Predicting Flight Delays in Commercial Aviation

## Abstract

Airborne transportation is widely accepted as a cornerstone in mobility worldwide. Its meaningful impact on the global community is remarkable from a social and economic viewpoint. Societies all across the globe greatly benefit from this industrial activity. However, in order to do so, flight tickets must be fair. Considering the various costs airlines have to face, margins are undoubtedly tight-fitting; so, in order to make it possible for air travelers to fly at a reasonable price, air carriers require minimizing operational costs. There are several challenging tasks an operator has to tackle to maximize its profit, such as optimizing maintenance activities or route planning to minimize fuel consumption.  Among all these tasks, this project will focus on predicting flights delays; in accordance with the definition convened by the Federal Aviation Administration (FAA), a flight is considered to be delayed when it is 15 minutes behind its schedule time. According to the FAA, the total cost of delays from 2016 to 2019 has steadily increased from $23.7 to $33 billion ($5.6 to $8.3 billion corresponding to air carriers) [1]. By addressing the punctuality issue, operators can better understand what the main reasons are behind delays, and set up corrective measures accordingly in advance. Since meteorological predictions are quite accurate nowadays, the proposed model takes these valuable data into consideration in order to generate actionable insights. A comprehensive analysis is undertaken accounting for various predictors besides weather data. In this paper, predictive modelling approaches are applied based on machine learning techniques using publicly available flight and meteorology datasets from the US for year 2019. A binary classifier is built on such data, and optimized using recall as the primary metrics, achieving a value for delayed flights of 0.65, together with 0.70 for on-time flights.

*\[1] Michael Lukacs, FAA - APO-100 Deputy Division Manager, \[2019], Cost of Delay Estimates*

## Objectives

1. Help airlines predict potential delays in advance so as to minimize costs incurred from unpunctuality (increased operating expenses comprising extra crew, fuel, or maintenance among others).
2. Provide carriers with causality suggestions to better understand the reason behind these delays. This information is gathered through various methods, such as feature importance or SHAP values.

## Relevant figures

|                |     precision    |     recall    |     f1-score    |
|----------------|------------------|---------------|-----------------|
|     on-time    |     0.89         |     0.70      |     0.78        |
|     delayed    |     0.34         |     0.65      |     0.44        |
