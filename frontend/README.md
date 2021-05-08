<p align="center">
  <img width=447.9 height=100.2 src="https://user-images.githubusercontent.com/71399207/117346880-a28fbd80-aea8-11eb-9074-24ef6fe5d804.jpeg">
</p>
&nbsp;

<h1 style="text-align:center">
  <img width=30 style="float: right;" src="https://docs.streamlit.io/en/0.79.0/_static/favicon.png">
  &#128279 &#128313 &#128311 &#128313 &#9992 &#128104;&#8205;&#9992;&#65039;
  <a href="https://share.streamlit.io/malonsol/tfm_kschool/main/frontend/flight_delay_predictor.py">→ CLEAR SKY ←</a>
  &#128105;&#8205;&#9992;&#65039; &#9992 &#128313 &#128311 &#128313 &#128279
  <img width=30 style="float: right;" src="https://docs.streamlit.io/en/0.79.0/_static/favicon.png">
</h1>
&nbsp;


*An independent README file has been created for the Frontend section in order to tidy up the repo.*  
*:arrow_right_hook: For those users interested in a comprehensive explanation about the underlying project behind the application, it can be found [here](https://github.com/malonsol/TFM_KSchool/blob/main/README.md).*

&nbsp;

## Description

A light web-based application under the name of ***CLEAR SKY*** was both developed and deployed using Streamlit. The main purpose of this service is to ease airlines workload concerning flight planning operations.

I happily encourage everybody to have a look and play with it freely!

Of course, please do not hesitate to contact me should any doubt may arise during tests:
&nbsp;

&nbsp;&nbsp;&nbsp;<img width=20 style="float: right;" src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg">
&nbsp;&nbsp;&nbsp;https://www.linkedin.com/in/marioalonsolopez/

&nbsp;&nbsp;&nbsp;<img width=20 style="float: right;" src="https://www.google.com/gmail/about/static/images/logo-gmail.png?cache=1adba63">
&nbsp;&nbsp;&nbsp;m.alonso.lopez123@gmail.com

**Constructive feedback is also more than welcome!**
&nbsp;

## Quick user guide

There are many different logics applying behind each field. Behaviour is defined in accordance with user sequential inputs. Hereafter are defined some of these dynamics.  
*:arrow_right_hook: Complete code can be checked out [here](./7_frontend.ipynb).*

### Flight data
#### Origin and Destination
- `Carrier` : ordered in accordance with total number of flights in 2019
- `Origin` / `Destination` : user can either select 
  - an airport currently operated by the airline, or
  - any airport in the database
- `Latitude` / `Longitude` : provided for information purposes
- `Taxi-out time` / `Taxi-int time` : 
  - if the airline has flown from/to the airport before, the median of its times is set as default
  - otherwise, a general default value is provided based on the entire dataset
- `Distance` : informative value, calculated based on Origin and Destination geographic location. The distance is computed drawing upon the [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula), which determines the great-circle distance between two points on a sphere given their longitudes and latitudes
#### Time
- `Flight date` : initialized with current date (UTC)
- `Departure time hour` : initialized on 0 by default
- `Arrival time hour` : based on the most frequent value *(mode)* for the combination of: Origin, Destination, Carrier and Departure time hour

### Meteorological data

