After the first tutorial session, the project has been reoriented:
- Fuel consumption prediction will be disregarded due to:
  - It is impossible to tackle without A/L data at low aggregation level (fligths).
  - That topic has been broadly investigated, so there is not much room for improvement.
- **Flight delays** will presumably be the central topic of the project:
  - Is it better to address it in a numerical predictor (regressor), or a binomial classifier (yes/no, >15min)?
  - Additional questions to cover, time permitting:
    - When is the best time of day/day of week/time of year to fly to minimize delays?
    - Do older planes suffer more delays?
    - How does the number of people flying between different locations change over time?
    - How well does weather predict plane delays?
    - Can you detect cascading failures as delays in one airport create delays in others? Are there critical links in the system?
    
### [How are these categories defined?](https://www.bts.dot.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations#q1)
- **Air Carrier**: The cause of the cancellation or delay was due to circumstances within the airline's control (e.g. maintenance or crew problems, aircraft cleaning, baggage loading, fueling, etc.).
- **Extreme Weather**: Significant meteorological conditions (actual or forecasted) that, in the judgment of the carrier, delays or prevents the operation of a flight such as tornado, blizzard or hurricane.
- **National Aviation System (NAS)**: Delays and cancellations attributable to the national aviation system that refer to a broad set of conditions, such as non-extreme weather conditions, airport operations, heavy traffic volume, and air traffic control.
- **Late-arriving aircraft**: A previous flight with same aircraft arrived late, causing the present flight to depart late.
Security: Delays or cancellations caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.

### [What have the airline reports on the causes of delay shown about flight delays?](https://www.bts.dot.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations#q3)
![Delay Cause by Year, as a Percent of Total Delay Minutes](https://www.bts.dot.gov/sites/bts.dot.gov/files/Delay%20Cause%20by%20Year%2C%202003-2019%20crop.png)

### [Is it true that weather causes only 4 percent of flight delays?](https://www.bts.dot.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations#q6)
That category consists of **extreme weather that prevents flying**. There is another category of weather within the NAS category. This type of weather slows the operations of the system but does not prevent flying. Delays or cancellations coded "NAS" are the type of weather delays that could be reduced with corrective action by the airports or the Federal Aviation Administration. **During 2019, 56.8% of NAS delays were due to weather. NAS delays were 24.0% of total delays in 2019.**

### [How many flights were really delayed by weather?](https://www.bts.dot.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations#q6)
A true picture of total weather-related delays requires several steps. First, the extreme weather delays must be combined with the NAS weather category. Second, a calculation must be made to determine the weather-related delays included in the "late-arriving aircraft" category. Airlines do not report the causes of the late-arriving aircraft but an allocation can be made using the proportion of weather related-delays and total flights in the other categories. Adding the weather-related delays to the extreme weather and NAS weather categories would result in weather's share of all flight delays.
![Weather's Share of Delay as Percent of Total Delay-Minutes, by Year](https://www.bts.dot.gov/sites/bts.dot.gov/files/Weather%27s%20Pct%20Share%20of%20Delay%2C%202003-%20thru%202019%20%20crop.png)

# Soruces

### [Airline Service Quality Performance 234 (On-Time performance data)](https://www.bts.dot.gov/browse-statistical-products-and-data/bts-publications/airline-service-quality-performance-234-time)
