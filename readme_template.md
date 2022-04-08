# Zillow Clustering Project

# Overview
This project seeks to build a model to predict, and determine drivers of, error in Zillow home value predictions using data analysis and statistics testing. The dataset comes from the Codeup data warehouse.

# Plan
1. Wrangle 'zillow' data
    * Limit data to 2017 single family residences
    * Include a few features for minimum viable product (MVP)
    * Tidy up data for analysis
    * Split data into in-sample and out-of-sample data
    * Isolate target from data splits
    * Scale numeric features
    * Merge scaled, unscaled, and target data into an exploration dataframe
2. Explore for feature engineering and clustering
    * Analyze correlations between features and prediction error
    * Identify potential trends for high-error residences (location, home design, home size, etc)
3. Conduct feature engineering and clustering
    * Create features that group metrics associated with high- and low-error predictions
    * Look at latitude and longitude specifically for trends
4. Push multiple approaches to regression shotgun / evaluation
    * modeling.py/regression_shotgun creates multiple regression models with a simple input
    * modeling.py/y_df_RMSE_r2 pushes model performance for RMSE and r^2 of in-sample and out-of-sample data to dataframe
5. Clean up work for presentation
6. Present!

# Results
1. Best-performing model barely beat baseline for engineered features and clustering

# Future
1. Continue to conduct feature engineering to separate high- and low-error properties for modeling

# Data Dictionary
| Feature          | Datatype               | Definition                                |
|:-----------------|:-----------------------|:------------------------------------------|
| Parcel.ID        | 45842 non-null object  | Unique ID for a property                  |
| County           | 45842 non-null object  | LA, Orange, and Ventura Counties          |
| Latitude         | 45842 non-null int64   | Decimal Degrees without the decimal point |
| Longitude        | 45842 non-null int64   | Decimal Degrees without the decimal point |
| Home.Value       | 45842 non-null int64   | Home value in dollars                     |
| Prediction.Error | 45842 non-null float64 | logerror of value predictions vs actuals  |
| Baths            | 45842 non-null float64 | Number of bathrooms                       |
| Beds             | 45842 non-null float64 | Number of bedrooms                        |
| Finished.Area    | 45842 non-null int64   | Size in sqft of home's finished area      |
| is_coastal       | 45842 non-null bool    | True if property is coastal               |
| cool_places      | 45842 non-null bool    | True if property is in high-error zones   |

# Instructions for replicating my work
1. Clone this repository to your local machine
2. Run the notebook final_notebook.ipynb in Jupyter Notebook

# Ideas:
- *Done* is_coastline (for each latitude, westmost 5 properties)
    * Visualize as a 4th hue for coordinate plot, each county + is_coastline
    * Should be considered **a proof of concept** because it is only westmost (properties with coastline to south/north/east are not considered coastline)
    * Work done: 
        * 'relax' the lats and longs to enable an average of 10 properties per latitude (raw latitude was too precise)
        * Start with northmost *coastal* property (more-northern properties aren't coastal)
- *Future?* is_highvalue (against raw home value)
    * Establish what correlates with home value, then look for things that should be low value but still yields high value
    * EX: High-value, small-lot
    * EX: High-value, low-bednbath
- *Future?* is_highvaluecity or is_highvalueneighborhood (regioncityid, high-value cities/neighborhoods on the list)
    * Group by city ID, show average taxvaluedollarcnt, split into "high-cost", "medium cost", "low cost" groupings, use these as a feature