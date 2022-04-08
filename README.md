# Zillow Clustering Project 

# Project Description and Goals
In this project, my goals are to determine drivers of error in Zillow home value predictions using data analysis and statistical testing and use this analysis to help build a machine learning model that can predict the error. 

# Plan
1. Wrangle 'zillow' data
   * Restricted data to only include single unit properties.
   * I renamed certain columns for clarity.
   * Dropped columns that I judged unhelpful or unnecessary.
   * Removed outliers that were skewing my data in order to achieve more normal distributions.
   * Changed data types where appropriate for readability or to denote categorical variables. (Floats to ints, ints to strings/objects.)
   * For clearer exploration, I converted the fips column into a county column, replacing the numerical values with the corresponding county names.
   * I also created a new age column calculated from the year_built column.
   * I imputed null values using mean or median for certain continuous variables such as home_sqft
   * I filled in null values for heating_system as 'None'
    
    Isolate target from data splits
    * Scale numeric features
    * Merge scaled, unscaled, and target data into an exploration dataframe
2. Explore for feature engineering and clustering
    * Analyze correlations between features and prediction error
    * Identify potential trends for high-error residences (location, home design, home size, etc)
3. Conduct feature engineering and clustering
    * Create features that group metrics associated with high- and low-error predictions
    * Look at latitude and longitude specifically for trends
4. Create regression models
    * Creates multiple regression models: OLS, LASSO + LARS, Polynomial Regression, Generalized Linear Model
5. Clean up work for presentation
6. Present.


# Hypotheses

1. There is a signifcant difference between the county logerror and the total population logerror.
- Null Hypothesis Rejected
2. There is a linear correlation between property age and logerror.
- Failed to Reject Null Hypothesis Rejected
3. There is a linear correlation between home square feet and logerror.
- Null Hypothesis Rejected
4. There is a linear correlation between building quality and logerror.
- Null Hypothesis Rejected

Features Used in Modeling = ['bathrooms', 'bedrooms', 'building_quality', 'home_sqft',
       'latitude', 'longitude','assessed_value', 'age']

## Data Dictionary

| Feature           | Datatype   | Definition                                                                    |    
|:------------------|:-----------|:------------------------------------------------------------------------------|
| bedrooms          | float64    | Number of bedrooms in the property                                          |
| bathrooms         | float64    | Number of bathrooms in the property                                           |
| age               | object     | Year that the property was built                                              |
| home_sqft         | float64    | Total size of the property in square feet                                     |
| assessed_value    | float64    | Tax assessed value of the property                                            |
| building_quality  | float64    | Assessment of condition of the building from best (lowest) to worst (highest) |
| latitude          | float64    | Latitude of the middle of the parcel                                          |
| logitude          | float64    | Longitude of the middle of the parcel                                         |
| logerror          | float64    | Zestimate error score                                                         |
 
