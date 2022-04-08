from env import host, user, password, get_db_url
import pandas as pd 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def acquire(use_cache=True):
    '''
    This function takes in no arguments, uses the imported get_db_url function to establish a connection 
    with the mysql database, and uses a SQL query to retrieve telco data creating a dataframe,
    The function caches that dataframe locally as a csv file called zillow.csv, it uses an if statement to use the cached csv
    instead of a fresh SQL query on future function calls. The function returns a dataframe with the telco data.
    '''
    filename = 'zillow.csv'

    if os.path.isfile(filename) and use_cache:
        print('Using cached csv...')
        return pd.read_csv(filename)
    else:
        print('Retrieving data from mySQL server...')
        df = pd.read_sql('''
    SELECT
        prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        landuse.propertylandusedesc,
        story.storydesc,
        construct.typeconstructiondesc
    FROM properties_2017 prop
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid) pred USING(parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE prop.latitude IS NOT NULL AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31';''' , get_db_url('zillow'))
        print('Caching data as csv file for future use...')
        df.to_csv(filename, index=False)
    return df

def attribute_nulls(df):
    nulls = df.isnull().sum()
    rows = len(df)
    percent_missing = nulls / rows 
    dataframe = pd.DataFrame({'rows_missing': nulls, 'percent_missing': percent_missing})
    return dataframe

def column_nulls(df):
    new_df = pd.DataFrame(df.isnull().sum(axis=1), columns = ['cols_missing']).reset_index()\
    .groupby('cols_missing').count().reset_index().\
    rename(columns = {'index': 'rows'})
    new_df['percent_missing'] = new_df.cols_missing/df.shape[1]
    return new_df

def get_single_units(df):
    single_unit = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_unit)]
    return df

def get_hists(df, cols, bins):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = cols

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=bins)

        # Hide gridlines.
        plt.grid(False)

        plt.tight_layout()

    plt.show()
    
def get_box(df, cols):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = cols

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
    
def get_upper_outliers(series, k):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return series.apply(lambda x: max([x - upper_bound, 0]))

def get_lower_outliers(series, k):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return series.apply(lambda x: min([x + lower_bound, 0]))

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df
    
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def encode_categorical(df, cat_vars):
    return pd.get_dummies(df, columns = cat_vars, drop_first = True)

def scale_data(train, validate, test, columns_to_scale, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    '''
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

def split_data(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    return train, validate, test

def get_dummies(df, cols):
    dummies = pd.get_dummies(df[cols], dummy_na =False, drop_first=True)
    df = pd.concat([df, dummies], axis =1).drop(columns = cols)
                   
    return df

## dictionary to be used in imputing_missing_values function
columns_strategy = {
'mean' : [
       'home_sqft',
    'structure_tax_value',
        'assessed_value',
        'land_tax_value',
        'tax_amount'
    ],
    'most_frequent' : [
         'year_built'
     ],
     'median' : [
         'lot_sqft',
         'building_quality'
     ]
 }

def impute_missing_values(df, columns_strategy):
    train, validate, test = split_data(df)
    
    for strategy, columns in columns_strategy.items():
        imputer = SimpleImputer(strategy = strategy)
        imputer.fit(train[columns])

        train[columns] = imputer.transform(train[columns])
        validate[columns] = imputer.transform(validate[columns])
        test[columns] = imputer.transform(test[columns])
    
    return train, validate, test



def prepare_zillow(df):
    '''Prepare zillow for data exploration.'''
    df = get_single_units(df)
    columns_to_drop = ['parcelid', 'id', 'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 
                   'heatingorsystemtypeid', 'propertycountylandusecode', 'propertylandusetypeid', 
                   'propertyzoningdesc', 'rawcensustractandblock', 'unitcnt', 'assessmentyear', 'transactiondate']
    df = df.drop(columns = columns_to_drop)
    df = handle_missing_values(df)
    df.heatingorsystemdesc.fillna('None', inplace=True)
    df = df.dropna(subset=['regionidcity', 'regionidzip', 'censustractandblock'])
    # remove outliers
    cols = ['taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 
        'structuretaxvaluedollarcnt', 'taxamount']
    df = remove_outliers(df, 1.5, cols)
    # Creating a new county column using fips values 
    counties = {6037:'Los Angeles', 6059:'Orange', 6111:'Ventura'}
    df['county'] = df.fips.map(counties)
   # rename columns for clarity and readability
    df = df.rename(columns={'bedroomcnt': 'bedrooms', 'bathroomcnt':'bathrooms', 'roomcnt':'rooms',
                            'heatingorsystemdesc':'heating_system', 'propertylandusedesc':'land_use', 'yearbuilt':'year_built',
                            'calculatedfinishedsquarefeet':'home_sqft',
                             'taxvaluedollarcnt':'assessed_value','landtaxvaluedollarcnt':'land_tax_value',
                            'structuretaxvaluedollarcnt':'structure_tax_value',
                            'taxamount':'tax_amount', 'buildingqualitytypeid':'building_quality', 
                            'lotsizesquarefeet': 'lot_sqft'})
    #Converting certain
    df.fips = df.fips.astype(object)
    df.regionidcity = df.regionidcity.astype(object)
    df.regionidcounty = df.regionidcounty.astype(object)
    df.regionidzip = df.regionidzip.astype(object)
    df.censustractandblock = df.censustractandblock.astype(object)
 
    train, validate, test = impute_missing_values(df, columns_strategy)
    
    return train, validate, test
    
def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire())
    
    return train, validate, test
    
