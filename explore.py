
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statistics

#Statistical analysis 
def pearsonr(variable, target, alpha =.05):
    corr, p = stats.pearsonr(variable, target)
    print(f'The correlation value between the two variables is {corr:.4} and the P-Value is {p}.')
    print('----------------------------------------------------------------------------')
    if p < alpha:
        print('Since the P value is less than the alpha, we reject the null hypothesis.')
    else:
        print('Since the P value is greater than the alpha, we fail to reject the null hypothesis.')

def t_test_ind(variable, target, alpha =.05):
    t, p = stats.ttest_ind(variable, target)
    print(f'The t value between the two variables is {t:.4} and the P-Value is {p}.')
    print('----------------------------------------------------------------------------')
    if p < alpha:
        print('Since the P value is less than the alpha, we reject the null hypothesis.')
    else:
        print('Since the P value is greater than the alpha, we fail to reject the null hypothesis.')

def chi2(variable, target, alpha=.05):
    observed = pd.crosstab(variable, target)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'The chi2 value between the two variables is {chi2} and the P-Value is {p}.')
    print('----------------------------------------------------------------------------')
    if p < alpha:
        print('Since the P value is less than the alpha, we reject the null hypothesis.')
    else:
        print('Since the P value is greater than the alpha, we fail to reject the null hypothesis.')


#Function to visualize correlations
def plot_correlations(df):
    plt.figure(figsize= (15, 8))
    df.corr()['logerror'].sort_values(ascending=False).plot(kind='bar', color = 'darkcyan')
    plt.title('Correlations with Logerror', fontsize = 18)
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.show()

# Univariate exploration

def plot_distributions(df):
    for col in df.columns:
        sns.histplot(x = col, data=df)
        plt.title(col)
        plt.show()

def plot_distribution(df, var):
    sns.histplot(x = var, data=df)
    plt.title(f'Distribution of {var}', fontsize=15)
    plt.show()


# catergorical vs continuous

def plot_variable_pairs(df):
    sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}})

def plot_categorical_and_continuous_vars(df, cat_var, cont_var):
    sns.barplot(data=df, y=cont_var, x=cat_var)
    plt.show()
    sns.boxplot(data=df, y=cont_var, x=cat_var)
    plt.show()
    sns.stripplot(data=df, y=cont_var, x=cat_var)
    plt.show()

## Bivariate Quant

def plot_swarm(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

## Bivariate Categorical

def run_chi2(train, cat_var, target):
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected

def plot_cat_by_target(train, target, cat_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(cat_var, target, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p

### Multivariate

def plot_all_continuous_vars(train, target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing the target variable. 
    '''
    my_vars = [item for sublist in [quant_vars, [target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()

def plot_violin_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.violinplot(x=cat, y=quant, data=train, split=True, 
                           ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()

def plot_swarm_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.swarmplot(x=cat, y=quant, data=train, ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()

def plot_boxen(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_bar(train, cat_var, quant_var):
    average = train[quant_var].mean()
    p = sns.barplot(data=train, x=cat_var, y=quant_var, palette='Set1')
    p = plt.title(f'Relationship between {cat_var} and {quant_var}.', fontsize=15)
    p = plt.axhline(average, ls='--', color='black')
    return p
    
