# Achraf Safsafi
# DSC530
# Final project

# __________________________________________________________
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf

# ______________________________________________________

# importing data
data = pd.read_csv('datasets_movies.csv', encoding='latin-1')
##Understanding the structure of the data
mydata = pd.DataFrame(data, columns=['year', 'budget', 'genre', 'rating', 'gross', 'runtime', 'votes', 'score'])
print(mydata.head())
##Understanding the structure of the data
# dataframe shape
print(mydata.shape)
# data types
print(mydata.info())
# factorize objects)
mydata['Rating'] = pd.factorize(mydata.rating)[0]
print(mydata.rating.head())
mydata['Genre'] = pd.factorize(mydata.genre)[0]
print(mydata.genre.head())
print(mydata.head())
print(mydata.info())
df = pd.DataFrame(mydata, columns=['year', 'budget', 'Genre', 'Rating', 'gross', 'runtime', 'votes', 'score'])
print(df.info())
## visualizing data
df.drop(columns="year").hist()
plt.show()
## detect and remove outliers
# detect the outlier using Z-score
z = np.abs(stats.zscore(df))
# filter the outliers and get the clean data
df = df[(z < 3).all(axis=1)]
print(df.shape)
# plot after removing outlier
df.drop(columns="year").hist()
plt.show()
## describe the data
print(df.describe())

## plotting PMF
scoreB2000 = df.score[(df.year <= 2000)]
probabilities = scoreB2000.value_counts(normalize=True)
scoreA2000 = df.score[(df['year'] > 2000)]
probabilities1 = scoreA2000.value_counts(normalize=True)
ax = plt.gca()
year = ["<=2000", ">2000"]
sns.barplot(probabilities.index, probabilities.values, color='blue')
sns.barplot(probabilities1.index, probabilities1.values, color='red')
plt.xlabel('score', fontsize=16)
plt.ylabel('probability', fontsize=16)
plt.legend(year, loc=1)
ax.set_xticklabels(df.score, fontsize=5)
plt.show()
## Correlation
# Scatter plots
s1 = sns.scatterplot(data=df, x='score', y='budget')
plt.show()
s2 = sns.scatterplot(data=df, x='score', y='gross')
plt.show()
s3 = sns.scatterplot(data=df, x='score', y='runtime')
plt.show()
s4 = sns.scatterplot(data=df, x='score', y='votes')
plt.show()
s5 = sns.scatterplot(data=df, x='score', y='Rating')
plt.show()
s6 = sns.scatterplot(data=df, x='score', y='Genre')
plt.show()
#  plot the heatmap
plt.title('Correlation', fontsize=15)
sns.heatmap(df.corr().astype(float).corr(), vmax=1.0, annot=True)
plt.show()

# the Pearson's Correlation test
from scipy.stats import pearsonr

data1 = df.score
data2 = df.budget
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')

## Pearson's Correlation test
data1 = df.score
data2 = df.gross
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')
# the Pearson's Correlation test
data1 = df.score
data2 = df.runtime
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')
# the Pearson's Correlation test
data1 = df.score
data2 = df.votes
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')
# the Pearson's Correlation test
data1 = df.score
data2 = df.Rating
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')


## buildind model
# encoding score
def func(x):
    if 0 < x <= 6.3:
        return 0
    elif 6.3 < x <= 10:
        return 1


df['score'] = df['score'].apply(func)
print(df.score.head())
print(type(df.score))
# logistic regression
f = "score ~ budget  "

r = smf.logit(formula=f, data=df).fit()
print(r.summary())


