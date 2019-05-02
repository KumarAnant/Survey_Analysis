#E:/OneDrive/Hult/Machine Learning/Assignments/Assignment - 3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:44:26 2019

@author: Anant Kumar

Assignment - 3, Machine Learning

"""
## Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA # principal component analysis
from sklearn.cluster import KMeans
import os

# Read the data file
os.chdir('E:/OneDrive/Hult/Machine Learning/Assignments/Final Exam')
df = pd.read_excel('./Data/finalExam_Mobile_App_Survey_Data.xlsx')

#CHeck if there is any null value
df.isnull().any().any()

## Correlation Heatmap
df_corr = df.corr().round(2)
plt.figure(figsize=(30, 18))
sns.heatmap(df_corr, cmap='coolwarm')
plt.show()


# Seperate demography data, named here as analyzeBase
analyzeBaseCols = ['q50r1','q50r2', 'q50r3','q50r4', 'q50r5', 
                        'q49', 'q54', 'q55', 'q56', 'q57']
analyzeBaseLabel = ['Age', 'Marital_Status','No Child', 'Child 6Yrs', 'Child 6-12', 
                'Child 13-17', 'Child 18+', 'Race', 'Ethnicity', 'Income', 'Gender']
analyzeBase = df['q1']
df.drop('q1', axis = 1, inplace = True)
for col in df:
        if col in analyzeBaseCols:                
                analyzeBase =  pd.concat([analyzeBase, df[col]], axis = 1)
                df.drop(col, inplace = True, axis = 1)
analyzeBase.columns = analyzeBaseLabel

## Some Feature Engineering
analyzeBase['Ethnicity'].unique()


# Create a new column for total number of devices 
df['DevCount'] = df['q2r1'] +\
                df['q2r2'] +\
                df['q2r3'] +\
                df['q2r4'] +\
                df['q2r5'] +\
                df['q2r6'] +\
                df['q2r7'] +\
                df['q2r8'] +\
                df['q2r9']
# Do not include df['q2r10'], as it is an entry for no device

# Create a new column for total number of apps
df['AppCount'] = df['q4r1'] +\
                df['q4r2'] +\
                df['q4r3'] +\
                df['q4r4'] +\
                df['q4r5'] +\
                df['q4r6'] +\
                df['q4r7'] +\
                df['q4r8'] +\
                df['q4r9'] +\
                df['q4r10']

# Do not include df['q4r10'], as it is an entry for no app

## Replace the age in Q1 wiht its  value

ageResponse = {1: 'Under 18', 2: '18 - 24', 3: '25 - 29',
                4: '30 - 34', 5: '35 - 39', 6: '40 - 44', 
                7: '45 - 49', 8: '50 - 54', 9: '55 - 59', 
                10: '60 - 64', 11: '65 Above' }

analyzeBase['Age'].replace(ageResponse, inplace = True)

## Change values for marital status
maritalStatus = {1: 'Married', 
                2: 'Single', 
                3: 'Single with Partner',
                4: 'Sep/Wid/Div'}

analyzeBase['Marital_Status'].replace(maritalStatus, inplace = True)

# Change values for response
raceResponse = {1: 'White/Caucacian', 
                2: 'Black/African American', 
                3: 'Asian',
                4: 'Native Hawaiian',
                5: 'American Indian',
                6: 'Other'
                }
analyzeBase['Race'].replace(raceResponse, inplace = True)

# Change values for gender
genderResponse = {1: 'Male', 
                  2: 'Female'
                }
analyzeBase['Gender'].replace(genderResponse, inplace = True)

# Change values for enthnicity
ethnicityResponse = {1: 'Hispanic/Latino', 
                     2: 'Other'
                }
analyzeBase['Ethnicity'].replace(ethnicityResponse, inplace = True)

# Change values for income in demography data
incomeResponse = {1: '<10k', 
                2: '10-15K', 
                3: '15-20K',
                4: '20-30K',
                5: '30-40K',
                6: '40-50K',
                7: '50-60K',
                8: '60-70K', 
                9: '70-80K',
                10: '80-90K',
                11: '90-100K',
                12: '100-125K',
                13: '125-150K',
                14: '> 150K',
                }
analyzeBase['Income'].replace(incomeResponse, inplace = True)


## Replace the Q11 values to actual mean value in that category
## So, for group of 6-10, the value imputed = 8
## I presume in Q11, "Don't know" segment of users,
## are not actively using apps. Imputing value 5 (assuming
## 6 apps installed by the segement)

appsCount = {1: 3, 2: 8, 3: 20, 4: 40, 5: 6}


## Here value 6 is imputed for "Don't know" assuming those who do
## not know how many apps they have, they are not so frequently using
## apps.
df['q11'].replace(appsCount, inplace = True)


## Impute mean value for Q12, i.e. Free download apps
freeAppsCount = {1: 0, 2: 13, 3: 38, 4: 63, 5: 88, 6: 100}
df['q12'].replace(freeAppsCount, inplace = True)

####################################
## Data exploration
###################################
## Age
print("Age distribution: \n", analyzeBase['Age'].value_counts())
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'Age',
              data = analyzeBase)
plt.xlabel("Age")
plt.ylabel('Count')
plt.title("Age distribution")
plt.show()

## Marital Status
print("Marital Status: \n", analyzeBase['Marital_Status'].value_counts())
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'Marital_Status',
              data = analyzeBase)
plt.xlabel("Status")
plt.ylabel('Count')
plt.title("Marital Status")
plt.show()

## Race
print("Race distribution: \n", analyzeBase['Race'].value_counts())
print(analyzeBase['Race'].value_counts())
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'Race',
              data = analyzeBase)
plt.xlabel("Race")
plt.ylabel('Count')
plt.title("Race distribution")
plt.show()

## Ethnicity
print("Ethnicity distribution: \n", analyzeBase['Ethnicity'].value_counts())
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'Ethnicity',
              data = analyzeBase)
plt.xlabel("Ethnicity")
plt.ylabel('Count')
plt.title("Ethnicity distribution")
plt.show()

## Income
print("Income distribution: \n", analyzeBase['Income'].value_counts())
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'Income',
              data = analyzeBase)
plt.xlabel("Income")
plt.ylabel('Count')
plt.title("Income distribution")
plt.show()

## Gender
print("Gender distribution: \n", analyzeBase['Gender'].value_counts())
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'Gender',
              data = analyzeBase)
plt.xlabel("Gender")
plt.ylabel('Count')
plt.title("Gender distribution")
plt.show()

## Number of devices
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'DevCount',
              data = df)
plt.xlabel("Devices count")
plt.ylabel('Person Count')
plt.title("Number of devices held by a person")
plt.show()

## Number of applications installed
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'AppCount',
              data = df)
plt.xlabel("Applications count")
plt.ylabel('Person Count')
plt.title("Number of Applications installed by a person")
plt.show()



## Scaling the dataset
df_feat = df.drop(['caseID'], axis = 1)
df_feat.columns
scaler = StandardScaler()
scaler.fit(df_feat)
X_scaled = pd.DataFrame(scaler.transform(df_feat))

#Getting the columns name back in scaled data
X_scaled.columns = df_feat.columns



## Create PCA model with 3 principal component to visualize
pca = PCA(n_components=3)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

fig = plt.figure(figsize=(30, 20))
ax = Axes3D(fig)
ax.scatter(xs= X_pca[:,0], ys= X_pca[:, 1], 
        zs = X_pca[:, 2], c = 'r', marker = 'o')
plt.xlabel("First component")
plt.ylabel("Second component")
ax.set_zlabel("Third component")
plt.show()


## PCA Model with n_component value unknown
pca = PCA(  n_components=None,
            random_state=508)
pca.fit(X_scaled)

## Scree plot to determine optimum n_component value
pca.n_components_
plt.figure(figsize=(30, 20))
plt.plot(range(pca.n_components_),
        pca.explained_variance_ratio_,
        linewidth = 2,
        marker = 'o',
        markersize = 10,
        markeredgecolor = 'b',
        markerfacecolor = 'g'
        )
plt.xticks(range(pca.n_components_))
plt.show()


## Create PCA model with optimum principal components
## THe optimum value of principal components is 4
pca = PCA(n_components=4,
                random_state = 805
                )
pca.fit(X_scaled)

## Factor loading to understand principal components
factor_loading_df = pd.DataFrame(pd.np.transpose(pca.components_))
factor_loading_df = factor_loading_df.set_index(X_scaled.columns)
factor_loading_df.to_excel('factor_loading.xlsx')
## Analyze factor strength
X_pca = pd.DataFrame(pca.transform(X_scaled))
X_pca.columns = ['Extravagant', 'CouchPotato', 'Techjunkie', 'Executive']
final_pca_df = pd.concat([analyzeBase, X_pca], axis = 1)

# Analyzing age groups


# Extravagant
sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Age',
            y =  'Extravagant',
            data = final_pca_df)
plt.title("Distribution of Extravagant group over various age group")
plt.ylim(-5, 5)
plt.tight_layout()
plt.show()

# CouchPotato
sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Age',
            y =  'CouchPotato',
            data = final_pca_df)
plt.title("Distribution of CouchPotato group over various age group")
plt.ylim(-5, 5)
plt.tight_layout()
plt.show()

# Techjunkie
sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Age',
            y =  'Techjunkie',
            data = final_pca_df)
plt.title("Distribution of Techjunkie class over various age group")
plt.ylim(-5, 5)
plt.tight_layout()
plt.show()

# Executive Class
sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Age',
            y =  'Executive',
            data = final_pca_df)
plt.title("Distribution of Executive group over various age group")
plt.ylim(-5, 5)
plt.tight_layout()
plt.show()

#################################################
######################### Marital Status#########
#################################################

# Analyzing age groups


# Extravagant
sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Marital_Status',
            y =  'Extravagant',
            data = final_pca_df)
plt.title("Distribution of Extravagant group over various marital status group")
plt.tight_layout()
plt.show()

# CouchPotato
sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Marital_Status',
            y =  'CouchPotato',
            data = final_pca_df)
plt.title("Distribution of CouchPotato group over various marital status group")
plt.tight_layout()
plt.show()

# Techjunkie
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Marital_Status',
            y =  'Techjunkie',
            data = final_pca_df)
plt.title("Distribution of Techjunkie class over various age group")
plt.tight_layout()
plt.show()

# Executive Class
sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Marital_Status',
            y =  'Executive',
            data = final_pca_df)
plt.title("Distribution of Excutive group over various age group")
plt.tight_layout()
plt.show()



###############################
######################### Ethnicity
########################################

# Analyzing age groups

# Extravagant
sns.set(font_scale =2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Ethnicity',
            y =  'Extravagant',
            data = final_pca_df)
plt.title("Distribution of Extravagant group over Ethnicity")
plt.tight_layout()
plt.show()

# CouchPotato
sns.set(font_scale =2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Ethnicity',
            y =  'CouchPotato',
            data = final_pca_df)
plt.title("Distribution of CouchPotato group over Ethnicity")
plt.tight_layout()
plt.show()

# Moody Class
sns.set(font_scale =2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Ethnicity',
            y =  'Techjunkie',
            data = final_pca_df)
plt.title("Distribution of Techjunkie class over Ethnicity")
plt.tight_layout()
plt.show()

# Tech_Laggard Class
sns.set(font_scale =2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Ethnicity',
            y =  'Executive',
            data = final_pca_df)
plt.title("Distribution of Executived group over Ethnicity")
plt.tight_layout()
plt.show()


##########################################
########### K Means cluster
#########################################


## Determine optimum value of k
ks = range(1, 50)
inertia = []
for k in ks:
        model = KMeans(n_clusters = k,
                random_state = 508)
        model.fit(X_scaled)
        inertia.append(model.inertia_)
sns.set(font_scale =2)
fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(ks, inertia, '-o')
plt.title("Finding optimum K Value")
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

## Optimum value of k = 5
## Redefining model for K = 5
KM = KMeans(n_clusters = 5,
               random_state = 508)
KM.fit(X_scaled)
KM_clusters = pd.DataFrame({'cluster': KM.labels_})
KM_centre = pd.DataFrame(KM.cluster_centers_)
KM_centre.columns = df_feat.columns
KM_centre.to_excel('KM_centre.xlsx')

## Analyse cluster centroids
centroids = pd.DataFrame(KM.cluster_centers_)
centroids.columns = X_scaled.columns
# centroids.to_excel('centroids.xlsx')
KM_clusters = pd.DataFrame({'cluster': KM.labels_})
print(KM_clusters.iloc[: , 0].value_counts())
# Analyze cluster membership
X_KM = pd.concat([analyzeBase, df, KM_clusters], axis = 1)

######################################
### KM Analysis
######################################

########################
# Channel
########################

# 
X_KM.columns
sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (25, 15))
sns.boxplot(x = 'Age',
            y = 'DevCount',
            hue = 'cluster',
            data = X_KM)

sns.set(font_scale =2)
plt.show()

####################################################
## Combining PCA and cluster
####################################################

scaler = StandardScaler()
scaler.fit(X_pca)
X_pca_clust = pd.DataFrame(scaler.transform(X_pca))
X_pca_clust.columns = X_pca.columns
print(pd.np.var(X_pca_clust))

###### K Mean cluster######
###########################
ks = range(1, 50)
inertia = []
for k in ks:
        model = KMeans(n_clusters = k,
                random_state = 508)
        model.fit(X_pca_clust)
        inertia.append(model.inertia_)
fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(ks, inertia, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.title("Finding optimum K Value")
plt.xticks(ks)
plt.show()

## Runing K Means cluster for optimum K = 5
KM_pca = KMeans(n_clusters = 5,
                         random_state = 508)
KM_pca.fit(X_pca_clust)
kmeans_pca = pd.DataFrame({'cluster': KM_pca.labels_})
print(kmeans_pca.iloc[: , 0].value_counts())

########################
# Step 4: Analyze cluster centers
########################

centroids_pca = pd.DataFrame(KM_pca.cluster_centers_)


# Rename principal components
centroids_pca.columns = X_pca.columns
print(centroids_pca)

# Sending data to Excel
centroids_pca.to_excel('pca_centriods.xlsx')
clst_pca_df = pd.concat([kmeans_pca, X_pca_clust], axis = 1)
final_clst_pca = pd.concat([analyzeBase, clst_pca_df ], axis = 1, sort = True)
allData = pd.concat([analyzeBase, clst_pca_df, df], axis = 1, sort = True)

##########################
##### Analysis ###########
##########################
############
###### Age
############



# Plots for Age group vs PCA groups
for cntr in range((len(X_pca.columns))):
        yVar = centroids_pca.columns[cntr]
        title = 'Distribution of '+ yVar + ' group in various clusters'
        sns.set(font_scale = 2)
        fig, ax = plt.subplots(figsize = (25, 15))
        sns.boxplot(x = 'Age',
                y = centroids_pca.columns[cntr],
                hue = 'cluster',
                data = final_clst_pca)
        plt.xlabel('Age groups')
        plt.ylabel(yVar)
        plt.title(title)
        plt.show()

# Plots for Age group vs PCA groups
for cntr in range((len(X_pca.columns))):
        yVar = centroids_pca.columns[cntr]
        title = 'Distribution of '+ yVar + ' group in various clusters'
        sns.set(font_scale = 2)
        fig, ax = plt.subplots(figsize = (25, 15))
        sns.boxplot(x = 'Income',
                y = centroids_pca.columns[cntr],
                hue = 'cluster',
                data = final_clst_pca)
        plt.xlabel('Income groups')
        plt.ylabel(yVar)
        plt.title(title)
        plt.show()

# Plots for Age group vs PCA groups
for cntr in range((len(X_pca.columns))):
        yVar = centroids_pca.columns[cntr]
        title = 'Distribution of '+ yVar + ' group in various clusters'
        sns.set(font_scale = 2)
        fig, ax = plt.subplots(figsize = (25, 15))
        sns.boxplot(x = 'Income',
                y = centroids_pca.columns[cntr],
                hue = 'cluster',
                data = final_clst_pca)
        plt.xlabel('Income groups')
        plt.ylabel(yVar)
        plt.title(title)
        plt.show()




### Identification of various app markets.


## Music and sound identification app
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r1',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Music and sound identification app installation")
plt.show()

print("Age distribution: \n", allData['q4r1'].value_counts())
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r2',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("TV Check in  app installation")
plt.show()

## Entertainment in app

sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r3',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Entertainment app installation")
plt.show()

## TV Show app

sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r4',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("TV Show app installation")
plt.show()

## Gaming app

sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r5',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Gaming app installation")
plt.show()

## Social networking app

sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r6',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Social networking app installation")
plt.show()


## News app

sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r7',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("News app installation")
plt.show()

## Shopping app

sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r8',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Shopping app installation")
plt.show()

## Publication news app

sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r9',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Publication news app installation")
plt.show()

## Other  app
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q4r10',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Other app installation")
plt.show()

## Total apps
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q11',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Total app installation")
plt.show()

## Total apps
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q12',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("App installation")
plt.ylabel('Count')
plt.title("Free app installation")
plt.show()

## Total apps
sns.set(font_scale = 2)
plt.figure(figsize = (25, 15))
sns.countplot(x = 'q48',
              data = allData,
              hue = 'cluster'
              )
plt.xlabel("Education level")
plt.ylabel('Count')
plt.title("Education Level")
plt.show()




