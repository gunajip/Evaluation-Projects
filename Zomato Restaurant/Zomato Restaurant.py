#!/usr/bin/env python
# coding: utf-8

# # Project Name:- Zomato Restaurant 

# # Project Description
# Zomato Data Analysis is one of the most useful analysis for foodies who want to taste the best cuisines of every part of the world which lies in their budget. This analysis is also for those who want to find the value for money restaurants in various parts of the country for the cuisines. Additionally, this analysis caters the needs of people who are striving to get the best cuisine of the country and which locality of that country serves that cuisines with maximum number of restaurants.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv("E:\Zomato Restaurant\zomato.csv",encoding='latin1')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


print('Number of Rows :',df.shape[0])
print('Number of Columns :',df.shape[1])


# In[7]:


# This will check if any duplicate Value
df.duplicated().sum() 


# In[8]:


#check if any whitespace, 'NA' or '-' exist in dataset.
df.isin([' ','NA','-']).sum().any()


# # Checking Nulls

# In[9]:


sns.heatmap(df.isna().sum().to_frame(),annot=True,cmap='viridis')
print(df.isnull().sum())


# In[10]:


# Fill nulls with Cuisines column Mode.
df['Cuisines'] = df['Cuisines'].fillna(df['Cuisines'].mode()[0])


# # Exploratory Data Analysis

# # Univaariate Analysis

# In[11]:


plt.figure(figsize=(12,8))
sns.histplot(df['City'],kde=True,color='y')
plt.xticks( fontsize=7,rotation=90)
plt.show()


# Comment:-
# NewDelhi, Noida, Gurgoan and Faridabad have demand of zomato 

# In[12]:


sns.histplot(df['Currency'],kde=True,color='y')
plt.xticks( fontsize=7,rotation=90)
plt.show()


# Comment -  Mostly Indian Rupee used in transection for Zomato Orders.

# In[14]:


sns.histplot(df['Price range'],kde=True,color='g')
plt.show()


# In[15]:


sns.histplot(df['Aggregate rating'],kde=True,color='c')
plt.show()


# In[16]:


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df['Rating color'],kde=True,color='b')
plt.subplot(1,2,2)
sns.histplot(df['Rating text'],kde=True,color='r')
plt.show()


# # Bivariate Analysis 

# In[17]:


plt.figure(figsize=(8, 5))
sns.barplot(data=df,x='Aggregate rating', y='Price range',color='orange')
plt.show()


# # Label Encoding 

# In[18]:


pd.set_option('display.max_columns', None)
df.head()


# In[20]:


object_columns = df.select_dtypes(include=['object'])


# In[21]:


object_column_names = object_columns.columns.tolist()
object_column_names


# In[22]:


df['Restaurant ID'].value_counts()


# In[23]:


df['Locality'].value_counts()


# In[24]:


df['Cuisines'].value_counts()


# In[25]:


df['Locality Verbose'].value_counts()


# In[26]:


df['Restaurant Name'].value_counts()


# In[27]:


df['Currency'].value_counts()


# In[28]:


df['Has Table booking'].value_counts()


# In[29]:


df['Has Online delivery'].value_counts()


# In[30]:


df['Is delivering now'].value_counts()


# In[31]:


df['Switch to order menu'].value_counts()


# In[32]:


# column contain unique value needs to drop.Because it is no more useful for model training.
df.drop(columns=['Switch to order menu','Restaurant ID'],axis=1,inplace=True)


# In[33]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# Assuming 'df' is your DataFrame containing the categorical columns
categorical_columns = ['Restaurant Name', 'City', 'Address', 'Locality', 'Locality Verbose', 
                       'Cuisines', 'Currency', 'Has Table booking', 'Has Online delivery', 
                       'Is delivering now', 'Rating color', 'Rating text']

# Apply label encoding to each categorical column
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])


# In[34]:


df.head()


# # Statistical Analysis

# In[35]:


df.describe().T


# # Correlation Between Dataset

# In[36]:


plt.figure(figsize=(22,18))
sns.heatmap(df.corr(),annot=True,linewidths=1,linecolor='black',fmt=' .2f' )


# In[37]:


plt.figure(figsize=(12,8))
df.corr()['Price range'].sort_values(ascending=False).drop(['Price range']).plot(kind='bar',color='brown')
plt.xlabel('Features',fontsize=10)
plt.ylabel('Price range',fontsize=10)
plt.title('Correlation between Price range and features using bar plot',fontsize=20)
plt.show()


# In[38]:


plt.figure(figsize=(12,8))
df.corr()['Average Cost for two'].sort_values(ascending=False).drop(['Average Cost for two']).plot(kind='bar',color='k')
plt.xlabel('Features',fontsize=10)
plt.ylabel('Average Cost for two',fontsize=10)
plt.title('Correlation between Average Cost for two and features using bar plot',fontsize=20)
plt.show()


# # Outlier Detection

# In[39]:


plt.figure(figsize=(12,30))
index=1
for column in df:
    if index <=20:
        ax = plt.subplot(5,4,index)
        sns.boxplot(df[column], palette='rainbow')
        plt.xlabel(column,fontsize=12)
    index+=1
plt.show()


# Comment - [ 'Country Code', 'City','Longitude', 'Latitude','Average Cost for two', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now','Votes'] these column contains Outlier.

# # Outlier Removal

# In[40]:


from scipy.stats import zscore

# Specify the columns to remove outliers
columns_to_remove_outliers = [ 'Country Code', 'City','Longitude', 'Latitude', 'Currency', 'Has Table booking',
    'Has Online delivery', 'Is delivering now','Votes']

# Calculate z-scores for each specified column
z_scores = df[columns_to_remove_outliers].apply(zscore)

# Set a threshold for z-scores 
threshold = 3

# Remove rows with z-scores beyond the threshold in any specified column
df1 = df[(z_scores.abs() < threshold).all(axis=1)]


# # Checking Skewness Of Dataset

# In[41]:


plt.figure(figsize=(16,30))
plotnumber=1
for column in df:
    if plotnumber <=20:
        ax = plt.subplot(5,4,plotnumber)
        sns.distplot(df[column], color='r',hist=False,kde_kws={"shade": True})
        plt.xlabel(column,fontsize=10)
    plotnumber+=1
plt.show()


# In[42]:


df.skew()


# [ 'Country Code', 'Cuisines','Average Cost for two', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Rating color', 'Rating text', 'Votes'] these ccolumn contain right skewness.
# ['City', 'Longitude', 'Latitude','Aggregate rating'] these column have left skewness.

# # Skewness Removing

# In[43]:


# columns with right-skewed data containing zeros
right_skewed_columns =  [ 'Country Code', 'Cuisines','Average Cost for two', 'Currency', 'Has Table booking',
                         'Has Online delivery', 'Is delivering now', 'Rating color', 'Rating text', 'Votes']

from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method='yeo-johnson')

# Transfroming skew data
df[right_skewed_columns] = scaler.fit_transform(df[right_skewed_columns].values)


# In[45]:


from scipy.stats import boxcox

# Columns with left-skewed data containing zeros
left_skewed_columns = ['City', 'Longitude', 'Latitude', 'Aggregate rating']

# Add a small constant to ensure all data points are positive
df[left_skewed_columns] += abs(df[left_skewed_columns].min()) + 1

# Apply Box-Cox transformation
df[left_skewed_columns] = df[left_skewed_columns].apply(lambda x: boxcox(x)[0])


# In[46]:


df.skew()


# # Splitting data Into Feature And Target Variable

# In[47]:


feature_columns=['Restaurant Name', 'Country Code', 'City', 'Address', 'Locality',
       'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines',
        'Currency', 'Has Table booking','Has Online delivery', 'Is delivering now', 
       'Aggregate rating', 'Rating color', 'Rating text', 'Votes']
label_columns=['Average Cost for two']
labelcolumn=['Price range']


# In[48]:


X = df[feature_columns]
Y = df[labelcolumn]
y = df[label_columns]


# # Feature Scaling

# In[49]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# # Variance_inflation_factor

# In[50]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
vif['Features'] = X.columns


# In[51]:


vif


# 1 < VIF < 5: Moderate multicollinearity. The variance of the coefficient is moderately inflated.
# VIF > 5: High multicollinearity.

# # Machine Learning Model 'Average Cost For Two'

# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score


# In[54]:


# Spliting the data for training & testing.
for i in range(0,200):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=i, test_size=.3)


# # LogisticRegression

# In[55]:


from sklearn.linear_model import LogisticRegression
maxAccu=0
maxRS=0
for i in range(1,250):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state=i)
    lr=LogisticRegression()
    lr.fit(X_train,Y_train)
    Y_pred=lr.predict(X_test)
    acc=accuracy_score(Y_test,Y_pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print('Best accuracy is', maxAccu*100 ,'on Random_state', maxRS)


# # DecisionTreeRegressor

# In[62]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()


# In[64]:


maxAccu1=0
maxRS1=0
for i in range(0,250):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state=i)
    dtr.fit(X_train,Y_train)
    Y_pred1=dtr.predict(X_test)
    acc1=r2_score(Y_test,Y_pred1)
    if acc1>maxAccu1:
        maxAccu1=acc1
        maxRS11=i
print('Best accuracy is', maxAccu1*100 ,'on Random_state', maxRS1)


# # RandomForestRegressor

# In[56]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)

pred2 = rfr.predict(X_test)

# Evaluate the model using regression metrics
R3 = r2_score(Y_test, pred2)

print("R-squared:", R3*100)


# # LinearRegression

# In[57]:


from sklearn.linear_model import LinearRegression
rf=LinearRegression()
rf.fit(X_train,Y_train)
pred3=rf.predict(X_test)
R4=r2_score(Y_test,pred3)
print("R-squared:", R4*100)


# # ExtraTreesRegressor

# In[58]:


from sklearn.ensemble import ExtraTreesRegressor
etr=ExtraTreesRegressor()
etr.fit(X_train, Y_train)

pred5 = etr.predict(X_test)

# Evaluate the model using regression metrics
R6 = r2_score(Y_test, pred5)

print("R-squared:", R6*100)


# # Cross Validation

# In[65]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import  Ridge
from sklearn.linear_model import  Lasso
from xgboost import XGBRegressor


# In[66]:


rf = RandomForestRegressor()
dtc = DecisionTreeRegressor()
XT = ExtraTreesRegressor()
adb=AdaBoostRegressor()
gradb=GradientBoostingRegressor()
xgb=XGBRegressor()
model = [rf,XT,dtc,adb,gradb,xgb]

for m in model:
    m.fit(X_train,Y_train)
    m.score(X_train,Y_train)
    Y_pred = m.predict(X_test)
    print('\n')                                        
    print('\033[1m'+' Error of ', m, ':' +'\033[0m')
    print('Mean absolute error :', mean_absolute_error(Y_test,Y_pred))
    print('Mean squared error :', mean_squared_error(Y_test,Y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test,Y_pred)))
    print('\n')

    print('\033[1m'+' R2 Score :'+'\033[0m')
    print(r2_score(Y_test,Y_pred)) 
    print('==============================================================================================================')


# # Hyper Parameter Tuning

# In[67]:


from sklearn.model_selection import GridSearchCV


# In[68]:


parameter =  {
    'n_estimators': [100], 
    'max_depth': [ 10],      
    'min_samples_split': [2,3],  
    'min_samples_leaf': [1,2]    
}


# In[69]:


GCV = GridSearchCV(RandomForestRegressor(),parameter,verbose =5)


# In[70]:


GCV.fit(X_train,Y_train)


# In[71]:


GCV.best_params_


# In[72]:


Final_mod =  RandomForestRegressor(max_depth= 10,min_samples_leaf= 2,min_samples_split= 3,n_estimators=100 )
Final_mod.fit(X_train,Y_train)
pred=Final_mod.predict(X_test)
print('\n')                                        
print('\033[1m'+' Error in Final Model :' +'\033[0m')
print('Mean absolute error :', mean_absolute_error(Y_test,pred))
print('Mean squared error :', mean_squared_error(Y_test,pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test,pred)))
print('\n')
print('\033[1m'+' R2 Score of Final Model :'+'\033[0m')
print(r2_score(Y_test,pred)) 
print('\n')


# # Saving Model For 'Average Cost For Two'

# In[73]:


import joblib
joblib.dump(Final_mod,'Average_Cost_for_two.pkl')


# In[74]:


['Average_Cost_for_two.pkl']


# In[75]:


# Loading the saved model
Model = joblib.load('Average_Cost_for_two.pkl')


# In[76]:


# prediction  DataFrame
actual = np.array(Y_test).flatten()
predicted = np.array(Model.predict(X_test)).flatten()
df_Predicted = pd.DataFrame({"Actual Values": actual, "Predicted Values": predicted}, index=range(len(actual)))
df_Predicted


# # Machine Learning Model 'Price Range'

# In[77]:


# Spliting the data for training & testing.
for i in range(0,200):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i, test_size=.3)


# In[78]:


rf = RandomForestRegressor()
dtc = DecisionTreeRegressor()
etr = ExtraTreesRegressor()
adb=AdaBoostRegressor()
gbr=GradientBoostingRegressor()
xgb=XGBRegressor()
model = [rf,etr,dtc,adb,gbr,xgb]

for m in model:
    m.fit(X_train,y_train)
    m.score(X_train,y_train)
    max_pred = m.predict(X_test)
    print('\n')                                        
    print('\033[1m'+' Error of ', m, ':' +'\033[0m')
    print('Mean absolute error :', mean_absolute_error(y_test,max_pred))
    print('Mean squared error :', mean_squared_error(y_test,max_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test,max_pred)))
    print('\n')

    print('\033[1m'+' R2 Score :'+'\033[0m')
    print(r2_score(y_test,max_pred)) 
    print('==============================================================================================================')


# # Hyperparameter Tuning

# In[79]:


from sklearn.model_selection import GridSearchCV


# In[81]:


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [ 4, 5],
    'learning_rate': [ 0.1, 0.2],
    'subsample': [ 0.9, 1.0],
    'colsample_bytree': [ 0.9, 1.0]
}


# In[82]:


gsc = GridSearchCV(XGBRegressor(),param_grid,verbose =5)


# In[83]:


gsc.fit(X_train,y_train)


# In[84]:


gsc.best_params_


# In[85]:


Final_mod=  XGBRegressor(colsample_bytree= 0.9,learning_rate = 0.1,max_depth = 5, n_estimators = 200,subsample= 0.9)

Final_mod.fit(X_train,y_train)
max_pred=Final_mod.predict(X_test)
print('\n')                                        
print('\033[1m'+' Error in Final Model :' +'\033[0m')
print('Mean absolute error :', mean_absolute_error(y_test,max_pred))
print('Mean squared error :', mean_squared_error(y_test,max_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test,max_pred)))
print('\n')
print('\033[1m'+' R2 Score of Final Model :'+'\033[0m')
print(r2_score(y_test,max_pred)) 
print('\n')


# # Saving Model Of "Price Range"

# In[86]:


import joblib
joblib.dump(Final_mod,'Price_Range.pkl')


# # Prediction Based On Model

# In[87]:


# Loading the saved model
Model = joblib.load('Price_Range.pkl')

# prediction  DataFrame
actual = np.array(y_test).flatten()
predicted = np.array(Model.predict(X_test)).flatten()
df_Predicted = pd.DataFrame({"Actual Values": actual, "Predicted Values": predicted}, index=range(len(actual)))
df_Predicted


# In[ ]:




