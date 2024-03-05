<H3>ENTER YOUR NAME</H3>HAMSINI K
<H3>ENTER YOUR REGISTER NO.</H3>212222040049
<H3>EX. NO.1</H3>25/02/2024
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
## >READ THE DATA:

~~~
df = pd.read_csv('Churn_Modelling.csv')
print(df)
~~~

## >CHECK DATA:

~~~
df.head()
df.tail()
df.columns
~~~

## >CHECK THE MISSING DATA:

~~~
print(df.isnull().sum())
~~~

## >CHECK FOR DUPLICATES:

~~~
df.duplicated()
~~~

## >ASSIGNING X:
~~~
X = df.iloc[:, :-1].values
print(X)
~~~

## >ASSIGNING Y:

~~~
y = df.iloc[:,-1].values
print(y)
~~~

## >HANDLING MISSING VALUES:

~~~
df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
~~~

## >CHECK FOR OUTLIERS:

~~~
df.describe()
~~~

## >DROPPING STRING VALUES DATA FROM DATASET: & CHECKING DATASETS
##  AFTER DROPPING STRING VALUES DATA FROM DATASET:

~~~
df1 = df.drop(['Surname','Geography','Gender'],axis=1)
df1.head()
~~~

## >NORMALIE THE DATASET USING (MinMax Scaler):

~~~
scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df1))
print(df2)
~~~

## >SPLIT THE DATASET:

~~~
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:,-1].values
print(y)
~~~

## >TRAINING AND TESTING MODEL:

~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train)
print("Length of X_train:",len(X_train))
print(X_test)
print("Length of X_test:",len(X_test))
~~~

## OUTPUT:
## DATA CHECKING:

![Screenshot 2024-02-25 160903](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/85f3956e-cae4-40f0-811d-a9cef10ee5e9)

## MISSING DATA:

![Screenshot 2024-02-25 161035](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/6206f73b-a6e5-49e3-94c9-084308b1ef8d)

## DUPLICATES IDENTIFICATION:

![Screenshot 2024-02-25 161140](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/fc83084e-b510-4da4-9a47-d40dabe1f2ed)

## VALUE OF Y:

![Screenshot 2024-02-25 161250](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/b5258909-fc22-46e6-b0e2-952ca471630b)

## OUTLIERS:

![Screenshot 2024-02-25 161418](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/a20f0127-47ee-419d-b076-29705d4618ea)

## CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:

![Screenshot 2024-02-25 161521](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/3def6415-c0c7-44b0-b326-c57b8ce4259f)

## NORMALIZE THE DATASET:

![Screenshot 2024-02-25 161654](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/4e8f54b9-c0c8-49f0-9b89-030651ca033b)

## SPLIT THE DATASET:

![Screenshot 2024-02-25 161801](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/8ec16ad6-597c-4a47-9b7a-9f4f486e302e)

## TRAINING AND TESTING MODEL:

![Screenshot 2024-02-25 161902](https://github.com/shanmugapriyatharani/Ex-1-NN/assets/119393427/74ebc1da-bacd-4cfa-991c-3f9071225695)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.




