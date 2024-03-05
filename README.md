<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
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
 ## >IMPORT LIBRARIES:
 ~~~#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
~~~

## >READ THE DATA:
~~~df = pd.read_csv('Churn_Modelling.csv')
print(df)
~~~

## >CHECK DATA:
~~~df.head()
df.tail()
df.columns
~~~

## >CHECK THE MISSING DATA:
~~~print(df.isnull().sum())
~~~

## >CHECK FOR DUPLICATES:
~~~df.duplicated()
~~~

## >ASSIGNING X:
~~~X = df.iloc[:, :-1].values
print(X)
~~~

## >ASSIGNING Y:
~~~y = df.iloc[:,-1].values
print(y)
~~~

## >HANDLING MISSING VALUES:
~~~df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
~~~

## >CHECK FOR OUTLIERS:
~~~df.describe()
~~~

## >DROPPING STRING VALUES DATA FROM DATASET: & CHECKING DATASETS
##  AFTER DROPPING STRING VALUES DATA FROM DATASET:
~~~df1 = df.drop(['Surname','Geography','Gender'],axis=1)
df1.head()
~~~

## >NORMALIE THE DATASET USING (MinMax Scaler):
~~~scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df1))
print(df2)
~~~

## >SPLIT THE DATASET:
~~~X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:,-1].values
print(y)
~~~

## >TRAINING AND TESTING MODEL:
~~~X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train)
print("Length of X_train:",len(X_train))
print(X_test)
print("Length of X_test:",len(X_test))
~~~

## >OUTPUT:
## >DATA CHECKING:
![Screenshot 2024-03-05 193652](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/742baa47-bb4b-4b43-be97-74f99701c5bb)

## >MISSING DATA:
![Screenshot 2024-03-05 193756](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/2675212d-7eba-4a61-b845-03c18f072699)

## >DUPLICATES IDENTIFICATION:
![Screenshot 2024-03-05 193854](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/20ac822f-f9cb-4ac3-a6e2-bad33d4df9e6)

## >VALUE OF Y:
![Screenshot 2024-03-05 193951](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/666d43a7-57ce-4d25-b2e6-f8b2b74b9808)

## >OUTLIERS:
![Screenshot 2024-03-05 194038](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/87a795d5-5817-411a-9de7-d5a7473dc4a0)

## >CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:
![Screenshot 2024-03-05 194121](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/8b0fc241-2dc9-4694-8dc0-080653912e5c)

## >NORMALIZE THE DATASET:
![Screenshot 2024-03-05 194211](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/45324a01-c04b-481f-92e8-9e31f0e939a2)

## >SPLIT THE DATASET:
![Screenshot 2024-03-05 194251](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/ea12848f-ff84-400d-ab27-d2ba630f4d7f)

## >TRAINING AND TESTING MODEL:
![Screenshot 2024-03-05 194344](https://github.com/Lavanyajoyce/Ex-1-NN/assets/119393929/a2004fb5-0243-4072-8364-6676333b49ca)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


