# EX NO : 01 Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm

STEP 1: Read the given Data
STEP 2: Get the information about the data
STEP 3: Remove the null values from the data
STEP 4: Save the Clean data to the file
STEP 5: Remove outliers using IQR
STEP 6: Use zscore of to remove outliers

# Coding and Output

## 1. Data Cleaning Process

```
import numpy as np
import pandas as pd
dt=pd.read_csv("/content/SAMPLEIDS.csv")
dt
```
<img width="740" height="633" alt="op1 1" src="https://github.com/user-attachments/assets/066353ce-4fe9-4003-b144-3fec20535930" />

```
dt.head()
```
<img width="705" height="186" alt="op1 2" src="https://github.com/user-attachments/assets/dba4235d-d4ec-4f38-bb04-a6a6a3659042" />

```
dt.tail()
```
<img width="756" height="182" alt="op1 3" src="https://github.com/user-attachments/assets/c92b01cf-d191-47ee-b778-b797b0e98330" />

```
dt.isnull()
```
<img width="617" height="591" alt="op1 4" src="https://github.com/user-attachments/assets/be8a46ae-68c9-451c-8442-8fb5d5cd9190" />

```
dt.notnull()
```
<img width="613" height="589" alt="op1 5" src="https://github.com/user-attachments/assets/0fde7e44-3883-45ee-9586-0004ed7893b1" />

```
dt.isnull().sum()
```
<img width="260" height="387" alt="op1 6" src="https://github.com/user-attachments/assets/24d58199-b1ae-4d6a-a9b4-462ce855fb97" />

```
dt.isnull().any()
```
<img width="337" height="393" alt="op1 7" src="https://github.com/user-attachments/assets/560762a0-a813-4407-913a-fcc97bd8f7ec" />

```
dt.dropna(axis=0)
```
<img width="748" height="384" alt="op1 8" src="https://github.com/user-attachments/assets/191bea95-8e01-4704-bc06-56dbeb666002" />

```
dt.dropna(axis=1)
```
<img width="400" height="587" alt="op1 9" src="https://github.com/user-attachments/assets/a915b4e7-299f-49a5-9f43-4fdb7fe9383d" />

```
dt.fillna(500)
```
<img width="847" height="591" alt="op1 10" src="https://github.com/user-attachments/assets/042b053d-4b0f-4402-ad7b-8a3d4f91e58b" />

```
dt.ffill()
```
<img width="765" height="593" alt="op1 11" src="https://github.com/user-attachments/assets/c5b0d20c-e2f2-413b-b15f-81b67ef60990" />

```
dt.bfill()
```
<img width="738" height="586" alt="op1 12" src="https://github.com/user-attachments/assets/b5053ee5-3a6c-46d3-9923-2fbf30373066" />

```
dt.fillna({'NAME':'SK', 'M1':'0'})
```
<img width="786" height="590" alt="op1 13" src="https://github.com/user-attachments/assets/721036e1-cea8-4d85-a29e-a8482304ba00" />


## 2. Outlier Detection and Removal

```
iris=pd.read_csv("/content/iris.csv")
iris
```
<img width="538" height="372" alt="op2 1" src="https://github.com/user-attachments/assets/c5ffe010-7bcf-4ef6-9276-bc31b87683f3" />

```
iris.describe()
```
<img width="507" height="264" alt="op2 2" src="https://github.com/user-attachments/assets/15ad656d-0fa9-4980-a99c-5e12ef9106f0" />

```
import seaborn as sea
sea.boxplot(x="sepal_width",data=iris)
```
<img width="519" height="413" alt="op2 3" src="https://github.com/user-attachments/assets/44b639dc-22d5-4bbb-8a22-202fd93736da" />

```
q1=iris.sepal_width.quantile(0.25)
q3=iris.sepal_width.quantile(0.75)
iqr=q3-q1
print(iqr)
```
<img width="293" height="96" alt="op2 4" src="https://github.com/user-attachments/assets/eef28987-da8c-4664-a4bf-8d516b3ce3ab" />

```
rid=iris[((iris.sepal_width<(q1-1.5*iqr))|(iris.sepal_width>(q3+1.5*iqr)))]
rid['sepal_width']
```
<img width="541" height="210" alt="op2 5" src="https://github.com/user-attachments/assets/11d9e29c-5fd5-48ae-8ebf-7efe4996dc5d" />

```
rid=iris[~((iris.sepal_width<(q1-1.5*iqr))|(iris.sepal_width>(q3+1.5*iqr)))]
rid
```
<img width="578" height="371" alt="op2 6" src="https://github.com/user-attachments/assets/0722f6e7-4af0-4f63-b20b-63f0a1883f18" />

```
rid=iris[((iris.sepal_width>(q1-1.5*iqr))&(iris.sepal_width<(q3+1.5*iqr)))]
rid['sepal_width']
```
<img width="568" height="405" alt="op2 7" src="https://github.com/user-attachments/assets/1cf42cf3-29f9-4142-b741-25ed2a689849" />

```
import numpy as np
import scipy.stats as stats
z=np.abs(stats.zscore(iris.sepal_width))
z
```
<img width="571" height="499" alt="op2 8" src="https://github.com/user-attachments/assets/9ccd161f-dc4e-4367-a0d2-ec384a6d87ab" />

```
tv=iris[z>3]  #threshold_value (constant=3)
tv
```
<img width="537" height="111" alt="op2 9" src="https://github.com/user-attachments/assets/f9d6af1a-e0a0-4b2c-b1a1-5c10d3780966" />

# Result
Thus the programs are executed and verified successfully.
