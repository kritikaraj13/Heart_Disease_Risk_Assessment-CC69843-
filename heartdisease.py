import pandas as pd
from sklearn.linear_model import LogisticRegression

# Read CSV file
df = pd.read_csv('CVD_cleaned.csv')

# Explore the data
print(df.head())  # Display the first few rows(5)
print(df.tail())
print(df.info())  # Summary information

print(df.shape)

print(df.describe())

#To check the missing values
print(df.isna().sum())
print(df.duplicated().sum())#To check the duplicated values
#Now hear we get the 80 duplicated values , lets remove this
duplicates = df.duplicated()
print(df[duplicates])
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())
#Vizulizating the Data
import matplotlib.pyplot as plt
import seaborn as sns

#sns.boxplot(data=df)
#plt.show()

#df.hist(figsize=(12, 8))
#plt.show()

#'Distribution of GENDER'
#sns.countplot(x='Sex', data=df)
#plt.title('Distribution of GENDER')
#plt.show()


#Lets do Model Building now
df.head()
x = df.iloc[:,:1]
print(x.shape)
print(x.columns)
# Independent variables (features)
X = df.drop(columns=['General_Health'])

# Dependent variable (target)
y = df['General_Health']
print(y.shape)

x = df[['Checkup', 'Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Smoking_History', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']]
print(x.shape)

import sklearn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#to convert the x into numaric value because not all values is in numaric value for the data modeling we have to convert

print(df['General_Health'].unique())
# 'General_Health'
health_mapping = {'Excellent':4,'Very Good': 3,'Good': 2, 'Fair': 1, 'Poor': 0}
df['General_Health'] = df['General_Health'].map(health_mapping)
#Now lets read
print(df.head())

# Assuming 'x' is your DataFrame containing the features
binary_columns = ['Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 'Smoking_History']

for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

print(df.head())
#  SEX
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

checkup_mapping = {
    'Within the past year': 1,
    'Within the past 2 years': 2,
    '5 or more years ago': 5,
    'Within the past 5 years' :6
    # Add more mappings as needed
}

df['Checkup'] = df['Checkup'].map(checkup_mapping)
from pandas.api.types import is_numeric_dtype

if is_numeric_dtype(df['Checkup']):
    print("Column is numeric.")
else:
    print("Column is not numeric.")

# now lets do data modeling




#print(df['Heart_Disease'].describe())
#print(df['Skin_Cancer'].describe())
#print(df['Other_Cancer'].describe())
#print(df['Depression'].describe())
#print(df['Diabetes'].describe())
#print(df['Sex'].describe())
#print(df['Age_Category'].describe())
#print(df['Smoking_History'].describe())
#print(df['Alcohol_Consumption'].describe())


import pandas as pd

import pandas as pd

# Assuming 'df' is your DataFrame
# First, extract the lower bound of the age range
df['Age_Category'] = df['Age_Category'].apply(lambda x: x.split('-')[0])

# Next, replace the '+' sign with an empty string
df['Age_Category'] = df['Age_Category'].str.replace('+', '')

# Finally, convert the modified column to float
df['Age_Category'] = pd.to_numeric(df['Age_Category'], errors='coerce')

# Now the 'Age_Category' column should contain numeric values
print(df['Age_Category'].describe())

import pandas as pd

# Assuming 'df' is your DataFrame
# Replace non-numeric values with NaN
df['Age_Category'] = pd.to_numeric(df['Age_Category'], errors='coerce')

# Drop rows with NaN values (if needed)
df.dropna(subset=['Age_Category'], inplace=True)

# Now the 'Age_Category' column should contain numeric values
print(df['Age_Category'].describe())

# Continue with your data modeling
# ...
y = df['General_Health']
print(y.shape)

x = df[['Checkup', 'Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Smoking_History', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']]
print(x.shape)

# Assuming you have a DataFrame 'df' with an 'Age_Category' column
ages = df['Age_Category']  # Include all your data points here

# Calculate the average age
average_age = sum(ages) / len(ages)

print(f"The average age is approximately: {average_age:.2f}")
import pandas as pd

# Assuming 'df' is your DataFrame
# Replace 'Within the past year' with the average age (e.g., 50)
average_age = 52.56
df['Age_Category'] = df['Age_Category'].replace('Within the past year', average_age)

# Alternatively, drop rows with non-numeric values
df.dropna(subset=['Age_Category'], inplace=True)

# Now the 'Age_Category' column should contain numeric values
print(df['Age_Category'].describe())
import pandas as pd

# Assuming 'df' is your DataFrame
# Convert non-numeric values to NaN
df['Age_Category'] = pd.to_numeric(df['Age_Category'], errors='coerce')

# Now the 'Age_Category' column should contain numeric values
print(df['Age_Category'].describe())

# Continue with your data modeling
# ...
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
lr_model = LogisticRegression()

# Train the model on the training data
lr_model.fit(x_train, y_train)



# Continue with your data modeling
# ...













