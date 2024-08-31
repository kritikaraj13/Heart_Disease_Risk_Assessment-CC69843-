import pandas as pd

# Read CSV file
df = pd.read_csv('CVD_cleaned.csv')

# Explore the data
print(df.head())  # Display the first few rows
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

import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(df)
plt.show()
df.hist(figsize=(12, 8))
plt.show()

print(df.columns.tolist())#list of the column
sns.countplot(x='Sex', data=df)
plt.title('Distribution of GENDER')
plt.show()
#Vizulization on Age_Category
import matplotlib.pyplot as plt
plt.hist(df['Age_Category'], bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()
#Vizulization between Green_Vegetables_Consumption vs. FriedPotato_Consumption
plt.scatter(df['Green_Vegetables_Consumption'], df['FriedPotato_Consumption'])
plt.xlabel('Green_Vegetables_Consumption')
plt.ylabel('FriedPotato_Consumption')
plt.title('Green_Vegetables_Consumption vs. FriedPotato_Consumption')
plt.show()
#To visualization the smoking history
smoking_data = {'Smoking_History': ['Yes', 'No'],
                'Count': [120, 80]}

# Create a bar chart
sns.barplot(x='Smoking_History', y='Count', data=smoking_data)
plt.title('Smoking History')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.DataFrame({
    'Age_Category': [25, 30, 35, 40, 45, 50],
    'BMI': [22.5, 24.0, 26.5, 28.2, 29.8, 31.0],
    'Weight_(kg)': [60, 65, 70, 75, 80, 85],
    'Height_(cm)': [165, 170, 175, 180, 185, 190]
})

# Create a histogram plot
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', bins=10, kde=True, color='skyblue', label='Age')
sns.histplot(data=data, x='BMI', bins=10, kde=True, color='salmon', label='BMI')
sns.histplot(data=data, x='Weight', bins=10, kde=True, color='green', label='Weight')
sns.histplot(data=data, x='Height', bins=10, kde=True, color='purple', label='Height')

plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of Age, BMI, Weight, and Height')
plt.legend()
plt.show()



