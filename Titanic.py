import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data Cleaning
# Fill missing values in 'Age' with the median age
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common port
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to a large number of missing values
train_data.drop('Cabin', axis=1, inplace=True)

# Convert 'Sex' and 'Embarked' to numerical values
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# EDA
# Summary Statistics
print(train_data.describe())

# Visualize Data Distributions
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_data['Fare'], kde=True, bins=30)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Visualize Categorical Data
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=train_data)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Sex', data=train_data)
plt.title('Survival Count by Sex')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(['Male', 'Female'])
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Pclass', data=train_data)
plt.title('Survival Count by Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Advanced EDA
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=train_data)
plt.title('Age vs. Survival')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=train_data)
plt.title('Fare vs. Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Pclass vs. Survival')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.show()
