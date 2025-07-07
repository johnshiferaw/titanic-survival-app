#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

# Load the dataset (make sure titanic.csv is in the same folder!)
df = pd.read_csv("train.csv")  # If error, replace with full file path

# Show first 5 rows to check it worked
df.head()


# In[ ]:





# In[ ]:





# In[9]:


df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[11]:


df['Age'] = df['Age'].fillna(df['Age'].median())


# In[12]:


df['Age'] = df['Age'].fillna(df['Age'].median())


# In[13]:


# Convert gender to numbers (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})


# In[15]:


# Fill missing "Embarked" values with 'S' (most common port)

df['Embarked'] = df['Embarked'].fillna('S')


# In[16]:


# Check for remaining missing values
print("Missing values left:\n", df.isnull().sum())


# In[ ]:





# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('train.csv')



sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender (0=Male, 1=Female)")
plt.show()


# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('train.csv')



# Age distribution of survivors vs non-survivors
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=20)
plt.title("Who Survived? Age Matters!")
plt.show()


# In[4]:


import pandas as pd


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("train.csv")

# Convert "Embarked" to numbers (S=0, C=1, Q=2)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Split data into features (X) and target (y)
X = df.drop('Survived', axis=1)  # Input features
y = df['Survived']               # What we predict


# In[6]:


# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("train.csv")

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert "Embarked" to numbers (S=0, C=1, Q=2)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Split data into features (X) and target (y)
X = df.drop('Survived', axis=1)  # Input features
y = df['Survived']               # What we predict

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Check accuracy
print("Model Accuracy:", model.score(X_test, y_test))


# In[11]:


import pandas as pd

# Add column names same as your training data
your_data = pd.DataFrame([[3, 0, 25, 0, 0, 7.25, 0]], columns=[
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
])

# Predict
prediction = model.predict(your_data)
print("Survived!" if prediction[0] == 1 else "Did not survive :(")


# In[12]:


import pandas as pd

# Add column names same as your training data
your_data = pd.DataFrame([[3, 1, 99, 0, 0, 7.25, 0]], columns=[
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
])

# Predict
prediction = model.predict(your_data)
print("Survived!" if prediction[0] == 1 else "Did not survive :(")


# In[13]:


import pandas as pd

# Add column names same as your training data
your_data = pd.DataFrame([[3, 0, 1000, 0, 0, 7.25, 0]], columns=[
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
])

# Predict
prediction = model.predict(your_data)
print("Survived!" if prediction[0] == 1 else "Did not survive :(")


# In[14]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


st.title("Titanic Survival Predictor")




# 📥 User Inputs
age = st.slider("Age", 0, 100, 25)
sex = st.selectbox("Sex", ["Male", "Female"])

# Convert sex to numeric
sex_num = 0 if sex == "Male" else 1

# Use default values for other features
pclass = 3
sibsp = 0
parch = 0
fare = 7.25
embarked = 0  # S


# Add your model prediction code here




df = pd.read_csv("train.csv")

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert "Embarked" to numbers (S=0, C=1, Q=2)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Split data into features (X) and target (y)
X = df.drop('Survived', axis=1)  # Input features
y = df['Survived']               # What we predict

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)






your_data = pd.DataFrame([[pclass, sex_num, age, sibsp, parch, fare, embarked]],
                         columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])


# Predict
if st.button("Predict Survival"):
    prediction = model.predict(your_data)
    if prediction[0] == 1:
        st.success("🎉 Survived!")
    else:
        st.error("💀 Did not survive.")




# In[ ]:




