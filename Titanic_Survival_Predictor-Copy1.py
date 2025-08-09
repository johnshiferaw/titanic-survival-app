#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[36]:


import pandas as pd

# Load the dataset (make sure titanic.csv is in the same folder!)
df = pd.read_csv("train.csv")  # If error, replace with full file path


# Show first 5 rows to check it worked
df.head()


# In[ ]:





# In[37]:


df['Age'] = df['Age'].fillna(df['Age'].median())


# In[38]:


df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(df)


# In[39]:


print(df['Age'].isnull().sum()) #check for missin ena 0 kehone yelem malet new


# In[40]:


import os
print(os.getcwd())


# In[41]:


df.to_csv("train_updated.csv", index=False)


# In[ ]:





# In[42]:


# Convert gender to numbers (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})


# In[43]:


# Fill missing "Embarked" values with 'S' (most common port)

df['Embarked'] = df['Embarked'].fillna('S')


# In[ ]:





# In[44]:


# Check for remaining missing values
print("Missing values left:\n", df.isnull().sum())


# In[45]:


df.to_csv("train_updated.csv", index=False)


# In[46]:


print(df['Sex'].unique())
print(df['Sex'].dtype)


# In[52]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt






sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender (0=Male, 1=Female)")
plt.show()


# In[51]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt





# Age distribution of survivors vs non-survivors
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=20, multiple='dodge')

plt.title("Who Survived? Age Matters!")
plt.show()


# In[53]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



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


# In[ ]:





# In[6]:





# In[ ]:





# In[ ]:





# In[54]:


import pandas as pd

# Add column names same as your training data
your_data = pd.DataFrame([[3, 0, 25, 0, 0, 7.25, 0]], columns=[
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
])

# Predict
prediction = model.predict(your_data)
print("Survived!" if prediction[0] == 1 else "Did not survive :(")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




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
embarked = 0  






your_data = pd.DataFrame([[pclass, sex_num, age, sibsp, parch, fare, embarked]],
                         columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])


# Predict
if st.button("Predict Survival"):
    prediction = model.predict(your_data)
    if prediction[0] == 1:
        st.success("🎉 Survived!")
        
    else:
        st.error("💀 Did not survive.")


