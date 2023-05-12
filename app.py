import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
import streamlit as st


##give the title 
st.title("Knowledge Discovery using Data Analysis of COVID")
#load data 
df = pd.read_csv('patient_data.csv')
#st.write(df.head)



#feature extraction
covid =df[['Patient Number','Age Bracket','Gender','Num Cases']]
st.write(covid.head())

#dropping the null value
df = covid.dropna(axis=0)

df.replace({'Gender':{'F':0,'M':1,'Non-Binary':np.NAN,'M,':np.NAN, 'Femal e':np.NAN}},inplace=True)

#dropping remaning null values
df = df.dropna(axis=0)

# rounding the value till 
df.round(0)

df.reset_index()

#feature selection
df = df[['Age Bracket', 'Num Cases']]

#setting the num cases 
df['Num Cases'] = df[df['Num Cases'] == 1]['Num Cases']

df = df.dropna(axis=0)
st.write(df.head())



#creating a dict for age_range and assining label for each
age_bracket_dict = {
(0, 5): 1,
(6, 11): 2,
(12, 17): 3,
(18, 23): 4,
(24, 29): 5,
(30, 35): 6,
(36, 41): 7,
(42, 47): 8,
(48, 53): 9,
(54, 59): 10,
(60, 65): 11,
(66, 71): 12,
(72, 77): 13,
(78, 83): 14,
(84, 89): 15,
(90, 95): 16,
(96, 100): 17,
(101, 106): 18

}

#applying for loop for converting age range into labels
for age_range, label in age_bracket_dict.items():
  df.loc[(df['Age Bracket'] >= age_range[0]) & (df['Age Bracket'] <= age_range[1]), 'Age Bracket'] = label


#creating age group dataset where age bracket and count according to size and saving them in count column
age_group = df.groupby(['Age Bracket']).size().reset_index(name='count')
st.write(age_group.head())

#spliting the age group data into x and y 
x = age_group[['Age Bracket']]
y=age_group[['count']]


#spliting into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
model = LinearRegression()


from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

poly_reg = PolynomialFeatures(degree=7)
X_poly = poly_reg.fit_transform(x)


model = LinearRegression()
model.fit(X_poly, y) 

y_pred = model.predict(X_poly)

rmse = np.sqrt(mean_squared_error(y, y_pred))
st.write("Root Mean Squared Error: ", rmse)

plt.scatter(y, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

import seaborn as sns

fig, ax = plt.subplots()
sns.regplot(x=y, y=y_pred, ax=ax)
ax.set_xlabel("True Values")
ax.set_ylabel("Predictions")

# Display the plot in Streamlit
st.pyplot(fig)


# Plot the regression line
plt.scatter(x, y, color='red')
plt.plot(x, model.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


#def fucntion for converting age into label
def age_to_label(age):
    for age_range, label in age_bracket_dict.items():
        if age >= age_range[0] and age <= age_range[1]:
            return label
    return None



# New data to make predictions on
name = st.text_input("Enter your name")

new_data = st.number_input('Enter your age ')
new_age = age_to_label(new_data)
new_age=[new_age]

new_age = np.array(new_age)

# Transform the new data into polynomial terms
new_data_poly = poly_reg.transform(new_age.reshape(-1, 1))



# Make predictions using the trained model

predictions = model.predict(new_data_poly)
st.write("Prediction: ", predictions)
  





## knowkledge deiscovery for the user and this is for to show output
if  2000 < predictions < 3000:
  st.write(name, 'If youre feeling unwell or experiencing any COVID-19 symptoms, such as fever or cough, self-isolate and get tested as soon as possible')
elif 4000 < predictions < 5000:
  st.write(name, 'Its important to take extra precautions to protect yourself and others from COVID-19, such as avoiding crowded places and limiting your social interactions')
elif 5000 < predictions < 6000:
  st.write(name, 'Wearing a mask in public settings, particularly in indoor settings or areas where there is a high transmission of COVID-19')
elif 6000 < predictions < 7000:
  st.write(name, 'Using hand sanitizer or washing your hands frequently, especially after being in public places')
elif 7000 < predictions < 8000:
  st.write(name, 'Its essential to follow public health guidelines, such as wearing a mask, social distancing, and washing hands frequently, to prevent the spread of COVID-19')
elif 8000 < predictions < 9000:
  st.write(name, 'Staying home if you are feeling unwell or have been in close contact with someone who has tested positive for COVID-19')
elif 9000 < predictions < 10000:
  st.write(name, 'Consider getting vaccinated as soon as possible, as its one of the most effective ways to protect yourself from COVID-19')
else :
  st.write(name, 'Keep in mind that even if you are safe, the situation can change rapidly, and its important to remain vigilant and follow public health guidelines at all times')


