import numpy as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataset=pd.read_csv('home_data.csv')
pd.set_option('display.max_columNs',None)

print(dataset.head())

X=dataset[['bedrooms','bathrooms','condition','grade','sqft_living','sqft_living15','sqft_lot', 'floors', 'waterfront', 'view', 'yr_built', 'yr_renovated','lat','long']]
Y=dataset['price']

print(dataset.isnull().sum())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)

model=RandomForestRegressor()
model.fit(X_train, Y_train)

prediction=model.predict(X_test)
score = r2_score(Y_test, prediction)
print(score)

print("enter house details to get a price")

bedrooms=float(input("bedrooms:"))
bathrooms = float(input("Bathrooms: "))
condition = float(input("Condition (1-5): "))
grade = float(input("Grade (1-13): "))
sqft_living = float(input("Sqft living: "))
sqft_living15 = float(input("Sqft living15 (avg of 15 nearest neighbors): "))
sqft_lot = float(input("Sqft lot: "))
floors = float(input("Floors: "))
waterfront = float(input("Waterfront (0 or 1): "))
view = float(input("View (0-4): "))
yr_built = float(input("Year built: "))
yr_renovated = float(input("Year renovated (0 if never): "))
lat = float(input("Latitude: "))
long = float(input("Longitude: "))

user_input=pd.DataFrame([[bedrooms, bathrooms, condition, grade, sqft_living,
                             sqft_living15, sqft_lot, floors, waterfront, view,
                             yr_built, yr_renovated, lat, long]], columns=X.columns)

predicted_price=model.predict(user_input)
print("Predicted price: $", round(predicted_price[0], 2))