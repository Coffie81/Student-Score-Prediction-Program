import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('student_scores_real.csv')


X = df[['Hours']]
y = df['Scores']

#Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)


#Making a Prediction
study_hours = np.array([[4]])
predicted_score = model.predict(study_hours)

print("Predicted Score ",predicted_score)

# Prediction on the test set
y_pred = model.predict(X_test)

# Compare actual vs predicted
comparison = pd.DataFrame({
    "Hours (X_test)": X_test["Hours"].values,
    "Actual Scores": y_test.values,
    "Predicted Scores": y_pred
})

comparison

#visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.show()

