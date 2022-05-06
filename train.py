import pandas as pd, numpy as np, pickle, os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

os.chdir(os.path.dirname(__file__))

data = pd.read_csv("data/Advertising.csv", index_col=0)

X = data.drop("sales", axis=1)
y = data["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=24)

model = Lasso(alpha=6000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions) 
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

with open("data/model/hm_model.pkl", "wb") as a_guardar:
    pickle.dump(model, a_guardar)