from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

df = pd.read_csv('D:\MEENA\Sem 2 - Crop Recommendation\Agriculture crop recommendation\Crop_recommendation - Crop_recommendation.csv')
current_dir = os.getcwd()

# Load the trained model
loaded_model = pickle.load(open("model.pkl", 'rb'))
print(os.listdir(current_dir))


# Initialize the LabelEncoder
label_encoder = LabelEncoder()


X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the labels
y_encoded = label_encoder.fit_transform(y)

# Split the dataset with the encoded labels
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)



# Load the trained models
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)

random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

svm_model = SVR()
svm_model.fit(X_train, y_train)

# Calculate the mean squared error (MSE) of the models
decision_tree_mse = mean_squared_error(y_test, decision_tree_model.predict(X_test))
random_forest_mse = mean_squared_error(y_test, random_forest_model.predict(X_test))
svm_mse = mean_squared_error(y_test, svm_model.predict(X_test))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    crop_dict = {0: "Rice", 1: "Maize", 2: "Jute", 3: "Cotton", 4: "Coconut", 5: "Papaya", 6: "Orange",
                 7: "Apple", 8: "Muskmelon", 9: "Watermelon", 10: "Grapes", 11: "Mango", 12: "Banana",
                 13: "Pomegranate", 14: "Lentil", 15: "Blackgram", 16: "Mungbean", 17: "Mothbeans",
                 18: "Pigeonpeas", 19: "Kidneybeans", 20: "Chickpea", 21: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('index.html', prediction=result)



@app.route('/model')
def model():
    # Prepare data for the bar plot
    models = ['Decision Tree', 'Random Forest', 'SVM']
    mses = [decision_tree_mse, random_forest_mse, svm_mse]

    # Plot the bar chart
    plt.bar(models, mses)
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Mean Squared Errors')
    plt.ylim(0, max(mses) * 1.1)

    # Save the plot to a file
    plt.savefig('static/model_plot.png')

    return render_template('model.html', mses=mses)


if __name__ == '__main__':
    app.run(debug=True)