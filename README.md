# Big-Sales-Prediction
Big Sales Data Analysis and Prediction
Description

This project analyzes sales data to build a predictive model using a Random Forest Regressor. It involves data loading, preprocessing, visualization, modeling, and evaluation of the sales predictions.
Dataset

The dataset used in this project is the Big Sales Data, which contains various features related to sales performance.
Dependencies

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn

Installation

To install the necessary dependencies, you can use pip:

bash

pip install pandas numpy matplotlib seaborn scikit-learn

Usage

    Load the Dataset: Load the dataset from the provided URL.
    Data Visualization: Visualize the distribution of the sales and the relationships between numerical features.
    Data Preprocessing: Handle missing values, drop unnecessary columns, and encode categorical variables.
    Model Training: Train a Random Forest Regressor on the preprocessed data.
    Model Evaluation: Evaluate the model using Mean Squared Error (MSE) and make predictions.

File Description

    main.py: Contains the main code for loading data, preprocessing, visualizing, training the model, and evaluating its performance.

Modeling

The modeling process involves:

    Splitting the data into training and testing sets.
    Using a pipeline for scaling features and fitting a Random Forest Regressor.

python

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

Evaluation

The model's performance is evaluated using the Mean Squared Error (MSE):

python

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

Results

After running the model, the output will include the Mean Squared Error and the predictions for the test set.

plaintext

Mean Squared Error: 250.67
Predictions: [3456.78, 2345.67, 4567.89, ...]

Contact

For any questions or feedback, please contact [jhansigonuguntla6@gmail.com].
Contribution

Contributions to this project are welcome! Feel free to submit a pull request or open an issue.
Acknowledgements

    YBI Foundation for providing the dataset.
    The open-source community for libraries and tools used in this project.

License

This project is licensed under the MIT License.








