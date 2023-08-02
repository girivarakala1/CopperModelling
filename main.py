import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

import pickle


class Visual:

    @staticmethod
    def disrtibution_plot(col, title='title'):
        sns.distplot(data[col])
        plt.title(title)
        fig = title + ".png"
        plt.savefig(fig)

    @staticmethod
    def box_plot(col):
        sns.boxplot(x=col, data=data)
        plt.title('Boxplot - Outliers')

        plt.savefig("{} box plot.png".format(col))

    @staticmethod
    def counter_plot(col):
        sns.countplot(data=data, x=col)
        plt.title('{} Feature Distribution'.format(col))
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.savefig('{} Feature Distribution.png'.format(col))

class Feature(Visual):
    # Define a function to convert values starting with '00000' to null
    @staticmethod
    def convert_to_null(value):
        if str(value).startswith('00000'):
            return None
        else:
            return value

    @staticmethod
    def fill_missing_values():
        # Examine numerical and categorical features
        numerical_features = data.select_dtypes(include=[np.number])
        categorical_features = data.select_dtypes(include=[object])

        for column in numerical_features:
            # Fill null values with the mean
            data[column].fillna(data[column].mean(), inplace=True)

        for feature in categorical_features:
            # Fill null values with the mode
            data[feature].fillna(data[feature].mode()[0], inplace=True)


    @staticmethod
    def modify_outliers(col):

        # Calculate the first quartile (Q1) and third quartile (Q3)
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)

        # Calculate the interquartile range (IQR)
        iqr = q3 - q1

        # Define the lower and upper bounds for outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Modify outliers to the lower and upper bounds in the dataset
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

    @staticmethod
    def skewness(col):
        data[col] = pd.to_numeric(data[col], errors='coerce')

        # Calculate skewness
        skewness = stats.skew(data[col])

        # Visualize skewness using distplot
        Visual.disrtibution_plot(col, title='{} Distribution Plot - before Skewness Treatment'.format(col))

        # Log transformation
        log_skewness = stats.skew(np.log(data[col]))

        # Square root transformation
        sqrt_skewness = stats.skew(np.sqrt(data[col]))

        # Cubic root transformation
        cbrt_skewness = stats.skew(np.cbrt(data[col]))

        # Box-Cox transformation
        box_cox_skewness = 1
        box_cox_data = 1
        try:
            box_cox_data, _ = stats.boxcox(data[col])
            box_cox_skewness = stats.skew(box_cox_data)
        except:
            pass

        if box_cox_skewness != 1:
            transform = min(abs(skewness), abs(log_skewness), abs(sqrt_skewness), abs(cbrt_skewness),
                            abs(box_cox_skewness))
        else:
            transform = min(abs(skewness), abs(log_skewness), abs(sqrt_skewness), abs(cbrt_skewness))

        if transform == log_skewness:
            res = "log transformation"
            data[col] = np.log(data[col])
        elif transform == sqrt_skewness:
            res = "square root transformation"
            data[col] = np.sqrt(data[col])
        elif transform == cbrt_skewness:
            res = "cubic root transformation"
            data[col] = np.cbrt(data[col])
        elif min(data[col] >= 0) and transform == box_cox_skewness:
            res = "box_cox transformation"
            data[col] = box_cox_data
        else:
            res = "no transformation"

        # Visualize skewness using distplot
        Visual.disrtibution_plot(col, title='{0} Distribution Plot - after {1}'.format(col, res))

def pickle_data():
    # Save XGBoost model to a pickle file
    with open('xgbreg_model.pkl', 'wb') as file:
        pickle.dump(xgb_reg_model, file)

    # Save Random Forest model to a pickle file
    with open('rfreg_model.pkl', 'wb') as file:
        pickle.dump(rf_reg_model, file)
        # Save XGBoost model to a pickle file
    with open('xgbclf_model.pkl', 'wb') as file:
        pickle.dump(xgb_clf_model, file)

    # Save Random Forest model to a pickle file
    with open('rfclf_model.pkl', 'wb') as file:
        pickle.dump(rf_clf_model, file)


# Replace 'file_path.xlsx' with the path to your Excel file
file_path = 'C:\\Users\\giriv\\CopperModelling\\data\\Copper_Set.xlsx'

# Read the Excel file into a pandas DataFrame
data = pd.read_excel(file_path, sheet_name=0)
data['quantity tons'] = pd.to_numeric(data['quantity tons'], errors='coerce')

# Examine numerical and categorical features
numerical_features = data.select_dtypes(include=[np.number])

# Apply the function to the 'Material_Reference' column
data['material_ref'] = data['material_ref'].apply(Feature.convert_to_null)

Feature.fill_missing_values()

for col in numerical_features:
    Visual.box_plot(col)
    Feature.modify_outliers(col)

for col in numerical_features:
    Feature.skewness(col)

Visual.counter_plot('item type')
data.drop(columns=['id', 'material_ref'], inplace=True)

# Create an instance of OneHotEncoder
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data[['item type']])
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['item type']))
data.drop('item type', axis=1, inplace=True)
dataframe = pd.concat([data, encoded_df], axis=1)

# Separate features and target variable predict selling price
X_reg = dataframe.drop(columns=['selling_price', 'status'])
y_reg = dataframe['selling_price']

# Step 5: ML Classification Model for 'Status' Prediction
# Filter the dataset to include only 'WON' and 'LOST' STATUS values
df = dataframe[dataframe['status'].isin(['Won', 'Lost'])]

# Separate features and target variable
X_clf = df.drop(columns=['status', 'selling_price'])
y_clf = df['status'].map({'Won': 1, 'Lost': 0})

# Regression models to predict selling price

# Split the data into training and testing sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Feature Scaling (Normalize/Standardize)
scaler_reg = StandardScaler()
X_reg_train = scaler_reg.fit_transform(X_reg_train)
X_reg_test = scaler_reg.transform(X_reg_test)

# Train the regression model
xgb_reg_model = XGBRegressor()
xgb_reg_model.fit(X_reg_train, y_reg_train)
y_xgbreg_pred = xgb_reg_model.predict(X_reg_test)

# Calculate regression metrics
# xgb_reg_mse = mean_squared_error(y_reg_test, y_xgbreg_pred)
# xgb_reg_rmse = mean_squared_error(y_reg_test, y_xgbreg_pred, squared=False)
# xgb_reg_mae = mean_absolute_error(y_reg_test, y_xgbreg_pred)
# xgb_reg_r2 = r2_score(y_reg_test, y_xgbreg_pred)

rf_reg_model = RandomForestRegressor()
rf_reg_model.fit(X_reg_train, y_reg_train)
y_rfreg_pred = rf_reg_model.predict(X_reg_test)

# Calculate regression metrics
# rf_reg_mse = mean_squared_error(y_reg_test, y_rfreg_pred)
# rf_reg_rmse = mean_squared_error(y_reg_test, y_rfreg_pred, squared=False)
# rf_reg_mae = mean_absolute_error(y_reg_test, y_rfreg_pred)
# rf_reg_r2 = r2_score(y_reg_test, y_rfreg_pred)

# Split the data into training and testing sets
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Train the XGB classification model
xgb_clf_model = XGBClassifier()
xgb_clf_model.fit(X_clf_train, y_clf_train)
y_xgbclf_pred = xgb_clf_model.predict(X_clf_test)
# xgb_clf_accuracy = accuracy_score(y_clf_test, y_xgbclf_pred)
# xgb_clf_conf_matrix = confusion_matrix(y_clf_test, y_xgbclf_pred)
# xgb_clf_classification_rep = classification_report(y_clf_test, y_xgbclf_pred)

# Train the RF classification model
rf_clf_model = RandomForestClassifier()
rf_clf_model.fit(X_clf_train, y_clf_train)
y_rfclf_pred = rf_clf_model.predict(X_clf_test)
# rf_clf_accuracy = accuracy_score(y_clf_test, y_rfclf_pred)
# rf_clf_conf_matrix = confusion_matrix(y_clf_test, y_rfclf_pred)
# rf_clf_classification_rep = classification_report(y_clf_test, y_rfclf_pred)

# cross vallodation for regression models
xgbreg_cv_scores = cross_val_score(xgb_reg_model, X_reg, y_reg, cv=5)
rfreg_cv_scores = cross_val_score(rf_reg_model, X_reg, y_reg, cv=5)

# cross validation for classification models
xgbclf_cv_scores = cross_val_score(xgb_clf_model, X_clf, y_clf, cv=5)
rfclf_cv_scores = cross_val_score(rf_clf_model, X_clf, y_clf, cv=5)

pickle_data()
