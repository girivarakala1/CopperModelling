import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import datetime


def date_to_string(value):
    date_string = value.strftime('%Y-%m-%d')
    date_string = date_string.replace("-", "")
    value = float(date_string)
    return value


st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",

    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def main():
    tab1, tab2, tab3 = st.tabs(["Home","Prediction", "Feature Model"])

    with tab1:
        st.title("Copper Industry - ML Predictions")
        st.write("The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data")
        st.write("RandomForestRegressor, XGBRegressor are ML Regression models which predicts continuous variable ‘Selling_Price’. RandomForestClassifier and XGBClassifier are ML Classification models which predicts Status: WON or LOST")
        st.write("cross validation shows that it is better to use XGBRegressor to predict selling price and RandomForestClassifier to predict status.")
        st.write("in Prediction tab you can enter new input data to predict selling price and status")
        st.write("in Feature model tab you can see outliers, before and after treatment for every column")

    with tab2:
        st.title("ML Predictions")
        st.write("Home page text")


        input_data = {
                'item_date': 0.0,
                'quantity tons': 0.0,
                'customer': 0.0,
                'country': 0.0,
                'application': 0.0,
                'thickness': 0.0,
                'width': 0.0,
                'product_ref': 0.0,
                'delivery date': 0.0,
                'item type_IPL': 0,
                'item type_Others': 0,
                'item type_PL': 0,
                'item type_S': 0,
                'item type_SLAWR': 0,
                'item type_W': 0,
                'item type_WI': 0
        }

        # User input for date fields
        itemdate = st.date_input("Enter item date", datetime.date(2021, 7, 6))
        input_data['item_date'] = date_to_string(itemdate)

        delivery_date = st.date_input("Enter delivery date", datetime.date(2021, 7, 6))
        input_data['delivery date'] = date_to_string(delivery_date)

        input_data['quantity tons'] = st.number_input("Enter quantity in tons", value=0.0)
        input_data['customer'] = st.number_input("Enter customer", value=0.0)
        input_data['country'] = st.number_input("Enter country code", value=0.0)
        input_data['application'] = st.number_input("Enter application", value=0.0)
        input_data['thickness'] = st.number_input("Enter thickness", value=0.0)
        input_data['width'] = st.number_input("Enter width", value=0.0)
        input_data['product_ref'] = st.number_input("Enter product_ref", value=0.0)

        # User input for one-hot encoded categorical feature 'item type'
        item_type_values = ['IPL', 'Others', 'PL', 'S', 'SLAWR', 'W', 'WI']
        selected_item_types = st.selectbox("Select item types", item_type_values)

        for item_type in selected_item_types:
            input_data[f'item type_{item_type}'] = 1


        if st.button('Predict Selling Price'):
            # Prepare input_data and convert to a DataFrame
            input_data_df = pd.DataFrame([input_data])

            # Load XGBoost model from the pickle file
            with open('C:\\Users\\giriv\\CopperModelling\\xgbreg_model.pkl', 'rb') as file:
                loaded_xgbreg_model = pickle.load(file)
                # Predict Selling_Price
                xgb_predicted_selling_price = loaded_xgbreg_model.predict(input_data_df)
                st.write('XGB predicted selling price: ', xgb_predicted_selling_price[0])

        if st.button('Predict Status'):
            # Prepare input_data and convert to a DataFrame
            input_data_clf = pd.DataFrame([input_data])

            # Load RandomForest classifier model from the pickle file
            with open('C:\\Users\\giriv\\CopperModelling\\rfclf_model.pkl', 'rb') as file:
                loaded_rfclf_model = pickle.load(file)
                # Predict Status
                rf_predicted_status = loaded_rfclf_model.predict(input_data_clf)
                predicted_status = 'WON' if rf_predicted_status == 1 else 'LOST'
                st.write('RandomForest prediction: ', predicted_status[0:])

    with tab3:
        option = st.selectbox(
            'How would you like to be contacted?',
            ("SELECT", 'item_date', 'quantity tons', 'customer', 'country','item type', 'application',
             'thickness', 'width', 'product_ref', 'delivery date', ))

        if option != "SELECT" and option != 'item type':
            image = Image.open('C:\\Users\\giriv\\CopperModelling\\images\\{} box plot.png'.format(option))
            st.image(image, caption='Outliers in {}'.format(option))
            image = Image.open(
                'C:\\Users\\giriv\\CopperModelling\\images\\{} Distribution Plot - before Skewness Treatment.png'.format(
                    option))
            st.image(image, caption='before skewness effect on {}'.format(option))

            if option in ['application', 'country', 'product_ref', 'thickness', 'width']:
                image = Image.open(
                    'C:\\Users\\giriv\\CopperModelling\\images\\{} Distribution Plot - after box_cox transformation.png'.format(
                        option))
                st.image(image, caption='after skewness treatment on {}'.format(option))
            if option in ['customer', 'item_date']:
                image = Image.open(
                    'C:\\Users\\giriv\\CopperModelling\\images\\{} Distribution Plot - after log transformation.png'.format(
                        option))
                st.image(image, caption='after skewness treatment on {}'.format(option))
            if option == "quantity tons":
                image = Image.open(
                    'C:\\Users\\giriv\\CopperModelling\\images\\{} Distribution Plot - after cubic root transformation.png'.format(
                        option))
                st.image(image, caption='after skewness treatment on {}'.format(option))
        elif option == 'item type':
            image = Image.open('C:\\Users\\giriv\\CopperModelling\\images\\item type Feature Distribution.png')
            st.image(image, caption= 'item distribution')



if __name__ == "__main__":
     main()


