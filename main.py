import openai
import streamlit as st
import pandas as pd
#load model
import pickle
import numpy as np #run  pip install xgboost scikit-learn in shell / terminal and pip install openai
from typing import Dict, Any
import os # to access GROQ API key
# from openai import OpenAI
import google.generativeai as genai

from utils import create_gauge_chart
from utils import create_model_probability_chart
from utils import create_percentile_chart



genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file )
    return model

#####################################################################
# Load model
xgb_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
xgboot_SMOTE_model = load_model('xgboost-SMOT.pkl')
xgb_featuredEngineered_model = load_model('xgboost-featuredEngineered.pkl')

#####################################################################
# Function to prepare input data for the models

def prepare_input(credit_score , location , gender ,age ,tenure ,balance,
                 num_products, has_credit_card, is_active_member, estimated_salary):
  input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCreditCard': int(has_credit_card),
    'IsActiveMember': int(is_active_member),
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if location == 'France' else 0,
    'Geography_Germany': 1 if location == 'Germany' else 0,
    'Geography_Spain': 1 if location == 'Spain' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Gender_Female': 1 if gender == 'Female' else 0
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict
  
#####################################################################

#function to make prediction using the machin learning models we train

def make_predictions(input_df, input_dict):
  probabilities = {
    'XGBoost': xgb_model.predict_proba(input_df)[0][1],
    'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
    'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
  }
  avg_probability = np.mean(list(probabilities.values()))

  # Update the plots
  st.markdown("---")
  col1, col2 = st.columns(2)
  with col1:
      fig = create_gauge_chart(avg_probability)
      st.plotly_chart(fig, use_container_width=True)
      st.write(
          f'The customer has a {avg_probability:.2%} probability of churning.'
      )
  with col2:
      fig_probs = create_model_probability_chart(probabilities)
      st.plotly_chart(fig_probs, use_container_width=True)

  
  st.markdown("## Model Probabilities")
  
  for model, prob in probabilities.items():
    st.write(f"{model}: {prob}")
  st.write(f"Average Probability: {avg_probability}")
  return avg_probability 

#####################################################################

def explain_prediction(
  probability: float,
  input_dict: Dict[str, Any],
  surname: str,
) -> str:
  """Generate an explanation for the churn prediction using Gemini"""

  prompt = f"""You are an expert data scientist at a bank, where you specialize in 
  interpreting and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a 
  {round(probability * 100, 1)}% probability of churning, based on the information provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting churn:

  Feature | Importance
  -----------------------
  NumOfProducts | 0.323888
  IsActiveMember | 0.164146
  Age | 0.109550
  Geography_Germany | 0.091373
  Balance | 0.052786
  Geography_France | 0.046463
  Gender_Female | 0.045283
  Geography_Spain | 0.036855
  CreditScore | 0.035005
  EstimatedSalary | 0.032655
  HasCrCard | 0.031940
  Tenure | 0.030054
  Gender_Male | 0.000000

  Here are summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}

  Here are summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}

  WORD RESTRICTION: 100-150 words!! IT IS VERY IMPORTANT. 

  - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
  - If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.

  Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

  Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.
  """
  # response = client.chat.completions.create(
  #     model="gpt-4o",
  #     messages=[      
  #         {"role": "user", "content": prompt}
  #     ]
  # )
  #response = client.generate_content(prompt)
  response = model.generate_content(prompt)
  
  return response.text


#####################################################################

def generate_email(
  probability: float,
  input_dict: Dict[str, Any],
  explanation: str,
  surname: str
) -> str:
  """Generate a customized email for the customer using Gemini"""

  prompt = f"""You are a manager at HS Bank. You are responsible for 
  ensuring customers stay with the bank and are incentivized with various offers.

  WORD RESTRICTION: 250-350 words!! IT IS VERY IMPORTANT. 

  You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

  Here is the customer's information:
  {input_dict}

  Here is some explanation as to why the customer might be at risk of churning:
  {explanation}

  Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.
  Be specific about the incentives, and make sure to emphasize that the customer is not at risk of churning.
  Email should be straight forward, facts only 250-350 words restriction. 

  Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.
  """

  response = model.generate_content(prompt)
 
  return response.text


#####################################################################

def generate_percentiles(df, input_dict):
  all_num_products = df['NumOfProducts'].sort_values().tolist()
  all_balances = df['Balance'].sort_values().tolist()
  all_estimated_salaries = df['EstimatedSalary'].sort_values().tolist()
  all_tenures = df['Tenure'].sort_values().tolist()
  all_credit_scores = df['CreditScore'].sort_values().tolist()

  product_rank = np.searchsorted(all_num_products, input_dict['NumOfProducts'], side='right')
  balance_rank = np.searchsorted(all_balances, input_dict['Balance'], side='right')
  salary_rank = np.searchsorted(all_estimated_salaries, input_dict['EstimatedSalary'], side='right')
  tenure_rank = np.searchsorted(all_tenures, input_dict['Tenure'], side='right')
  credit_rank = np.searchsorted(all_credit_scores, input_dict['CreditScore'], side='right')


  N = 10000

  percentiles = {
    'CreditScore': int(np.floor((credit_rank / N) * 100)),
    'Tenure': int(np.floor((tenure_rank / N) * 100)),
    'EstimatedSalary': int(np.floor((salary_rank / N) * 100)),
    'Balance': int(np.floor((balance_rank / N) * 100)),
    'NumOfProducts': int(np.floor((product_rank / N) * 100)),
  }


  fig = create_percentile_chart(percentiles)
  st.plotly_chart(fig,use_container_width=True)


  return percentiles



#####################################################################

st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
  
    print("Selected customer ID:", selected_customer_id)
  
    selected_surname = selected_customer_option.split(" - ")[1]
  
    print("Selected surname:", selected_surname)
 
  #   customer_data = df[df["CustomerId"] == selected_customer_id]
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]
    print("Selected customer data:", selected_customer)
  
    

    col1,col2= st.columns(2)
  
    with col1:
      credit_score = st.number_input(
        "Credit Score", 
        min_value = 300,
        max_value = 850,
        value = int(selected_customer["CreditScore"]))
      location = st.selectbox("Location", ["Spain", "France", "Germany"],
                             index=["Spain", "France","Germany"].index(selected_customer["Geography"]))

      gender = st.radio("Gender", ["Male", "Female"],
                       index=0 if selected_customer["Gender"] == "Male" else 1)
      age = st.number_input("Age",
                            min_value=18,
                            max_value=100,
                            value=int(selected_customer["Age"]))
      tenure =st.number_input("Tenure(years)",
                             min_value=0,
                             max_value=100,
                             value=int(selected_customer["Tenure"]))
    with col2:
      balance = st.number_input("Balance",
                                min_value=0.0,
                                value = float(selected_customer["Balance"]))
      
      num_of_products = st.number_input("Number of Products",
                                        min_value = 1,
                                        max_value = 10,
                                        value =int(selected_customer["NumOfProducts"]))
      
      has_credit_card = st.checkbox("Has Credit Card",
                                   value = bool(selected_customer["HasCrCard"]  ))

      is_active_member = st.checkbox("Is Active Member",
                                     value = bool(selected_customer["IsActiveMember"]))

      estimated_salary = st.number_input("Estimated Salary",
                                         min_value = 0.0,
                                         value = float(selected_customer["EstimatedSalary"]))


input_df, input_dict = prepare_input(credit_score, location,
                                     gender, age, tenure,
                                     balance , num_of_products,
                                     has_credit_card,is_active_member,
                                     estimated_salary)

avg_probability = make_predictions(input_df, input_dict)


generate_percentiles(df, input_dict)

# Explain prediction  
with st.spinner("Explaining..."):
  explanation = explain_prediction(avg_probability, input_dict,
                                selected_customer["Surname"])
st.markdown("---")
st.subheader("Explanation of Prediction")
st.markdown(explanation)

#Email generation
with st.spinner("Writing an email..."):
  email = generate_email(avg_probability, input_dict, explanation,
                    selected_customer["Surname"])
st.markdown("---")
st.subheader("Personalized Email")
st.markdown(email)
      
        