"""Importing the libraries"""
import pandas as pd
import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

"""Reading the dataset"""
vendor_df = pd.read_csv("/content/vendor_recommendation.csv")
print(vendor_df.head())

"""Processing the Dataset"""
def calculate_total_cost(df):
  df["Total_Cost"] = df["Number_of_Units_Ordered"] * df["Cost_per_Unit"]
  return df

def calculate_fulfillment_rate(df):
  df["Fulfillment_Rate"] = (df["Number_of_Units_Sold"] / df["Number_of_Units_Ordered"]) * 100
  return df

def calculate_breach_rate(df, w_q=0.6, w_t=0.4, max_delay_days=7):
    def breach_formula(row):
        UO = row["Number_of_Units_Ordered"]
        US = row["Number_of_Units_Sold"]
        DE = row["Delivery_End_Date"]
        ED = row["Estimated_Delivery_Date"]

        # Case 1: No units sold → max breach
        if US == 0:
            return 1.0

        # Fulfillment ratio
        FR = US / UO

        # Delay ratio
        if pd.isnull(DE) or pd.isnull(ED):
            DR = 0
        else:
            DE = pd.to_datetime(DE)
            ED = pd.to_datetime(ED)
            delay_days = max(0, (DE - ED).days)
            DR = min(1.0, delay_days / max_delay_days)

        # Weighted breach rate
        return round(w_q * (1 - FR) + w_t * DR, 3)

    df["Breach_Rate"] = df.apply(breach_formula, axis=1)
    return df

def assign_delivery_status(df):
    def status_logic(row):
        UO = row["Number_of_Units_Ordered"]
        US = row["Number_of_Units_Sold"]
        DE = row["Delivery_End_Date"]
        ED = row["Estimated_Delivery_Date"]

        if US == 0:
            return "Not Delivered"

        if pd.isnull(DE) or pd.isnull(ED):
            return "Unknown"

        DE = pd.to_datetime(DE)
        ED = pd.to_datetime(ED)
        delay_days = (DE - ED).days

        if US < UO and delay_days <= 0:
            return "Partially Fulfilled – On Time"
        elif US < UO and delay_days > 0:
            return "Partially Fulfilled – Late"
        elif US == UO and delay_days <= 0:
            return "Delivered On Time"
        elif US == UO and delay_days > 0:
            return "Delivered Late"
        else:
            return "Unknown"

    df["Delivery_Status"] = df.apply(status_logic, axis=1)
    return df

vendor_df = calculate_total_cost(vendor_df)
print("Vendor DataFrame with Total Cost:")
print(vendor_df.head())

vendor_df = calculate_fulfillment_rate(vendor_df)
print("Vendor DataFrame with Fulfillment Rate:")
print(vendor_df.head())

vendor_df = calculate_breach_rate(vendor_df)
print("Vendor DataFrame with Breach Rate:")
print(vendor_df.head())

vendor_df = assign_delivery_status(vendor_df)
print("Vendor DataFrame with Delivery Status:")
print(vendor_df.head())

"""Converting the dataframe to document"""
docs = []
for i, row in vendor_df.iterrows():
    text = f"Vendor {row['Vendor_Name']} has product {row['Product_Name']}, operating at various regions such as {row['Region']} with cost of {row['Total_Cost']}, along with turnaround time being {row['Turnaround_Time']} days, along with fulfillment rate of {row['Fulfillment_Rate']}%, and a breach rate of {row['Breach_Rate']}."
    docs.append(Document(page_content=text, metadata={"Vendor_ID": row["Vendor_ID"], 'Product_ID': row['Product_ID']}))

print("Printing first five documents:")
for doc in docs[:5]:
    print(doc)

"""Creating the Embedding and storing it in Vector DB"""
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

"""Store docs in Chroma vector DB"""
vector_db = Chroma.from_documents(docs, embeddings, persist_directory="./vendor_db")
vector_db.persist()

generator = pipeline(
    "text-generation",
    model= "tiiuae/falcon-7b-instruct",
    device=0,
    max_new_tokens=256
)

"""Loading the LLM"""
llm = HuggingFacePipeline(pipeline=generator)

"""Creating the RAG Pipeline"""
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever()
)

"""Running the Query"""
query = "Give me top 5 vendor names along with their cost which has low cost and low breach rate for LED Monitor product"
response = qa.run(query)
print(response)