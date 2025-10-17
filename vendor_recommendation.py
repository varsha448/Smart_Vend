# Importing necessary libraries
import pandas as pd
import numpy as np
import logging
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vendor_rag")

# Vector Creation Function
def batch_insert_chroma(texts, metadatas, embeddings, batch_size=5000):
    client_settings = Settings(anonymized_telemetry=False)
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory="./vendor_db",
        client_settings=client_settings
    )

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(batch_texts, batch_metadatas)
        ]
        vector_db.add_documents(batch_docs)
        logger.info(f"Inserted batch {i//batch_size + 1}")

    return vector_db

# Function to calculate total cost
def calculate_total_cost(df):
    logger.info("Calculating total cost")
    df["Total_Cost"] = df["Number_of_Units_Ordered"] * df["Cost_per_Unit"]
    return df

# Function to calculate Fulfillment rate
def calculate_fulfillment_rate(df):
    logger.info("Calculating fulfillment rate")
    df["Fulfillment_Rate"] = (df["Number_of_Units_Sold"] / df["Number_of_Units_Ordered"]) * 100
    return df

# Function to calculate breach rate    
def calculate_breach_rate(df, w_q=0.6, w_t=0.4, max_delay_days=7):
    logger.info("Calculating breach rate")
    def breach_formula(row):
        UO, US = row["Number_of_Units_Ordered"], row["Number_of_Units_Sold"]
        DE, ED = row["Delivery_End_Date"], row["Estimated_Delivery_Date"]
        if US == 0:
            return 1.0
        FR = US / UO
        if pd.isnull(DE) or pd.isnull(ED):
            DR = 0
        else:
            DE, ED = pd.to_datetime(DE), pd.to_datetime(ED)
            delay_days = max(0, (DE - ED).days)
            DR = min(1.0, delay_days / max_delay_days)
        return round(w_q * (1 - FR) + w_t * DR, 3)
    df["Breach_Rate"] = df.apply(breach_formula, axis=1)
    return df

# Function to assign the delivery status
def assign_delivery_status(df):
    logger.info("Assigning delivery status")
    def status_logic(row):
        UO, US = row["Number_of_Units_Ordered"], row["Number_of_Units_Sold"]
        DE, ED = row["Delivery_End_Date"], row["Estimated_Delivery_Date"]
        if US == 0:
            return "Not Delivered"
        if pd.isnull(DE) or pd.isnull(ED):
            return "Unknown"
        DE, ED = pd.to_datetime(DE), pd.to_datetime(ED)
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

# Vendor recommendation function
def load_vendor_rag():
    logger.info("Reading vendor dataset")
    vendor_df = pd.read_csv("../vendor_recommendation.csv")

    # Calling the necessary functions
    vendor_df = calculate_total_cost(vendor_df)
    vendor_df = calculate_fulfillment_rate(vendor_df)
    vendor_df = calculate_breach_rate(vendor_df)
    vendor_df = assign_delivery_status(vendor_df)

    # Convert to LangChain documents
    logger.info("Converting rows to LangChain documents")
    texts, metadatas = [], []
    for _, row in vendor_df.iterrows():
        text = (
            f"Vendor {row['Vendor_Name']} has product {row['Product_Name']}, operating at various regions such as {row['Region']} "
            f"with cost of {row['Total_Cost']}, turnaround time {row['Turnaround_Time']} days, "
            f"fulfillment rate {row['Fulfillment_Rate']}%, and breach rate {row['Breach_Rate']}."
        )
        texts.append(text)
        metadatas.append({"Vendor_ID": row["Vendor_ID"], "Product_ID": row["Product_ID"]})

    # Load embedding model
    logger.info("Loading embedding model")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    # Precompute embeddings
    logger.info("Generating embeddings")
    vectors = embeddings.embed_documents(texts)

    # Build vector store
    logger.info("Building Chroma vector store")
    vector_db = batch_insert_chroma(texts, metadatas, embeddings)

    # Load GPT-Neo LLM
    logger.info("Loading GPT-Neo model pipeline")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=generator)
    
    custom_prompt = PromptTemplate.from_template("""
    You are a vendor recommendation assistant. Based on the user's query and the provided context, extract only the relevant vendor recommendation lines.

    Each line should include: vendor name, product, region, cost, turnaround time, fulfillment rate, and breach rate.

    Only include vendor lines that match the user's query. Round cost, fulfillment rate, and breach rate to two decimal places. Do not include any explanation or extra text.

    Return up to the number of vendors requested in the user's query.

    User query: {question}

    Context:
    {context}

    """)

    # Creating RAG pipeline
    logger.info("Creating RetrievalQA pipeline")
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 20}),
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt},
    input_key="question"
    )

    logger.info("RAG pipeline ready")
    return qa