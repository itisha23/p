## Combination of free embeddings and free Hugging face model

import os
from urllib.request import urlretrieve
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from rouge_score import rouge_scorer

# Load the environment variables
load_dotenv()

################# CREATE KNOWLEDGE BASE ###################

## Extract the data from PDF
os.makedirs("pdf_data", exist_ok=True)
files = [
    "https://assets.churchill.com/motor-docs/policy-booklet-0923.pdf"
]
for url in files:
    file_path = os.path.join("pdf_data", url.rpartition("/")[2])
    urlretrieve(url, file_path)


# Load pdf files in the local directory
loader = PyPDFDirectoryLoader("./pdf_data/")

docs_before_split = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 50,
)
docs_after_split = text_splitter.split_documents(docs_before_split)

avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
avg_char_before_split = avg_doc_length(docs_before_split)
avg_char_after_split = avg_doc_length(docs_after_split)

print(f'Before split, there were {len(docs_before_split)} documents loaded, with average characters equal to {avg_char_before_split}.')
print(f'After split, there were {len(docs_after_split)} documents (chunks), with average characters equal to {avg_char_after_split} (average chunk length).')

## Create Embeddings from source data using Open Source Hugging Face Embeddings
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
    model_kwargs={'device':'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

sample_embedding = np.array(huggingface_embeddings.embed_query(docs_after_split[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)

## Create vector store from embeddings using FAISS
vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)

# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

hf = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature":0.1, "max_length":500})

llm = hf 
 
prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
"""

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=['context', 'question']
)

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

st.title("RAG-powered Chatbot")
query1 = st.text_input("Ask a question")
if query1:
    relevant_documents = vectorstore.similarity_search(query1)
    print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
    print(relevant_documents[0].page_content)
    
    context = " ".join([doc.page_content for doc in relevant_documents])  # Combine content of relevant documents
   
    # Define the query using context and question
    query = {
        'context': context,
        'question': 'What does the policy cover for accidental damage?'
    }

    result = retrievalQA.invoke({"query": query1})
    st.write(f'Final Result is {result['result']}')

## Evaluating Performances

dataset = [
    {"query": "What does the policy cover for accidental damage?", "response": "The policy covers accidental damage to your vehicle, including collisions and crashes."},
    {"query": "How can I renew my policy?", "response": "You can renew your policy by contacting our customer service or through your online account."},
    {"query": "Does the policy cover theft?", "response": "Yes, the policy provides coverage for theft of your vehicle."},
    {"query": "What are the exclusions of the policy?", "response": "Exclusions include damage caused by driving under the influence of alcohol or drugs, intentional damage, and using the vehicle for unlawful purposes."},
    {"query": "How do I file a claim for windshield damage?", "response": "To file a claim for windshield damage, contact our customer service and provide details of the incident."},
    {"query": "Can I add a driver to my policy?", "response": "Yes, you can add a driver to your policy by contacting our customer service or through your online account."},
    {"query": "What is covered under third-party liability?", "response": "Third-party liability covers damages and injuries caused to other people or their property by your vehicle."},
    {"query": "How do I cancel my policy?", "response": "You can cancel your policy by contacting our customer service and following the cancellation process."},
    {"query": "Does the policy cover natural disasters?", "response": "Yes, the policy provides coverage for damages caused by natural disasters such as floods and earthquakes."},
    {"query": "What documents do I need to provide for a claim?", "response": "You will need to provide documents such as the police report, photographs of the damage, and proof of ownership for a claim."},
    {"query": "Is roadside assistance included in the policy?", "response": "Yes, roadside assistance is included in the policy for emergencies such as breakdowns and flat tires."},
    {"query": "Can I transfer my policy to a new vehicle?", "response": "Yes, you can transfer your policy to a new vehicle by contacting our customer service and providing the necessary details."},
    {"query": "What is the deductible for comprehensive coverage?", "response": "The deductible for comprehensive coverage is the amount you are responsible for paying before the insurance coverage kicks in."},
    {"query": "How long does it take to process a claim?", "response": "The time to process a claim varies depending on the complexity of the case and the availability of required documents."},
    {"query": "What is the grace period for policy renewal?", "response": "The grace period for policy renewal is usually 30 days from the expiration date to renew without any penalties."},
    {"query": "Are rental car expenses covered under the policy?", "response": "Yes, rental car expenses may be covered under the policy if your vehicle is undergoing repairs due to a covered claim."},
    {"query": "Can I customize my coverage options?", "response": "Yes, you can customize your coverage options based on your specific needs and preferences."},
    {"query": "Does the policy cover personal belongings left in the vehicle?", "response": "Personal belongings left in the vehicle are typically not covered under the policy. It's advisable to remove valuables from the vehicle."},
    {"query": "What is the process for policy cancellation?", "response": "To cancel your policy, you will need to contact our customer service and submit a cancellation request in writing."},
    {"query": "How are premiums calculated for the policy?", "response": "Premiums are calculated based on factors such as the make and model of your vehicle, your driving history, and the coverage options selected."},
    {"query": "Can I change my coverage options mid-policy term?", "response": "Yes, you can make changes to your coverage options mid-policy term by contacting our customer service and requesting the changes."},
    {"query": "What is the maximum coverage limit for property damage?", "response": "The maximum coverage limit for property damage varies depending on the policy and coverage options selected."},
    {"query": "Does the policy cover rental reimbursement?", "response": "Yes, rental reimbursement coverage may be available as an optional add-on to the policy."},
    {"query": "What is the process for policy renewal?", "response": "Policy renewal involves reviewing your current coverage, making any necessary updates, and paying the renewal premium to continue coverage."},
    {"query": "Are there any discounts available for policyholders?", "response": "Yes, there may be discounts available for policyholders based on factors such as a good driving record, vehicle safety features, and bundled policies."},
    {"query": "How can I update my contact information on the policy?", "response": "You can update your contact information on the policy by logging into your online account or contacting our customer service."},
    {"query": "Is there a waiting period for new policies to take effect?", "response": "New policies typically take effect immediately upon purchase, but specific coverage details may vary."},
    {"query": "Can I make changes to my deductible amount?", "response": "Yes, you can make changes to your deductible amount by contacting our customer service and requesting the change."},
    {"query": "What happens if I miss a premium payment?", "response": "If you miss a premium payment, your policy may lapse, and coverage may be suspended until the payment is made."},
    {"query": "What are the benefits of having comprehensive coverage?", "response": "Comprehensive coverage provides protection against a wide range of risks including theft, fire, vandalism, and natural disasters."},
    {"query": "How do I report an accident?", "response": "To report an accident, contact our claims department immediately and provide all necessary details and documentation."},
    {"query": "Is there a discount for having multiple policies?", "response": "Yes, there is a discount for bundling multiple policies together, such as home and auto insurance."},
    {"query": "What is the policy term for this insurance?", "response": "The policy term is typically one year, after which it can be renewed annually."},
    {"query": "Are there any penalties for early cancellation?", "response": "Yes, early cancellation may incur a penalty fee. It's best to contact our customer service for details."},
    {"query": "Does the policy cover damage caused by natural disasters?", "response": "Yes, the policy covers damage caused by natural disasters such as earthquakes, floods, and storms."},
    {"query": "How can I file a claim?", "response": "To file a claim, contact our claims department by phone or online and provide the necessary details and documentation."},
    {"query": "Are there any exclusions for comprehensive coverage?", "response": "Yes, exclusions for comprehensive coverage include damage caused by intentional acts, racing, and illegal activities."},
    {"query": "Can I add additional coverage options to my policy?", "response": "Yes, you can add additional coverage options such as roadside assistance, rental reimbursement, and personal injury protection."},
    {"query": "What should I do if my vehicle is stolen?", "response": "If your vehicle is stolen, report it to the police immediately and then contact our claims department with the police report and other required documentation."},
    {"query": "How do I update my policy details?", "response": "To update your policy details, log in to your online account or contact our customer service for assistance."},
    {"query": "Does the policy cover damage to third-party property?", "response": "Yes, third-party liability coverage includes damage to third-party property caused by your vehicle."},
    {"query": "What is the process for adding a new driver to my policy?", "response": "To add a new driver to your policy, provide their details to our customer service or update your policy through your online account."},
    {"query": "Are there any benefits for safe driving?", "response": "Yes, policyholders with a safe driving record may be eligible for discounts and lower premiums."},
    {"query": "What types of vehicles are covered under the policy?", "response": "The policy covers a wide range of vehicles including cars, trucks, motorcycles, and recreational vehicles."},
    {"query": "Can I transfer my policy if I sell my vehicle?", "response": "Yes, you can transfer your policy to the new owner of the vehicle by contacting our customer service and completing the necessary paperwork."},
    {"query": "What happens if I get into an accident with an uninsured driver?", "response": "If you get into an accident with an uninsured driver, your uninsured motorist coverage will protect you against losses."},
    {"query": "How do I know if my policy is active?", "response": "You can check the status of your policy by logging into your online account or contacting our customer service."},
    {"query": "What should I do if I lose my insurance card?", "response": "If you lose your insurance card, contact our customer service to request a replacement card."},
    {"query": "Does the policy cover damage from vandalism?", "response": "Yes, the policy covers damage to your vehicle caused by vandalism."}
]

def evaluate_rouge_scores(dataset):
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for data in dataset:
        query = data['query']
        expected_response = data['response']
        # relevant_docs = retrieve(query)
        result = retrievalQA.invoke({"query": query1})

    
        # Calculate ROUGE score
        scores = scorer.score(expected_response, result['result'])
        rouge_scores.append(scores)
    return rouge_scores

rouge_scores = evaluate_rouge_scores(dataset)

# st.write(f'ROUGE SCORES shape is  {len(rouge_scores)}')
# st.write(f'ROUGE SCORES ARE {rouge_scores}')

