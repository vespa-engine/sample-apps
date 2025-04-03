
from vespa.application import Vespa
from vespa.io import VespaResponse, VespaQueryResponse

import random
import requests
import time
import sys
import os

import streamlit as st

OPENAI_API_KEY = st.secrets["api_keys"]["llm"]
TAVILY_API_KEY = st.secrets["api_keys"]["tavily"]
VESPA_URL = st.secrets["vespa"]["url"]
PUBLIC_CERT_PATH = st.secrets["vespa"]["public_cert_path"]
PRIVATE_KEY_PATH = st.secrets["vespa"]["private_key_path"]


# Fetch documents from Vespa
def fetch_documents(num_results=100):
    

    myyql=f"select id, quantity from product where true limit {num_results}"
    print(myyql)

    with vespa_app.syncio(connections=1) as session:
        response: VespaQueryResponse = session.query(
            yql= myyql,
            hits=num_results,
        )
    assert response.is_successful()

    # Extract only the 'fields' content from each entry
    filtered_data = [hit["fields"] for hit in response.hits]

    return filtered_data

# Update document quantity in Vespa
def update_document_quantity(doc_id, new_quantity):

    response = vespa_app.update_data(
        schema="product",
        data_id=doc_id,
        fields={
            "quantity": {"assign": new_quantity}
        },
        auto_assign=False
    )
    print("OK, update should be done for doc_id: " + str(doc_id))
    print(response.json)

    if response.is_successful():
            #print(f"Successfully updated document {doc_id}: quantity set to {new_quantity}")
            return True  # Return True on success
    else:
            print(f"Failed to update document {doc_id}: {response.get_json()}")
            return False  # Return False on failure

# Run the update process in a loop every second
if __name__ == "__main__":

    vespa_app = Vespa(url=VESPA_URL,
                  cert=PUBLIC_CERT_PATH,
                  key=PRIVATE_KEY_PATH)
    
    while True:
        print("Fetching and updating random document quantities...")
        
        # Fetch documents from Vespa
        documents = fetch_documents(num_results=40)

        # Ensure there are documents before proceeding
        if documents:
            # Select a random subset of documents to update
            num_to_update = random.randint(1, len(documents) // 2)  # Random subset
            documents_to_update = random.sample(documents, num_to_update)
            print(documents_to_update)

        for doc in documents_to_update:
                doc_id = doc["id"]
                current_quantity = doc["quantity"]

                # Randomly decide to increase or decrease quantity
                if current_quantity == 0:
                    new_quantity = 5  # If quantity is 0, set it to 5
                else:
                    change = random.choice([-1, 1])  # Randomly increase or decrease
                    new_quantity = max(0, current_quantity + change)  # Ensure it doesn't go negative    
                
                print(f"Updating document {doc_id}: {current_quantity} → {new_quantity}")
                update_document_quantity(doc_id, new_quantity)
                              
        print("Sleeping for 3 second before next update...")
        time.sleep(3)  # Pause execution for 1 second before next iteration