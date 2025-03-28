import os
from dotenv import load_dotenv

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

mongo_key = os.getenv("MONGO_KEY")  
uri = mongo_key



# Function to write the file to database
def write(uniprot_data):
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client['shiru']
    collection = db['shiru']

    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    result = collection.insert_one(uniprot_data)
    print(f"Inserted document Ids:{result.inserted_ids}")

    client.close()
    