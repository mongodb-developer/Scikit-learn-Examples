from pymongo import MongoClient
import pandas as pd

# Connect to the MongoDB server
client = MongoClient('<connection_string>')  # Replace <connection_string> with your actual connection string

# Access the desired database and collection
db = client['your_database']  # Replace 'your_database' with the name of your database
collection = db['your_collection']  # Replace 'your_collection' with the name of your collection

# Fetch data from MongoDB collection
data = []
for document in collection.find():
    data.append(document)

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Perform data analysis with pandas
print("Summary Statistics:")
print(df.describe())
print("\nData Head:")
print(df.head())
