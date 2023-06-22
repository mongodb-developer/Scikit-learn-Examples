from pymongo import MongoClient
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Plot a Seaborn visualization
sns.set(style="ticks")
sns.pairplot(df, hue="label")  # Replace "label" with the column name containing the labels in your data
plt.show()
