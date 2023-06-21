from pymongo import MongoClient
from sklearn.linear_model import LinearRegression

# Connect to the MongoDB server
client = MongoClient('<connection_string>')  # Replace <connection_string> with your actual connection string

# Access the desired database and collection
db = client['your_database']  # Replace 'your_database' with the name of your database
collection = db['your_collection']  # Replace 'your_collection' with the name of your collection

# Fetch data from MongoDB collection
data = []
target = []
for document in collection.find():
    data.append(document['feature'])
    target.append(document['target'])

# Create and train the linear regression model
model = LinearRegression()
model.fit(data, target)

# Generate predictions
new_data = [[1.5], [2.0], [2.5]]  # Example input data
predictions = model.predict(new_data)

# Print the predictions
print("Predictions:")
for i in range(len(new_data)):
    print(f"Input: {new_data[i]}, Prediction: {predictions[i]}")
