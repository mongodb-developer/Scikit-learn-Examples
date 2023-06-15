from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Connect to MongoDB Atlas
client = MongoClient('<your_connection_string>')

# Access the movie collection
db = client['sample_mflix']
movie_collection = db['movies']

# Retrieve the relevant dataset
movie_data = movie_collection.find({}, {
   'genre': 1,
   'director': 1,
   'actors': 1,
   'release_year': 1,
   'user_ratings': 1
})

# Preprocess the data (example: handling missing values)
for movie in movie_data:
   if 'user_ratings' not in movie:
       movie['user_ratings'] = 0.0

# Split the data into training and testing sets
X = [movie['genre'] + movie['director'] + movie['actors'] + [movie['release_year']] for movie in movie_data]
y = [movie['user_ratings'] for movie in movie_data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the model
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
report = classification_report(y_test, y_pred)
print(report)
