import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('../dataset/iris.csv')
df = pd.DataFrame(data)
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of KNN Classifier: {accuracy * 100:.2f}%')
predicted_class = knn.predict(new_sample)
predicted_class_name = label_encoder.inverse_transform(predicted_class)
print(f'Predicted variety for the new sample: {predicted_class_name[0]}')