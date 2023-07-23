import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report


dataset = pd.read_csv('CICID2017.csv')

X = dataset.drop('label', axis=1)
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Create the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Create the Grid Search object
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_rf_params = grid_search.best_params_
print("Best Random Forest hyperparameters:", best_rf_params)


grid_means = grid_search.cv_results_['mean_test_score']
grid_params = grid_search.cv_results_['params']
plt.figure(figsize=(10, 6))
plt.plot(grid_means, marker='o')
plt.title("Grid Search for Random Forest")
plt.xlabel("Hyperparameter Combinations")
plt.ylabel("Mean Test Score")
plt.xticks(np.arange(len(grid_params)), grid_params, rotation=45, ha='right')
plt.tight_layout()
plt.show()
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
