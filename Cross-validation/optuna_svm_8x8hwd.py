import optuna
from time import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load data sets
data = datasets.load_digits()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y)

# Examine the shape of the data sets
print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)

# Set the search range of candidate models
def create_model(trial):
    model_type = trial.suggest_categorical('model_type', ['Logistic-regression', 'SVM'])

    if model_type == 'SVM':
        kernel = trial.suggest_categorical('SVM-kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        C = trial.suggest_float('SVM-C', 0.1, 1000, log=True)
        gamma = trial.suggest_float('SVM-gamma', 0.0001, 10, log=True)
        model = SVC(kernel=kernel, C=C, gamma=gamma)

    if model_type == 'Logistic-regression':
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
        if penalty == 'l1':
            solver = 'saga'
        else:
            solver = 'lbfgs'
        regularization = trial.suggest_float('Logistic-regularization', 
                                             0.01, 100, log=True)
        model = LogisticRegression(penalty=penalty, 
                                   C=regularization, 
                                   solver=solver)

    if trial.should_prune():
            raise optuna.TrialPruned()

    return model

# Define the objective func. as the 5-fold cross-validation score
def objective(trial):
    model = create_model(trial)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return np.mean(scores)

start = time()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
print("\n\nBest parameters: ", best_params)

best_model = create_model(study.best_trial)
best_model.fit(X_train, y_train)
scores = cross_val_score(best_model, X_train, y_train, cv=5) 
print("Best cross-validation score: %.5f" % np.mean(scores))
y_pred = best_model.predict(X_test)
print("Performance on test data: %.5f" % accuracy_score(y_test,y_pred))
total_time = time()-start
print("Runtime: %.4f sec" % total_time)

cfm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cfm, 
                                    display_labels = np.arange(0,10))
cm_display.plot()
plt.title('Test Data: Confusion Matrix')
plt.show()
