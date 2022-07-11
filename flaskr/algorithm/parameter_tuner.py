param_grid={
    'n_estimators': [200, 500], 
    'max_features': ['auto', 'sqrt', 'log2'], 
    'max_depth': [4, 5, 6, 7, 8], 
    'criterion': ['gini', 'entropy']
}  

grid = GridSearchCV(estimator=clf, param_grid=param_grid, refit = True, verbose = 3,n_jobs=-1) 

# fitting the model for grid search 
grid.fit(X_train, y_train) 

# print best parameter after tuning 
print(grid.best_params_) 
grid_predictions = grid.predict(X_test) 
print(classification_report(y_test, grid_predictions))
# print classification report 