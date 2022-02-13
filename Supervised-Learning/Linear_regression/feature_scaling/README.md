### **Feature Scaling Exercise**

Previously, you saw how regularization will remove features from a model (by setting their coefficients to zero) if the penalty for removing them is small. In this exercise, you'll revisit the same dataset as before and see how scaling the features changes which features are favored in a regularization step. See the "Quiz: Regularization" page for more details. The only thing different for this quiz compared to the previous one is the addition of a new step after loading the data, where you will use sklearn's `[StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)` to standardize the data before you fit a linear regression model to the data with L1 (Lasso) regularization.

### **Perform the following steps:**

**1. Load in the data**

- The data is in the file called 'data.csv'. Note that there's no header row on this file.
- Split the data so that the six predictor features (first six columns) are stored in `X`, and the outcome feature (last column) is stored in `y`.

**2. (NEW) Perform feature scaling on data via standardization**

- Create an instance of sklearn's `[StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)` and assign it to the variable `scaler`.
- Compute the scaling parameters by using the `.fit_transform()` method on the predictor feature array, which also returns the predictor variables in their standardized values. Store those standardized values in `X_scaled`.

**3. Fit data using linear regression with Lasso regularization**

- Create an instance of sklearn's `[Lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)` class and assign it to the variable `lasso_reg`. You don't need to set any parameter values: use the default values for the quiz.
- Use the `Lasso` object's `.fit()` method to fit the regression model onto the data. Make sure that you apply the fit to the *standardized* data from the previous step (`X_scaled`), not the original data.

**4. Inspect the coefficients of the regression model**

- Obtain the coefficients of the fit regression model using the `.coef_` attribute of the `Lasso` object. Store this in the `reg_coef` variable: the coefficients will be printed out, and you will use your observations to answer the question at the bottom of the page.
