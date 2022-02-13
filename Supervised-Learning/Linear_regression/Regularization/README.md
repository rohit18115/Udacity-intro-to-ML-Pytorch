# **Regularization Exercise**

Perhaps it's not too surprising at this point, but there are classes in sklearn that will help you perform regularization with your linear regression. You'll get practice with implementing that in this exercise. In this assignment's data.csv, you'll find data for a bunch of points including six predictor variables and one outcome variable. Use sklearn's `[Lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)` class to fit a linear regression model to the data, while also using L1 regularization to control for model complexity.

### **Perform the following steps:**

**1. Load in the data**

- The data is in the file called 'data.csv'. Note that there's no header row on this file.
- Split the data so that the six predictor features (first six columns) are stored in `X`, and the outcome feature (last column) is stored in `y`.

**2. Fit data using linear regression with Lasso regularization**

- Create an instance of sklearn's `[Lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)` class and assign it to the variable `lasso_reg`. You don't need to set any parameter values: use the default values for the quiz.
- Use the `Lasso` object's `.fit()` method to fit the regression model onto the data.

**3. Inspect the coefficients of the regression model**

- Obtain the coefficients of the fit regression model using the `.coef_` attribute of the `Lasso` object. Store this in the `reg_coef` variable: the coefficients will be printed out, and you will use your observations to answer the question at the bottom of the page.
