# Linear regression

The two main families of algorithms and predictive machine learning are:

- **Classification**
    - Classification answers questions of the form yes-no. For example, is this email spam or not, or is the patient sick or not.
- **Regression**
    - Regression answers questions of the form how much. For example, how much does this house cost? Or how many seconds do expect someone to watch this video?

At the highest level, the entire topic of linear regression is based around the idea of trying to draw a line (called 'fitting') through an entire dataset of points. The algorithm uses the value of every point in the dataset to find the optimum line equation. Ultimately, the equation of that line can be used to plot new data. There are tricks and techniques for doing this. Sometimes this is perfectly clear and other times it is quite challenging, all based on the type of data in the dataset. Sometimes we want to split the data into different groups, other times we just want a line down the 'middle'. But rarely, if ever, is it perfect, much like data in real life. There are ways to determine how close you are getting, or whether your line is in the best possible place, in the best possible shape, and at the best possible slope. All pages in this lesson are centered around these ideas.

![https://video.udacity-data.com/topher/2021/June/60c92026_linea-regression/linea-regression.png](https://video.udacity-data.com/topher/2021/June/60c92026_linea-regression/linea-regression.png)

Linear Regression

## **Lesson outline**

In this lesson, we will focus on regression. What do you need to know about regression? We will talk about each of the following throughout this lesson.

- Fitting a line through data
- Gradient descent
- Errors
- Linear regression in Scikit-learn
- Polynomial regression
- Regularization
- Feature scaling

## **Learning Objectives**

By the end of the Linear Regression lesson, you should be able to

- Identify the difference between Regression and Classification
- Train a Linear Regression model to predict values
- Predict states using Logistic Regression

## Fitting a line through data:

Here is a refresher on how we move lines by changing the parameters. Assume we have a line represented by the equation $y = w_1x + w_2$, where $w_1$ is a constant and represents the slope and $w_2$ is a constant and represents the y-intercept.

- If we increase $w_1$, we increase the slope so the line rotates counterclockwise.
- If we decrease $w_1$, we decrease the slope so the line rotates clockwise.
- If we increase $w_2$, the line maintains its slope but moves up.
- If we decrease $w_2$, the line maintains its slope but moves down.

## **Absolute Trick**

The absolute trick works by starting with a point and a line.

- A point with coordinates $(p,q)$
- A line represented by $y = w_1x + w_2$.
    
    

To move the line closer to the point $(p,q)$ the absolute trick has two steps:

- add to the y-intercept so that the line moves up
- add to the slope to make the line rotate in the direction of the point

The first possible step is to add 1 to the y-intercept and p*p* to the slope, giving us the equation $y = (w_1 + p) x +( w_2 + 1)$

This ends up being too large of a step, and we have over-corrected our line. Instead, we will utilize a small number called a **learning rate**, referred to as alpha ($\alpha$), to take smaller steps.

We will add ($1\alpha$) to the y-intercept, and ($p\alpha$) to the slope. This gives us the equation

$y = (w_1 +p\alpha )x + (w_2 + \alpha)$

This works when the point (p,q) is above the line, but when the point is underneath the line we need to **subtract** in order to move the line appropriately.

$y = (w_1 -p\alpha )x + (w_2 - \alpha)$

### **Example**

Let's say we have the point $(5,15)$ and the line $y = 2x + 3$ and a learning rate of 0.1.

- Update the slope: We take 5 and multiply it by 0.1 and add to the existing slope, giving us a new slope of 2.5
- Update the y-intercept by adding 0.1, giving us a new y-intercept of 3.1

This means our new equation is $y = 2.5 x + 3.1$

If our point was $(-5,15)$ we would add 0.1 to the y-intercept to move the line upwards, but we update the slope by multiplying -5 by 0.1, or -0.5. This means our new equation is going to be $y = 1.5 x + 3.1.$

> The absolute trick is used extensively in linear regression.
> 

## **Square Trick**

If we have a point that is close to a line, then the distance is small and we want to move the line very little. If the point is far from the line, they want to move the line a lot more. The absolute trick does not take into account how far the point is from the line.

The square trick addresses this.

Let's look at this vertical distance between the point and the line. The point over the line has coordinates (p,q)(*p*,*q*) and the corresponding point on the line is $(p,q')$

The distance between the point and the line is $(q-q')$

We take this distance and multiply it into what we add to both the y-intercept and to the slope.

- Update the y-intercept by adding $\alpha(q-q')$
- Update the slope by adding $p\alpha(q-q')$

This gives us the equation

$y = (w_1 +p(q-q')\alpha )x + (w_2 + (q-q')\alpha)$

This trick automatically takes care of points that are under the line and we don't need two rules as we had on the absolute trick. We just have the same rule for both.

### **Example**

Let's say we have the point $(5,15)$ and the line $y = 2x + 3$ (which gives us $q' = 13$) and a learning rate of 0.01. Notice that this learning rate is smaller than the example that we used for the Absolute Trick.

- Update the slope: We take 5 and multiply it by 0.01, and then multiply this value by 2 before adding the result to the existing slope, giving us a new slope of **2.1**
- Update the y-intercept by adding 0.01*2, giving us a new y-intercept of **3.02**

This means our new equation is $y = 2.1 x + 3.02$

That's the square trick.

## **Gradient Descent**

The tricks we just learned still seem a little too magical, and we'd like to find their origin.

Let's say we have our points on our plan is to develop an algorithm that will find the line that best fits this set of points. And the algorithm works like this,

First, draw a random line, and calculate the error. The error is some measure of how far the points are from the line, in this drawing, it looks like it's the sum of these distances but it could be any measure that tells us how far we are from the points.

Now we're going to move the line around and see if we can decrease this error.

- We move in this direction and we see that the error kind of increases so that's not the way to go.
- We move in the other direction and see that the error decreased, so we pick this one and stay there.

Repeat these steps many times over and over every time descending the error a bit until we get to the perfect line.

To minimize this error, we're going to use something called **gradient descent**.

### **Descending from Mt. RainiError**

To build your intuition of what gradient descent is, imagine we're standing on top of a mountain called Mt rainiError as it measures how big our error is, and in order to descend from this mountain, we need to minimize our height.

On the left, we have a problem fitting the line to the data, which we can do by minimizing the error or the distance from the line to the points. Descending from the mountain is equivalent to getting the line closer to the points.

If we want to descend from the mountain we would look at the directions where we can walk down and find the one that makes us descend the most. Bit by bit, we descend a bit in this direction, this is equivalent to getting the line a little bit closer to the points.

So now our height is smaller because we're closer to the points since our distance to them is smaller. Again and again, we look at what makes us descend the most from the mountain.

Now we're at a point where we're descended from the mountain and on the right, we found the line that is very close to our points. Thus, we've solved our problem and that is gradient descent.

### **The mathematical definition**

Consider the plot in two dimensions allowing a reality that the plot will be in higher dimensions. We have our weights on the x-axis and our error on the y-axis. And we have an error function. The way to descend is to actually take the derivative or gradient of the error function with respect to the weights. This gradient is going to point to a direction where the function increases the most. Therefore, the negative of this gradient is going to point down in the direction where the function decreases the most. So what we do is we take a step in the direction of the negative of that gradient, this means we are taking our weights wi and changing them to wi minus the derivative of the error with respect to wi.

In real life, we'll be multiplying this derivative by the learning rate since we want to make small steps. This means the error function is decreasing and we're closer to the minimum. If we do this several times we get to either a minimum or a pretty good value where the error is small. Once we get to the point that we've reached a pretty good solution for our linear regression problem.

The two most common error functions for linear regression are:

- Mean absolute error
- Mean squared error

### **Mean Absolute Error**

Assume we have a point with coordinates $(x,y)$ and the line is called $\hat{Y}$ since it is our prediction. The corresponding point on the line is $(x, \hat{y})$, and the vertical distance from the point to the line is $(y-\hat{y})$. This is the **error**.

Our total error is going to be the sum of all these distances for all the points in our dataset.

$Error = \sum_{i=1}^{m} |y-\hat{y}|$

In some cases, we'll use the average or the mean absolute error, which is the sum of all the errors divided by m, the number of points in our dataset.

$Error = {\frac 1 m}\sum_{i=1}^{m} |y-\hat{y}|$

Using the sum or the average won't change our algorithms, since that would only scale our error by a constant, namely m.

Notice that we have an absolute value around $y-\hat{y}$, the reason is that if the point is on top of the line, the distance is $y-\hat{y}$, but if it's under the line then it's $\hat{y}-y$. We want the error to always be positive, otherwise negative errors will cancel with positive errors.

The graph has more dimensions but this is a two-dimensional simplification of that graph. As we descend in this graph using gradient descent, we get a better and better line until we find the best possible fit with the smallest possible mean absolute error.

## **Mean Squared Error**

The Mean Squared Error is very similar to the Mean Absolute Error.

Assume we still have our point and our prediction, but instead of taking the distance between the point and the prediction, we're going to draw a square with this segment as its side. This area is $(y - \hat{y})^2$

Notice that this is always non-negative, so we don't need to worry about absolute values.

Our mean squared error is going to be the average of all these series of squares.

$Error = {\frac 1 {2m}}\sum_{i=1}^{m} (y-\hat{y})^2$

If the point is over the line or underneath the line the square is always going to be a non-negative number because the square of a real number is *always going to be non-negative*.

The one half is going to be there for convenience because later we'll be taking the derivative of this error.

## **2 ways to fit a line**

We've learned two algorithms that will fit a line through a set of points.

- Using the absolute or square trick
- Minimizing any of the error functions namely the mean absolute error and the mean squared error.

`The interesting thing is that these two are actually the exact same thing!`

When we minimize the mean absolute error, we're using a gradient descent step and that gradient descent step is the exact same thing as the absolute trick. Likewise, when we minimize the squared error, the gradient descent step is the exact same thing as the square trick.

---

## **Mean squared error**

Let's start with the mean squared error. Assume we have a point with coordinates (x, y), and a line with the equation $\hat{y}= w_1 x + w_2$. So $\hat{y}$ is our prediction, and it predicts the y coordinate of the point that matches the x coordinate of the point $(x,y)$, the point $(x,\hat{y})$. The error for this point is ${\frac 1 2}( y - \hat{y})^2$, and the mean squared error for this set of points is the average of all these errors. Since the average is a linear function, whatever we do here applies to the entire error.

We know that the gradient descent step uses these two derivatives, namely the derivative with respect to the slope w_1*w*1 and the derivative with respect to the y-intercept $w_2$. If we calculate the derivatives, we get $\frac{\partial}{\partial w_1} Error = -(y-\hat{y})x$ for the one respect to the slope and $\frac{\partial}{\partial w_2} Error = -(y-\hat{y})$ for the one with respect to the y-intercept $w_2$.

Notice that the length of this red segment is precisely $(y - \hat{y}$) and the length of this green segment is precisely x.

## **Square trick**

If you remember correctly, the square trick told us that we have to update the slope by adding $(y - \hat{y}) x \alpha$, and updating the y-intercept by adding $(y - \hat{y}) \alpha$. But that is precisely what this gradient descent step is doing.

Think to yourself about how this is the exact same calculation.

So this is why the gradient descent step utilize when we minimize the mean squared error is the same as the square trick.

## **Absolute trick**

We can do the same thing with the absolute trick. The procedure is very similar except we have to be careful about the sign. Our error is $|y -\hat{y}|$ and the derivatives of the error with respect to $w_1$ and $w_2$ are $\pm{x} and \pm{1}$ based on if the point is on top of or underneath the line. Since the distance is x, then you can also check that this is precisely what the gradient descent step does when we minimize the mean absolute error.

That is why minimizing these errors with gradient descent is the exact same thing that using the absolute and the square tricks.

# **Development of the derivative of the error function**

Notice that we've defined the squared error to be

$Error = \frac{1}{2} (y - \hat{y})^2.$

Also, we've defined the prediction to be

$\hat{y} = w_1 x + w_2.$

So to calculate the derivative of the Error with respect to $w_1$ , we simply use the chain rule:

$\frac{\partial}{\partial w_1} Error = \frac{\partial Error}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial w_i}.$

The first factor of the right-hand side is the derivative of the Error with respect to the prediction $\hat{y}$, which is $-(y-\hat{y})$.

The second factor is the derivative of the prediction with respect to $w_1$, which is simply *x*.

Therefore, the derivative is

$\frac{\partial}{\partial w_1} Error = -(y-\hat{y})x$
