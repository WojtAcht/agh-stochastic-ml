## Linear Regression and Gradient Descent
In this laboratory session, you will implement linear regression with gradient descent from scratch. The focus is on understanding the optimization process through hands-on implementation. You will analyze the behavior of the algorithm under different conditions and visualize the results to gain insights into the training dynamics.

### 1. Generate Synthetic Data

Generate a synthetic dataset with a known linear relationship. Create 100 data points with a single feature, using a true weight of 2 and a bias of 5, plus a small amount of Gaussian noise. This will create data that follows the relationship $y=2x+5+\epsilon$ where $\epsilon$ is random noise.

Set a random seed for reproducibility. Your feature matrix X should have shape (100, 1) and your target vector y should have shape (100,).

### 2. Prepare the Data

To include the bias term in our weight vector, add a column of 1s to your feature matrix X. This allows us to represent both the bias and feature weight in a single vector, simplifying our implementation.

### 3. Initialize Parameters

Initialize your weight vector with zeros. Since we have one feature plus the bias term, your weight vector should have shape (2,), where the first element represents the bias and the second represents the feature weight.

### 4. Implement Gradient Descent
Compute the gradient (formula is provided in the introduction) and update weights.

### 5. Run the Training Loop
Put it all together to train the model. Track values of the loss function during this process. 

### 6. Check Your Results
After training, compare your weights to the true ones. Plot the data and compare fitted model with the ground truth line.

### 7. Experiment with Learning Rate
The learning rate α controls how big of a step we take during each iteration of gradient descent. Try training your model with different learning rates: $\alpha \in [10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}]$. For each value:
- Plot the loss curve (loss vs. epoch)
- Record the final weights and final loss
- Observe how quickly (or slowly) the model converges

What happens when the learning rate is too small? Too large? Is there an optimal value in this range?

### 8. Explore Training Duration
Set `num_epochs` to larger values like 100 or 1000 and observe the training process:
- Does the loss continue to decrease throughout training, or does it plateau?
- At what point does the model effectively converge?
- Plot the loss curve for the extended training and identify where diminishing returns begin
- Calculate how close your final weights are to the true weights after different numbers of epochs

This exercise helps you understand when to stop training to avoid wasting computational resources.

### 9. Visualize the Loss Landscape
Create a visualization of the loss function across different weight values:
- Generate a grid of weight values
- Calculate the loss at each point in this grid
- Create a contour plot showing the "landscape" of the loss function
- Mark your initial weights, final weights, and the true weights on this plot

This visualization will help you understand the shape of the objective function you're optimizing and how gradient descent navigates this landscape to find the minimum.

### 10. Analytical Solution
For linear regression, the closed-form solution is 

$$\widehat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

Compare the parameters and final MSE you get via gradient descent with those from the analytical solution. Are they similar?
