<b>CDOT Project Estimator</b>
===========================

In partnership with:
![Cross Validation Process](images/cdot_logo.png)


#### Project Objective:
<b>Use historical data to develop a method that can be used to accurately predict constructions costs for upcoming CDOT projects.</b>

#### Background:
Colorado Department of Transportation
(CDOT) construction projects are under-
estimated by 15.15% when compared with
proposals from the winning contractor.
When estimating the cost of a construction
project, there are many factors to consider.
Factors include: understanding the
potential for missed scope (change orders),
taking into account fluctuating construction
material rates, historical costs and work
histories of companies submitting
proposals, etc.
Current estimating processes rely heavily on
estimator knowledge and can be considered
just as much art as science. While this can be
an effective method of predicting
construction costs, it becomes highly
subjective and difficult to reciprocate. The
aim of this project is to simplify and
streamline the estimating process using
machine learning.
Data received from CDOT includes cost and
estimator information on roughly 1,400
projects. The predictor used to develop
these models is based off project unit
quantities as they relate to CDOT’s over
8,000 specific bid items.

#### Process

Cross Validation was used to create the model. The final model includes a stacked ExtraTreeRegressor, and GradientBoostingRegressor modules from sklearn.

![Cross Validation Process](images/cross_val.jpg)

### Final Model Performance

The summary for the three model's performance on the test set is shown below.

![MSE Table](images/results.jpg)


![Model 1](images/1.png)


![Model 2](images/2.png)


![Model 3](images/3.png)
