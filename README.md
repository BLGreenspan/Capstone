# Predicting Future Food Inspection Scores in Dallas, TX
#### Brandon Greenspan, Data Scientist [GitHub](https://github.com/BLGreenspan), [LinkedIn](https://www.linkedin.com/in/brandonlgreenspan/)

## Problem Statement
[An article in the Dallas Observer](https://www.dallasobserver.com/restaurants/dallas-restaurant-inspections-suffer-from-delays-poor-record-keeping-and-overworked-staff-10697588) unearthed a massive problem in the city's ability to follow up on restaurants requiring reinspection due to a low grade upon original inspection.  Dallas states that out of a scale from 1-100, any facility that scores between 70-79 requires reinspection within 30 days, between 60-69 requires reinspection within 10 days, and below 60 requires reinspection ASAP.

The article points out many flaws in the city's ability to reinspect restaurants within its own self-imposed timeframes.  Until the department can hopefully become better-staffed, I am looking to build a classification model that can predict how a restaurant will perform upon reinspection.  The classification model will take in inspection data, both text and numerical, and utilize NLP to predict the future inspection score based on the most recent inspection data.  This way, if the city is still struggling to reinspect restaurants in a timely manner, they can refer to the model in order to prioritize certain facilities to reinspect.

The target metric for my model is to improve upon the baseline, which would be an accuracy score of greater than __60.38%__.


## Executive Summary
According to their self-imposed food inspection timeline, the city of Dallas is having trouble adhering to said timeline.  A restuarant should not go more than 180 days without inspection if they scored above 80 on their previous inspection, and in a few cases, we found instances of restaurants going over 800 days (that's over two years!) between inspections.  This is more troublesome when restaurants score below 80, since a score in the 70's requires reinspection within 30 days, a score in the 60's requires reinspection within 10 days, and scores below 60 require reinspection ASAP in order to continue operating.

Our problem statement frames that we are using previous data and NLP in order to predict future inspection scores.  This meant that as we clean our data, we have to engineer a feature as our target column, which would be the follow-up inspection score.  While we are not treating this as a time series problem, we do need to shift our data, which does mean we need to drop every restaurant's most recent inspection score in order to build our model.

Through EDA, we start with the bad and look at restaurants that performed poorly on inspections, were not inspected within the given timeframe, and then performed poorly again.  We do become a bit more optimistic when we discover that about 94% of restaurants are able to score an 80 or above.  We then investigate possible correlations between certain features and future inspection scores.  Before modeling, we do need to send our text data through Countvectorizer, but need to limit our models to 2100 features (instead of 800,000).

After running a baseline model, two Logistic Regression models, Multinomial Naive Bayes, Decision Tree, and a Neural Network, are we able to select a model that beats our baseline model score of 60.38?  Regardless, can we investigate our coefficients in a manner that could help for future predictability, especially if we can run this again with more features?


## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|**Restaurant Name**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Name of Restaurant| 
|**Inspection Type**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Code indicating the inspection type, such as Routine, Follow-up, Complaint, Temporary and Mobile.|
|**Inspection Date**|*datetime*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Date the inspection for the facility was performed.| 
|**Inspection Score**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The aggregate score from the inspection violations. Please note not every violation will reflect a point deduction as establishments are allowed to correct violations during the inspection process, and therefore no reduction in the overall score is reflected for the violation.| 
|**Street Number**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Street number for the address of the facility.|
|**Street Name**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Street name for the address of the facility.| 
|**Street Direction**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Street direction for the address of the facility. For example, N, W, S, etc.|
|**Street Type**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Street type for the address of the facility. For example, AVE, LN, ST, etc.| 
|**Street Unit**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Unit number or apartment number for the address of the facility.| 
|**Street Address**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Full street address of the facility.| 
|**Zip Code**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Zip Code of the facility's address.| 
|**Violation Points - 1**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 1**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 1**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 2**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 2**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 2**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 3**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 3**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 3**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 4**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 4**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 4**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 5**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 5**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 5**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 6**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 6**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 6**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 7**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 7**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 7**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 8**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 8**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 8**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 9**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 9**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 9**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 10**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 10**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 10**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 11**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 11**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 11**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 12**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 12**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 12**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 13**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 13**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 13**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 14**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 14**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 14**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 15**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 15**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 15**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 16**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 16**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 16**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 17**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 17**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 17**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 18**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 18**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 18**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 19**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 19**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 19**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 20**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 20**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 20**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 21**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 21**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 21**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 22**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 22**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 22**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 23**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 23**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 23**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 24**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 24**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 24**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Violation Points - 25**|*integer*|Restaurant and Food Establishment Inspections (October 2016 to Present)|The amount of points assigned to this violation.| 
|**Violation Detail - 25**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Field used to describe the type of violation associated with the enforcement action.|
|**Violation Memo - 25**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|This field is used for any additional comments about the enforcement action.| 
|**Inspection Month**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Added column to improve visualization of inspections by month| 
|**Inspection Year**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Added column to improve visualization of inspections for specific fiscal years| 
|**Lat Long Location**|*string*|Restaurant and Food Establishment Inspections (October 2016 to Present)|Denotes a location point on a longitude line (perpendicular to the equator) and latitude line (parallel to the equator)| 


## Conclusion/Recommendations
Our Logistic Regression model with a Ridge penalty and usage of TFIDF was able to predict the next inspection grade of a restaurant with 69% accuracy, which did exceed our goal of defeating the baseline of 60.38% accuracy.

It is worth taking note that while our model did predict more accurately than the baseline, unbalanced classes likely played a role.  With almost 95% of our data falling in the A or B range, our model did not even give one prediction as a D or F.  While the occurrence of a D or F is rare, the model appears completely incapable of predicting that value, which could be catastrophic if the model cannot predict a restaurant that will ultimately fail a health inspection.

While keeping in mind that B's incorrectly predicted as A's and A's incorrectly predicted as B's technically result in the same outcome, any prediction of C should be a cause for alarm, since 2/3 of F's and 0 A's were predicted as C's.

As for suggestions for the city of Dallas, if the staffing situation cannot improve, we should look to decrease the cadence of inpections for restaurants that have never scored below an A after two years of inspections under the same management.  My data is not able to determine if places that score that highly do so because of maintained obedience of guidelines or if they prioritize health & safety only when they expect an inspection.  While we hope the former is the case, we don't have the capacity of staff to fully investigate if the latter is true.

## Next Steps
The biggest drawback to this model is that we had to put a cap on max features when running CountVectorizer.  Without a cap, we end up with about 800,000 columns.  By using the max features hyperparameter in CountVectorizer, we likely excluded meaningful words or phrases that could have helped further in predicting future inspection scores, possibly identifying certain features than may have been able to predict a D or F score.  

In order to get the best model, we need more budget and time to utilize AWS or any cloud computing service.


## References
- [Restaurant and Food Establishment Inspections (October 2016 to Present)](https://www.dallasopendata.com/City-Services/Restaurant-and-Food-Establishment-Inspections-Octo/dri5-wcct/data)
- [Failing Grade: Dallas Policies Protect the City’s Filthiest Restaurants from Health Inspectors](https://www.dallasobserver.com/restaurants/dallas-restaurant-inspections-suffer-from-delays-poor-record-keeping-and-overworked-staff-10697588?fbclid=IwAR1mRlqMu9cOYesZXcTHJt56Vds1LNJpwkNVHyGagjLFJAX0eIe9ONNXSh8)



```python

```