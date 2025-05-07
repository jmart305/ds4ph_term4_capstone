# Term 4 Capstone 
**Cassie Chou & Julia Martin**

**Note: data can be downloaded [here](https://bmore-open-data-baltimore.hub.arcgis.com/datasets/911-calls-for-service-2024-1/explore)**  

**[Link to presentation slides with screencast of app](https://docs.google.com/presentation/d/1ivnFPWpoD4gnpIj2G3Phzb8QEnDSExPduENvzyIbsMg/edit#slide=id.g354ba850da6_0_1882)**

The goal of this project is to understand where and when 911 calls are being made in Baltimore and where and when high priority calls are being made so that that the appropriate emergency response teams can be prepared to respond to calls. Using data from the [Open Baltimore API](https://bmore-open-data-baltimore.hub.arcgis.com/datasets/911-calls-for-service-2024-1/explore) on 1.6 million calls throughout 2024, we began by analyzing important features of 911 calls including date, time, location, description of emergency, and priority of call. Calls are recorded in the data as having one of the following priority levels: Out-of-Service, Non-Emergency, Low, Medium, High, or Emergency. After a descriptive analysis, we explored using neural networks to predict call priority.

First, we looked at the raw distribution of call priority and found that about 2/3 of calls overall are labeled as Non-Emergency, and fewer labeled as Low, Medium, High, or Emergency priority. To visualize potential spatial trends, we organized the call data into 55 Community Statistical Areas (CSA), which are clusters of neighborhoods in Baltimore organized around Census tract boundaries, and compared average call priority among each CSA. We also looked at average call priority in each hour of the day and month of the year. We found evidence of possible spatial and temporal trends related to call priority, which supported our plan to use these variables as predictors for our model. 

Next, we began fitting neural networks to predict call priority, with the goal of helping 911 operators prioritize calls that may be urgent vs. non-emergent (which make up around 70% of calls) and informing recommendations for redirecting resources to certain areas at certain times we would expect to see more urgent calls to hopefully improve response to emergencies in Baltimore. 

Our first neural network predicted priority as one of the following categories: Non-emergency, Low, Medium, High, Emergency, or Out-of-Service using CSA, month, date, time, and day of week as predictors. This model had 2 layers (256 and 128 nodes) and used a focal loss function. To account for the low prevalence of some of the priority categories, we weighted the training data to upweight the categories with low prevalence in the data. This model had a 45% accuracy rate which is an improvement from guessing one of the six categories as random but is not particularly strong. 

For our second neural network, we added call description as a predictor, using the top 50 most frequently occurring descriptions as categories and the remaining descriptions grouped into an "other" category. This model had a 93% accuracy rate overall, but only a 15% accuracy rate for the "other" category which is worse than just guessing a category at random. 

For our last model, we decided to make the outcome a binary variable, either non-emergency/out-of-service or low/medium/high/emergency priority to distinguish urgent and non-urgent calls. We used the same predictors as in our first model. We fit a 2 layer neural network (128 nodes, 64 nodes) with a cross entropy loss function and again used weighting to upweight emergent calls in the data. This model had 67% accuracy in the testing set which is a slight improvement over randomly guessing.

After fitting these models, we learned that, as expected, call description is a powerful predictor, but a model with call description as a predictor is no more useful than a human matching call description to a priority level. Without the description, predicting both a specific priority level or a binary outcome is difficult for a neural network, and this task is specifically difficult due to the extremely large non-emergent call volume for issues like noise complaints or requests to patrol an area. In the future, we could try different models such as logistic regression or random forest, or use other variables in the dataset as precitors such as police precinct and address of the call. To conclude, no model will be perfect and the inherently unpredictable nature of emergengies makes it difficult for models to forecast. 

See our streamlit app which includes a prediction tool!





Limitations: overfit to 2024 regarding month variable
