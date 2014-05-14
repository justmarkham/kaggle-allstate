## Ideas for improving my Allstate predictive models

As mentioned in my [paper](../allstate-paper.md) in Model Building section 12, I came up with many ideas for how to improve my predictive models that I didn't actually have time to execute. My rough notes are listed below:

1. Rather than coming up with a list of "unlikely" plans and always predicting a replacement plan, selectively predict whether to replace that plan for each customer based upon their characteristics.
2. Rather than always predicting the same replacement plan for a given "unlikely" plan, change which replacement plan to predict based upon the characteristics of each customer.
3. Experiment with adding interaction terms to the models.
4. Use a more sophisticated approach to feature selection (R packages: `relaimpo`, `bestglm`).
5. Use the quote just before the final quote as a set of additional features (A/B/C/D/E/F/G, day, time, cost).
6. Spend more time tuning the randomForest model (variable selection, ntree, mtry).
7. Create an ensemble of predictive models.
8. For predicting which customers will change between the last quote and the purchase point: Rather than using a high probability cutoff and then only predicting new options for that small group of customers, instead pass the "change probability" as a feature to the next model.
9. Truncate the training set to match the test set distribution.
10. Cluster customers into groups, and make predictions separately for each group.
11. Use PCA to discover latent features, and use those features instead of the raw or engineered features.
12. Come up with more ways to incorporate all of the training data (and not just the final quote) into the predictive models.
13. Come up with more ways to make use of the interactions and dependencies between individual options.
