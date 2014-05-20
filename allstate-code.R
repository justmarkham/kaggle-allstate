### Kaggle Competition: Allstate Purchase Prediction Challenge
### Code by Kevin Markham
### https://github.com/justmarkham/kaggle-allstate

# Notes about this file:
# 1. Most of the code is explained in my paper:
#    https://github.com/justmarkham/kaggle-allstate/blob/master/allstate-paper.md
# 2. For the sake of brevity, a lot of my exploratory code is not included.
# 3. Many of the variables are poorly named, my apologies.
# 4. The code assumes that "train.csv" and "test_v2.csv" are in your working
#    directory.
# 5. To generate my best set of predictions, simply load the required libraries
#    and then run the "Loading Data" section and "Model Building Part 10". All
#    other code is irrelevant.


# Load required libraries
library(dplyr)
library(caret)
library(ROCR)
library(randomForest)
library(nnet)
library(ggplot2)


## LOADING DATA AND ADDING FEATURES ##

# Define column classes for reading in the data sets
colClasses <- c(rep("integer", 4), "character", rep("factor", 2),
    "integer", "factor", "integer", "factor", rep("integer", 3),
    rep("factor", 2), "integer", rep("factor", 7), "integer")

# Function for pre-processing and feature engineering
preprocess <- function(data) {

    # add features
    data$plan <- paste0(data$A, data$B, data$C, data$D, data$E, data$F, data$G)
    data$hour <- as.integer(substr(data$time, 1, 2))
    data$timeofday <- as.factor(ifelse(data$hour %in% 6:15, "day",
                                ifelse(data$hour %in% 16:18, "evening",
                                    "night")))
    data$weekend <- as.factor(ifelse(data$day %in% 0:4, "No", "Yes"))
    data$family <- as.factor(ifelse(data$group_size > 2 & data$age_youngest <
                                    25 & data$married_couple==1, "Yes", "No"))
    data$agediff <- data$age_oldest-data$age_youngest
    data$individual <- as.factor(ifelse(data$agediff==0 & data$group_size==1,
                                        "Yes", "No"))
    data$stategroup <- as.factor(ifelse(data$state %in% c("SD","ND"), "g1",
                                 ifelse(data$state %in% c("AL","WY"), "g2",
                                 ifelse(data$state %in% c("OK","ME","AR",
                                     "WV"), "g3",
                                 ifelse(data$state %in% c("DC","NE","GA"),
                                     "g5", "g4")))))
    
    # fix NA's for duration_previous and C_previous
    data$duration_previous[is.na(data$duration_previous)] <- 0
    levels(data$C_previous) <- c("1", "2", "3", "4", "none")
    data$C_previous[is.na(data$C_previous)] <- "none"
    
    # replace risk_factor NA's by predicting a value
    datanorisk <- data[is.na(data$risk_factor), ]
    datarisk <- data[!is.na(data$risk_factor), ]
    lm.fit <- lm(risk_factor ~ age_youngest*group_size+married_couple+
        homeowner, data=datarisk)
    lm.pred <- predict(lm.fit, newdata=datanorisk)
    data$risk_factor[is.na(data$risk_factor)] <- round(lm.pred, 0)
    
    # for car_age greater than 30, "round down" to 30
    data$car_age[data$car_age > 30] <- 30
    
    return(data)
}

# read in training set
train <- read.csv("train.csv", colClasses=colClasses)
train <- preprocess(train)

# trainsub is subset of train that only includes purchases
trainsub <- train[!duplicated(train$customer_ID, fromLast=TRUE), ]

# trainex is subset of train that excludes purchases
trainex <- train[duplicated(train$customer_ID, fromLast=TRUE), ]

# trainex2 only includes last quote before purchase
trainex2 <- trainex[!duplicated(trainex$customer_ID, fromLast=TRUE), ]

# changed is anyone who changed from their last quote
changed <- ifelse(trainsub$plan == trainex2$plan, "No", "Yes")
changelog <- ifelse(trainsub$plan == trainex2$plan, FALSE, TRUE)
trainsub$changed <- as.factor(changed)
trainex2$changed <- as.factor(changed)
trainsub$changelog <- changelog
trainex2$changelog <- changelog

# compute "stability" feature from trainex and add to trainex2
customerstability <- trainex %.% group_by(customer_ID) %.%
    summarise(quotes=n(), uniqueplans=n_distinct(plan),
    stability=(quotes-uniqueplans+1)/quotes)
trainex2$stability <- customerstability$stability

# compute "planfreq" feature on trainex2
nrowtrainex2 <- nrow(trainex2)
planfreqs <- trainex2 %.% group_by(plan) %.%
    summarise(planfreq=n()/nrowtrainex2)
trainex2 <- left_join(trainex2, planfreqs)

# trainex3 is identical to trainex2 but also includes purchases
trainex3 <- cbind(trainex2, Apurch=trainsub$A, Bpurch=trainsub$B,
    Cpurch=trainsub$C, Dpurch=trainsub$D, Epurch=trainsub$E, Fpurch=trainsub$F,
    Gpurch=trainsub$G, planpurch=trainsub$plan, stringsAsFactors=FALSE)

# read in test set
test <- read.csv("test_v2.csv", colClasses=colClasses)
test <- preprocess(test)

# fix locations that are NA
s <- split(test$location, test$state)
s2 <- sapply(s, function(x) x[1])
NAstates <- test[is.na(test$location), "state"]
NAlocations <- s2[NAstates]
test$location[is.na(test$location)] <- NAlocations

# add "changed" variable and default to No
test$changed <- factor(rep("No", nrow(test)), levels=c("No", "Yes"))

# testsub only shows last (known) quote before purchase
testsub <- test[!duplicated(test$customer_ID, fromLast=TRUE), ]

# compute "stability" feature from test and add to testsub
customerstability <- test %.% group_by(customer_ID) %.% summarise(quotes=n(),
    uniqueplans=n_distinct(plan), stability=(quotes-uniqueplans+1)/quotes)
testsub$stability <- customerstability$stability

# compute "planfreq" feature on testsub
nrowtestsub <- nrow(testsub)
planfreqs <- testsub %.% group_by(plan) %.% summarise(planfreq=n()/nrowtestsub)
testsub <- left_join(testsub, planfreqs)


## DATA EXPLORATION ##

# check for NA values
sapply(train, function(x) mean(is.na(x)))
    # risk_factor, C_previous, duration_previous
sapply(test, function(x) mean(is.na(x)))
    # risk_factor, C_previous, duration_previous, location

uniquetrainplan <- unique(train$plan)
uniquetestplan <- unique(test$plan)
    # plans in train: 1809
    # plans in test: 1596
    # union: 1878 (69 plans in test that are not in train)
    # intersection: 1527


## VISUALIZATIONS ##

# Viz 1: Number of shopping points
shoptrain <- data.frame(maxpoint=trainex2$shopping_pt, dataset=rep("train",
    nrow(trainex2)))
shoptest <- data.frame(maxpoint=testsub$shopping_pt, dataset=rep("test",
    nrow(testsub)))
shoppingpoints <- rbind(shoptrain, shoptest)
shoppingpoints$dataset <- as.factor(shoppingpoints$dataset)
ggplot(shoppingpoints) + aes(factor(maxpoint)) + geom_histogram() +
    facet_grid(dataset ~ .) + labs(x="Number of Shopping Points",
    y="Frequency", title="Comparing Number of Shopping Points in
    Training vs Test Sets")
ggsave("allstate-viz-1.png")

# Viz 2: Predictive power of final quote before purchase
s <- split(trainex2, trainex2$shopping_pt)
s2 <- sapply(s, function(x) sum(x$changed=="No")/nrow(x))
s2b <- sapply(s, nrow)
acclastentry <- data.frame(numshoppoints=as.integer(names(s2)), accuracy=s2,
    Observations=s2b)
ggplot(acclastentry) + aes(numshoppoints, accuracy, size=Observations) +
    geom_point() + geom_line(size=0.5) + scale_x_continuous(breaks=1:12) +
    theme(panel.grid.minor=element_blank()) + labs(x="Number of Shopping
    Points", y="Prediction Accuracy", title="Effect of Number of Shopping
    Points on Predictive Power of Last Quote")
ggsave("allstate-viz-2.png")

# Viz 3: Effect of purchase hour on the predictive power of the final quote
s3 <- split(trainsub, trainsub$hour)
s4 <- sapply(s3, function(x) sum(x$changed=="Yes")/nrow(x))
s5 <- as.data.frame(table(trainsub$hour))$Freq
changebyhour <- data.frame(hour=as.integer(names(s4)),
    percentchanged=s4, count=s5)
ggplot(changebyhour) + aes(hour, percentchanged, color=count) +
    geom_point(size=4) + labs(x="Hour of Purchase", y="Percent Changed",
    title="Effect of Purchase Hour on Likelihood of Changing from Last Quote")
ggsave("allstate-viz-3.png")

# Viz 4: Dependencies between options
C_names <- list("1"="C=1", "2"="C=2", "3"="C=3", "4"="C=4")
C_labeller <- function(variable, value){ return(C_names[value]) }
ggplot(trainsub, aes(D)) + geom_bar() + facet_grid(. ~ C,
    labeller=C_labeller) + labs(x="Customer Selection of Option D (1, 2, 3)",
    y="Frequency", title="Customer selection of Option D based on their
    selection for Option C")
ggsave("allstate-viz-4.png")

# Viz 5: Clustering of states
# based on: http://is-r.tumblr.com/post/37708137014/us-state-maps-using-map-data
states <- map_data("state")
states$grp <- as.factor(ifelse(states$region %in% c("south dakota",
        "north dakota"), "1 (least likely)",
    ifelse(states$region %in% c("alabama","wyoming"), "2",
    ifelse(states$region %in% c("oklahoma","maine","arkansas","west virginia"),
        "3",
    ifelse(states$region %in% c("colorado","connecticut","delaware","florida",
        "iowa","idaho","indiana","kansas","kentucky","maryland","missouri",
        "mississippi","montana","new hampshire","new mexico","nevada",
        "new york","ohio","oregon","pennsylvania","rhode island","tennessee",
        "utah","washington","wisconsin"), "4",
    ifelse(states$region %in% c("district of columbia","nebraska","georgia"),
        "5 (most likely)", "unassigned"))))))
ggplot(states) + aes(x=long, y=lat, group=group, fill=grp) +
    geom_polygon(color="black") + theme_bw() +
    theme(panel.border=element_blank()) + scale_y_continuous(breaks=c()) +
    scale_x_continuous(breaks=c()) + labs(title="Clustering of States
    Based on Customer Likelihood of Changing from Last Quote", fill="Cluster",
    x="", y="") + scale_fill_brewer(palette="Pastel1")
ggsave("allstate-viz-5.png")


## MODEL BUILDING ##
## Note: Each "PART" is explained in the paper

# PART 1:

# Submit the baseline
pred <- data.frame(customer_ID = testsub$customer_ID, plan = testsub$plan)
write.csv(pred, file="submit1.csv", row.names=FALSE, quote=FALSE)    # 0.53793

# PART 2:

# Rule-based predictions
# Example: if C=4, then change D to 3
testsub$D[testsub$C==4] <- 3
filter(testsub, C==4, D!=3)
pred <- data.frame(customer_ID = testsub$customer_ID,
                   plan = paste0(testsub$A, testsub$B, testsub$C, testsub$D,
                                 testsub$E, testsub$F, testsub$G))
write.csv(pred, file="submit2.csv", row.names=FALSE, quote=FALSE)    # 0.53769

# PART 4:

# logistic regression for predicting "changed"
glm.fit <- glm(changed ~ state+cost+A+C+D+E+F+G+age_oldest+age_youngest+
    car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+C_previous+
    duration_previous+stability+planfreq, data=trainex2, family=binomial)
summary(glm.fit)
glm.probs <- predict(glm.fit, type="response")
glm.pred <- ifelse(glm.probs>0.5, "Yes", "No")
confusionMatrix(glm.pred, trainex2$changed, "Yes")
predob <- prediction(glm.probs, trainex2$changed)
acc <- performance(predob, "acc"); plot(acc)
prec <- performance(predob, "prec"); plot(prec)
roc <- performance(predob, "tpr", "fpr"); plot(roc)

# 5-fold CV for logistic regression
set.seed(5)
folds <- sample(rep(1:5, length = nrow(trainex2)))
for(k in 1:5) {
    fit <- glm(changed ~ state+cost+A+C+D+E+F+G+age_oldest+age_youngest+
        car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+C_previous+
        duration_previous+stability, data=trainex2[folds!=k, ],
        family=binomial)
    probs <- predict(fit, newdata=trainex2[folds==k, ], type="response")
    pred <- ifelse(probs>0.5, "Yes", "No")
    print(mean(pred==trainex2$changed[folds==k]))
}

# random forests for predicting "changed"
rf.fit <- randomForest(changed ~ stategroup+cost+A+C+D+E+F+G+age_oldest+
    age_youngest+car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+
    C_previous+duration_previous+stability+planfreq, data=trainex2, mtry=5)
rf.pred <- ifelse(rf.fit$votes[, 2]>0.5, "Yes", "No")
confusionMatrix(rf.pred, trainex2$changed, "Yes")

# PART 5:

# 5-fold CV for logistic regression - for precision at 0.85 cutoff
set.seed(5)
folds <- sample(rep(1:5, length = nrow(trainex2)))
predyestotal <- 0
tptotal <- 0
for(k in 1:5) {
    fit <- glm(changed ~ state+cost+A+C+D+E+F+G+age_oldest+age_youngest+
        car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+C_previous+
        duration_previous+stability+planfreq, data=trainex2[folds!=k, ],
        family=binomial)
    probs <- predict(fit, newdata=trainex2[folds==k, ], type="response")
    pred <- as.factor(ifelse(probs>0.85, "Yes", "No"))
    predyes <- sum(pred=="Yes")
    predyestotal <- predyestotal + predyes
    actual <- trainex2$changed[folds==k]
    actualwhenpredyes <- actual[pred=="Yes"]
    tp <- sum(actualwhenpredyes=="Yes")
    tptotal <- tptotal + tp
}
print(tptotal)
print(predyestotal)
print(tptotal/predyestotal)

# train model on trainex2 and predict changed on testsub
glm.fit <- glm(changed ~ state+cost+A+C+D+E+F+G+age_oldest+age_youngest+
    car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+C_previous+
    duration_previous+stability, data=trainex2, family=binomial)
glm.probs <- predict(glm.fit, newdata=testsub, type="response")
glm.pred <- ifelse(glm.probs>0.85, "Yes", "No")

# update "changed" variable on testsub to reflect prediction
testsub$changed <- as.factor(glm.pred)

# for records predicting changed, planpred = 9999999
testsub$planpred <- ifelse(testsub$changed=="Yes", "9999999", testsub$plan)
pred <- data.frame(customer_ID = testsub$customer_ID, plan = testsub$planpred)
write.csv(pred, file="submit6.csv", row.names=FALSE, quote=FALSE)
    # 52 changes, 16 affect leaderboard
    # 0.53769 means 4 below baseline

# PART 6:

# 5-fold CV for random forests - for precision at 0.85 cutoff
set.seed(5)
folds <- sample(rep(1:5, length = nrow(trainex2)))
predyestotal <- 0
tptotal <- 0
for(k in 1:5) {
    fit <- randomForest(changed ~ stategroup+cost+A+C+D+E+F+G+age_oldest+
        age_youngest+car_value+car_age+shopping_pt+timeofday+weekend+
        risk_factor+C_previous+duration_previous+stability+planfreq,
        data=trainex2[folds!=k, ], mtry=5)
    votes <- predict(fit, newdata=trainex2[folds==k, ], type="vote")
    pred <- as.factor(ifelse(votes[, 2]>0.85, "Yes", "No"))
    predyes <- sum(pred=="Yes")
    predyestotal <- predyestotal + predyes
    actual <- trainex2$changed[folds==k]
    actualwhenpredyes <- actual[pred=="Yes"]
    tp <- sum(actualwhenpredyes=="Yes")
    tptotal <- tptotal + tp
}
print(tptotal)
print(predyestotal)
print(tptotal/predyestotal)

# PART 7 and PART 8:

# multinom to predict A (repeat for each option)
a.mn.fit <- multinom(A ~ .-customer_ID-record_type-day-time-location-plan-
    hour-agediff-stategroup, data=trainex2)
a.mn.pred <- predict(a.mn.fit, newdata=trainex2, type="class")
confusionMatrix(a.mn.pred, trainsub$A)

# multinom to predict Apurch (repeat for each option)
a.mn.fit <- multinom(Apurch ~ .-customer_ID-record_type-day-time-location-
    plan-hour-agediff-stategroup-Bpurch-Cpurch-Dpurch-Epurch-Fpurch-Gpurch-
    planpurch, data=trainex3)
a.mn.pred <- predict(a.mn.fit, newdata=trainex3, type="class")
mean(a.mn.pred==trainex3$Apurch)
mean(a.mn.pred==trainex3$A)

# random forests to predict A (repeat for each option)
a.rf.fit <- randomForest(A ~ .-customer_ID-record_type-day-time-location-
    plan-hour-agediff-state, data=trainex2, importance=FALSE)
mean(a.rf.fit$predicted==trainex2$A)
mean(a.rf.fit$predicted==trainsub$A)

# random forests to predict Apurch (repeat for each option)
a.rf.fit <- randomForest(Apurch ~ .-customer_ID-record_type-day-time-location-
    plan-hour-agediff-state-Bpurch-Cpurch-Dpurch-Epurch-Fpurch-Gpurch-
    planpurch, data=trainex3, importance=FALSE)
mean(a.rf.fit$predicted==trainex3$Apurch)
mean(a.rf.fit$predicted==trainex3$A)

# random forests to predict Apurch, trained only on subset for which change is
    # predicted (repeat for each option)
trainex4 <- trainex3[trainex3$changed=="Yes", ]
a.rf.fit <- randomForest(Apurch ~ .-customer_ID-record_type-day-time-location-
    plan-hour-agediff-state-Bpurch-Cpurch-Dpurch-Epurch-Fpurch-Gpurch-
    planpurch, data=trainex4, importance=FALSE)
mean(a.rf.fit$predicted==trainex4$Apurch)
mean(a.rf.fit$predicted==trainex4$A)

# combine option predictions for A through G
pred.plan <- paste0(a.rf.fit$predicted, b.rf.fit$predicted, c.rf.fit$predicted,
    d.rf.fit$predicted, e.rf.fit$predicted, f.rf.fit$predicted,
    g.rf.fit$predicted)

# for people I predict changed in train, did I get it right?
predchange.ix <- which(glm.pred=="Yes")
predchange.cid <- trainex2[predchange.ix, "customer_ID"]
for (i in 1:length(predchange.cid)) {
    # customer_ID
    print(predchange.cid[i])
    # final quote
    print(trainex2[trainex2$customer_ID==predchange.cid[i], "plan"])
    # predicted purchase
    print(pred.plan[predchange.ix[i]])
    # actual purchase
    print(trainsub[trainsub$customer_ID==predchange.cid[i], "plan"])
}

# PART 9:

# train model on trainex2 and predict changed on testsub
glm.fit <- glm(changed ~ state+cost+A+C+D+E+F+G+age_oldest+age_youngest+
    car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+C_previous+
    duration_previous+stability+planfreq, data=trainex2, family=binomial)
glm.probs <- predict(glm.fit, newdata=testsub, type="response")
glm.pred <- ifelse(glm.probs>0.9, "Yes", "No")

# update changed variable on testsub to reflect prediction
testsub$changed <- as.factor(glm.pred)

# for records predicting changed, start by predicting plan, then modify each one
testsub$planpred <- testsub$plan
testsub$planpred[testsub$customer_ID=="10006040"] <- "1033021"
testsub$planpred[testsub$customer_ID=="10011049"] <- "1133112"
testsub$planpred[testsub$customer_ID=="10026297"] <- "0011002"
testsub$planpred[testsub$customer_ID=="10027789"] <- "0012002"
testsub$planpred[testsub$customer_ID=="10054016"] <- "0011002"
testsub$planpred[testsub$customer_ID=="10054825"] <- "0011001"
testsub$planpred[testsub$customer_ID=="10068571"] <- "0011002"
testsub$planpred[testsub$customer_ID=="10113734"] <- "2022032"
testsub$planpred[testsub$customer_ID=="10116125"] <- "0012002"

# submit
pred <- data.frame(customer_ID = testsub$customer_ID, plan = testsub$planpred)
write.csv(pred, file="submit11.csv", row.names=FALSE, quote=FALSE)
    # 9 changes, 3 affect leaderboard
    # 0.53793 means 0 below baseline

# PART 10:

# calculate change likelihood of plans
rec <- train %.% group_by(plan) %.% summarise(planpur=mean(record_type),
    plancnt=n()) %.% arrange(planpur, desc(plancnt))

# function to "fix" plans based upon thresholds
fixplans <- function(planpurmax, plancntmin, commonmin) {

    # make list of fixes
    rectop <- rec[rec$planpur<=planpurmax & rec$plancnt>=plancntmin, "plan"]
    rectopbest <- vector(mode="character", length=length(rectop))
    rectopcommon <- vector(mode="numeric", length=length(rectop))
    
    for (i in 1:length(rectop)) {
        # vector of unique customers that looked at that plan
        cust <- unique(train[train$plan==rectop[i], "customer_ID"])
        # what are all the plans that those customers purchased?
        purplan <- train[train$customer_ID %in% cust & train$record_type==1,
            "plan"]
        # what was the most common purchased?
        rectopbest[i] <- names(sort(table(purplan), decreasing=TRUE))[1]
        # how common was it?
        rectopcommon[i] <- sort(table(purplan),
            decreasing=TRUE)[1]/length(purplan)
    }
    
    fixes <- data.frame(old=rectop[rectopcommon>=commonmin],
        new=rectopbest[rectopcommon>=commonmin], stringsAsFactors=FALSE)
    fixes <- fixes[fixes$old!=fixes$new, ]
    print(nrow(fixes))
    print(fixes)
    
    # reset testsub, and check how many fixes will be made
    testsub <- test[!duplicated(test$customer_ID, fromLast=TRUE), ]
    testsub$planpred <- testsub$plan
    print(sum(testsub$planpred %in% fixes$old))
    
    # make fixes
    nrowtestsub <- nrow(testsub)
    for (i in 1:nrowtestsub) {
        if (testsub$planpred[i] %in% fixes$old) {
            testsub$planpred[i] <-
                fixes$new[which(fixes$old==testsub$planpred[i])]
        }
    }
    
    return(testsub)
}

# submission that had my best PUBLIC leaderboard score
# 5 plans, 322 fixes
# public score: 0.53853, above baseline by 10 picks
# private score: 0.53266, below baseline by 1 pick
ts <- fixplans(0.05, 70, 0.05)
pred <- data.frame(customer_ID = ts$customer_ID, plan = ts$planpred)
write.csv(pred, file="submit34.csv", row.names=FALSE, quote=FALSE)

# submission that had my best PRIVATE leaderboard score
# 2 plans, 305 fixes
# public score: 0.53847, above baseline by 9 picks
# private score: 0.53277, above baseline by 3 picks
ts <- fixplans(0.05, 500, 0.05)
pred <- data.frame(customer_ID = ts$customer_ID, plan = ts$planpred)
write.csv(pred, file="submit25.csv", row.names=FALSE, quote=FALSE)

# PART 11:

# predict change (for 12403 people, 100+ plans, 98 fixes)
glm.fit <- glm(changed ~ state+cost+A+C+D+E+F+G+age_oldest+age_youngest+
    car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+C_previous+
    duration_previous+stability+planfreq, data=trainex2, family=binomial)
glm.probs <- predict(glm.fit, newdata=testsub, type="response")
glm.pred <- ifelse(glm.probs>0.5, "Yes", "No")
table(glm.pred)
ts <- fixplans(0.2, 10, 0.05)
ts$changed <- as.factor(glm.pred)
nrow(ts[ts$planpred!=ts$plan & ts$changed=="Yes", ])
ts$planpred <- ifelse(ts$changed=="No", ts$plan, ts$planpred)
nrow(ts[ts$planpred!=ts$plan,])

# submit: 0.53787, 1 below baseline
pred <- data.frame(customer_ID = ts$customer_ID, plan = ts$planpred)
write.csv(pred, file="submit24.csv", row.names=FALSE, quote=FALSE)
