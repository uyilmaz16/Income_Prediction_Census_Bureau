library(onehot)
library(mltools)
library(h2o)
library(corrplot)
library(Boruta)
library(InformationValue)
library(e1071)
library(class)
library(tidyselect)
library(stringr)
library(DT)

# Data preparation
dataset <- read.csv('Dataset.csv', header = TRUE, sep = ";" )
set.seed(2)
rows <- sample(nrow(dataset))
Dataset <- dataset[rows, ]

# Removing some redundant columns
Dataset <- Dataset[,-c(1,6,16)]

# Specifying the target as a factor variable
Dataset[,14] <- as.factor(Dataset[,14])

# Separating the response and predictors
X_train <- Dataset[,-c(14)]
Y_train <- data.matrix(as.factor(Dataset[,14]))
colnames(Y_train) <- "TARGET"

# Checking for missing values
NAvalues <- data.matrix(which(X_train == '?', arr.ind = TRUE))
unique(NAvalues[,2])

# Numeric and factor variables
Numericvars <- X_train[,c(1,3,10,11,12)]
Factorvars <- X_train[,c(2,4,5,6,7,8,9,13)]
 
# Information Value of factor variables
factorvars <- c ("WORKCLASS", "EDUCATION", "MARITAL_STATUS", "OCCUPATION", "RELATIONSHIP", "RACE", "SEX", "COUNTRY")
infval <- data.frame(VARS=factorvars, IV=numeric(length(factorvars)), STRENGTH=character(length(factorvars)), stringsAsFactors = FALSE) 

# Information Value for each factor variable
for (factorcol in factorvars){
  infval[infval$VARS == factorcol, "IV"] <- InformationValue::IV(X=Factorvars[, factorcol], Y=Y_train)
  infval[infval$VARS == factorcol, "STRENGTH"] <- attr(InformationValue::IV(X=Factorvars[, factorcol], Y=Y_train), "howgood")
}
infval <- infval[order(-infval$IV), ]
print(infval)
DT::datatable(infval)


# Onehot encoding
encoder <- onehot(X_train, max_levels = Inf)
X_train_d <- data.frame(predict(encoder, data = X_train))

# GLM for Logistic Regression
lreg <- glm(TARGET~.,family = "binomial", Dataset)
summary(lreg)
lregpred <- lreg[["fitted.values"]]
Datasetn <- cbind(lregpred,Dataset)

# Significant variables
varsignf <- summary(lreg)$coeff[-1,4] < 0.05
signvars <- names(varsignf)[varsignf == TRUE]
X_trainvars <- names(X_train_d)

# Making the variable names the same between the output of the GLM and the onehot encoder, to match them afterwards
signvarss <- c()
for (i in 1:length(signvars)){
signvarss[i] <- gsub("-","", signvars[i])
}

X_trainvarss <- c()
for (i in 1:length(X_trainvars)){
  X_trainvarss[i] <- gsub("\\.","", X_trainvars[i])
}

# Selection of significant variables
colnames(X_train_d) <- X_trainvarss
X_train_ds <- X_train_d[,signvarss]


# Random forest including all variables and the output of the GLM, with stratified 5-fold validation.
h2o.init()
rf <- h2o.randomForest(x = c(1:14), y = 15, as.h2o(Datasetn),fold_assignment = "Stratified", nfolds = 5, ntrees = 200)
rf@model[["variable_importances"]]


Dataset_s <- cbind.data.frame(X_train_ds,Y_train)
rff <-  h2o.randomForest(x = c(1:44), y = 45, as.h2o(Dataset_s),
                         fold_assignment = "Stratified", nfolds = 5, ntrees = 100)

# KNN on the subset of the significant variables, with leave-one-out cross validation
KNN <- knn.cv(X_train_ds, cl = Y_train, k = 5)
KNN

# SVM on the significant variable subset, with 3-fold cross validation
SVM <- svm(TARGET~., data = Dataset_s, kernel = "radial", cross = 3)
SVMpred <- data.matrix(factor(data.matrix(SVM$fitted)))

# The final dataset and model
Dataset_ensmbl <- cbind(lregpred,KNN,SVMpred,Dataset)
rf_ensmbl <- h2o.randomForest(x = c(1:16), y = 17, as.h2o(Dataset_ensmbl),fold_assignment = "Stratified", nfolds = 5, ntrees = 120)
rf_ensmbl@model[["variable_importances"]]

# Prediction
predictions <- predict(rf_ensmbl, as.h2o(Dataset_ensmbl))
prediction <- data.matrix(predictions[,1])
write.table(prediction, file = "predictions.csv", row.names = FALSE, sep = ",")





