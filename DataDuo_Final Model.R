# Author: DataDuo
# Load necessary libraries
library(caret)       
library(dplyr)       
library(ROSE)        
library(gbm)
library(pROC)
library(glmnet)
library(MASS)

set.seed(42)

# Read train datasets
dat <- read.csv("train.csv")[-1]
dat_test <- read.csv("test.csv")


# Data Understanding
summary(dat)
table(dat$churn)

# Calculate class distribution 
class_proportions <- table(dat$churn) / nrow(dat)
print(class_proportions)
ggplot(dat, aes(x = factor(churn), fill = factor(churn))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "light blue", "1" = "light pink")) +
  labs(title = "Churn Distribution", x = "Churn Status", y = "Count") +
  theme_minimal()

# Calculate correlation
cor(dat)

# Plotting scatterplot 
dat$churn <- as.factor(dat$churn)
plot(dat$clicks, dat$visits, col=dat$churn)

# Plotting histogram to see variables distribution
hist(dat$daysInactiveAvg,
     main = "Histogram of daysInactiveAvg", 
     xlab = "Days Inactive Average", 
     col = "gray",
     border = "white" 
)
hist(dat$clicks ,
     main = "Histogram of Clicks ", 
     xlab = "Clicks", 
     col = "gray", 
     border = "white" 
)
hist(dat$timeOfDay  ,
     main = "Histogram of Time of Day", 
     xlab = "timeOfDay", 
     col = "gray", 
     border = "white" 
)
hist(dat$weekdayPercent  ,
     main = "Histogram of Weekday Percent", 
     xlab = "weekdayPercent", 
     col = "gray", 
     border = "white" 
)

# Check for missing values in datasets:
colSums(is.na(dat))
dim(dat)

# Detect outliers using IQR
detect_outliers_IQR <- function(x) {
  if(is.numeric(x)) { 
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    outliers <- x[x < lower_bound | x > upper_bound]
    return(outliers)
  } else {
    return(NA) 
  }
}
outliers_list <- lapply(dat, detect_outliers_IQR)
# Counting number of outliers for each variable
outlier_counts <- sapply(outliers_list, function(x) {
  if(is.numeric(x)) length(x) else NA
})
outlier_counts

# Data Preparation 

# Feature Engineering - Time of the Day Bin variable added
dat$timeOfDayBin <- cut(dat$timeOfDay, breaks=c(0,6,12,18,24), labels=c("Night","Morning","Afternoon","Evening"), include.lowest=TRUE)
dat_test$timeOfDayBin <- cut(dat_test$timeOfDay, breaks=c(0,6,12,18,24), labels=c("Night","Morning","Afternoon","Evening"), include.lowest=TRUE)

# One-hot encode timeofDayBin in the training data
dummies_train <- dummyVars(~ timeOfDayBin, data = dat)
dat_train_one_hot <- predict(dummies_train, newdata = dat)[,-1]
dat$timeOfDayBin <- NULL  # Drop the original categorical variable
dat <- cbind(dat, dat_train_one_hot)[,-11]

# One-hot encode timeofDayBin in the testing data
dummies_test <- dummyVars(~ timeOfDayBin, data = dat_test)
dat_test_one_hot <- predict(dummies_test, newdata = dat_test)[,-1]
dat_test$timeOfDayBin <- NULL  # Drop the original categorical variable
dat_test <- cbind(dat_test, dat_test_one_hot)[,-12]

# Capping outliers
cap_outliers_IQR <- function(x) {
  if(is.numeric(x)) {
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    
    # Cap values
    x[x < lower_bound] <- lower_bound
    x[x > upper_bound] <- upper_bound
  }
  return(x)
}
numeric_columns <- sapply(dat, is.numeric)
dat[numeric_columns] <- lapply(dat[numeric_columns], cap_outliers_IQR)
numeric_columns_test <- sapply(dat_test, is.numeric)
dat_test[numeric_columns_test] <- lapply(dat_test[numeric_columns_test], cap_outliers_IQR)

# Splitting dat into train and validation sets
dat$churn <- as.factor(dat$churn)
trainIndex <- createDataPartition(dat$churn, p = .7, list = FALSE, times = 1)
dat_train <- dat[trainIndex, ]
dat_val <- dat[-trainIndex, ]



# Filter out zero-variance variables
nzv <- nearZeroVar(dat_train, saveMetrics = TRUE)
nonZeroVarCols <- names(dat_train)[!nzv$nzv]

# Exclude "id" and "churn" for scaling
dat_train_filtered <- dat_train[, nonZeroVarCols]
dat_val_filtered <- dat_val[, nonZeroVarCols]
dat_test_filtered <- dat_test[, names(dat_test) %in% nonZeroVarCols]
features_train <- dat_train_filtered[, -which(names(dat_train_filtered) %in% c("churn"))]
features_val <- dat_val_filtered[, -which(names(dat_val_filtered) %in% c("churn"))]
features_test <- dat_test_filtered

# Scaling training set
preProcValues <- preProcess(features_train, method = c("center", "scale"))
features_train_scaled <- predict(preProcValues, features_train)

# Scaling validation and test set
features_val_scaled <- predict(preProcValues, features_val)
features_test_scaled <- predict(preProcValues, features_test)

#Combine everything for train, validation and test sets
dat_test_scaled <- cbind(dat_test[, "id", drop = FALSE], features_test_scaled)
dat_train_scaled <- cbind(dat_train[, "churn", drop = FALSE], features_train_scaled)
dat_val_scaled <- cbind(dat_val[, "churn", drop = FALSE], features_val_scaled)

# Stepwise Selection using AIC 
initialModel <- glm(churn ~ ., data = dat_train_scaled, family = "binomial")
stepwiseModel <- stepAIC(initialModel, direction = "both", trace = FALSE)
summary(stepwiseModel)

# Extracting selected predictors
features <- all.vars(formula(stepwiseModel))[-1] 
print(features)

# Define features and target for training data
X_train <- dat_train_scaled[, features]
y_train <- dat_train_scaled$churn

# Define features and target for validation data
X_val <- dat_val_scaled[, features]
y_val <- dat_val_scaled$churn
levels(y_train) <- levels(y_val) <- c("0", "1")

# Fitting models and performance comparison
models <- list()
predictions <- list()
auc_values <- list()
levels(y_val) <- levels(y_train) <- c("Level0", "Level1")

# GBM Model with Accuracy as Metric
models$gbm_accuracy <- train(x = X_train, y = y_train, method = "gbm", verbose = FALSE, trControl = trainControl(method = "cv", number = 10))

# GBM Model with AUC as Metric
train_control <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)

# Define a hyperparameter grid
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5),
                       n.trees = (1:3) * 50,
                       shrinkage = c(0.01, 0.1),
                       n.minobsinnode = c(10, 20))

# Train the GBM model with AUC as metric with a consistent grid
models$gbm_auc <- train(x = X_train, y = y_train, method = "gbm",
                         tuneGrid = gbmGrid,
                         metric = "ROC",
                         trControl = train_control,
                         verbose = FALSE)

# Predict on the validation set
predictions$gbm_accuracy <- predict(models$gbm_accuracy, newdata = X_val, type = "prob")[,2]
predictions$gbm_auc <- predict(models$gbm_auc, newdata = X_val, type = "prob")[,2]

# Calculate AUC
for(model_name in names(predictions)) {
  # Ensure predictions are numeric
  pred_numeric <- as.numeric(predictions[[model_name]])
  
  roc_obj <- roc(response = as.numeric(y_val), predictor = pred_numeric)
  auc_values[[model_name]] <- auc(roc_obj)
}
# Print AUC values for all models
print(auc_values)

# Plot ROC curve for the first model to establish the plot
roc_obj_gbm_accuracy <- roc(response = as.numeric(y_val), predictor = as.numeric(predictions$gbm_accuracy))
plot(roc_obj_gbm_accuracy, main="ROC Curves Comparison", col="blue")
# Add ROC curve for the second model
roc_obj_gbm_auc <- roc(response = as.numeric(y_val), predictor = as.numeric(predictions$gbm_auc))
lines(roc_obj_gbm_auc, col="red")
# Add a legend to the plot
legend("bottomright", legend=c("GBM-Accuracy", "GBM-AUC"), col=c("blue", "red"), lwd=2)

# Plot variable importance for GBM-Accuracy Model
var_imp_df <- as.data.frame(gbm_var_imp$importance)
var_imp_df$Feature <- rownames(var_imp_df)
ggplot(var_imp_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() + # Flip coordinates for horizontal bars
  theme_minimal() +
  labs(title = "Feature Importance from GBM Model",
       x = "Features",
       y = "Importance Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Fitting the chosen model on test set 
X_test <- dat_test_scaled[, features]
test_preds <- predict(models$gbm_accuracy, X_test, type = "prob")[,2]

# Extracting the results
submission <- data.frame(id = dat_test$id, churn = test_preds)
write.csv(submission, 'submission_final.csv', row.names = FALSE)