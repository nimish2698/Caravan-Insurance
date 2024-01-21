# Load necessary packages
library(caret)
library(ROSE)
library(ISLR)
library(randomForest)
library(MASS)
library(pROC)
# Reading the dataset from a CSV file
dataset = read.csv('C:/Nimish/Imp docs/University of Surrey/Semester 1/Principle of Analytics/Coursework/newtrainingdata.csv')
# Checking if there's any null value in our data or not
df <- is.null(dataset)
df
# Extracting all columns except the first one and storing it in D
D <- dataset[, -1]
# Setting the seed for random number generation
set.seed(123)
#Assigning Target
target <- D$CARAVAN
# Creating a data partition for training and testing sets (70% training, 30% testing)
I <- createDataPartition(target, p = 0.7, list = FALSE)
train <- D[I,]
test <- D[-I,]
# Creating a data partition for training and testing sets (80% training, 20% testing)
I1 <- createDataPartition(target, p = 0.8, list = FALSE)
train1 <- D[I1,]
test1 <- D[-I1,]
# Creating a data partition for training and testing sets (60% training, 40% testing)
I2 <- createDataPartition(target, p = 0.6, list = FALSE)
train2 <- D[I2,]
test2 <- D[-I2,]
# Performing Stepwise Backward Selection for Feature Selection on training data (70%)
model <- lm(CARAVAN ~., data = train)
step_model <- step(model, direction = "backward")
# Performing Stepwise Backward Selection for Feature Selection on training data1 (80%)
model1 <- lm(CARAVAN ~., data = train1)
step_model1 <- step(model1, direction = "backward")
# Performing Stepwise Backward Selection for Feature Selection on training data2 (60%)
model2 <- lm(CARAVAN ~., data = train2)
step_model2 <- step(model2, direction = "backward")
# Over Sampling the training data (70%) using ROSE Sampling method
train_balanced_over <- ovun.sample(CARAVAN~ Cont_Thirdparty_Private + Cont_Car + Cont_Life + Cont_FamilyAccident + 
                                     Cont_Disability + Cont_Fire + Cont_Boat + Life + FamilyAccident + 
                                     Bicycle, data = train, method = 'over', seed = 123)$data
train_balanced_over$CARAVAN <- as.factor(train_balanced_over$CARAVAN) # Converting the target column to factors
# Over Sampling the training data (80%) using ROSE Sampling method
train_balanced_over1 <- ovun.sample(CARAVAN~ Cont_Thirdparty_Private + Cont_Car + Cont_Life + Cont_FamilyAccident + 
                                     Cont_Disability + Cont_Fire + Cont_Boat + Life + FamilyAccident + 
                                     Bicycle, data = train1, method = 'over', seed = 123)$data
train_balanced_over1$CARAVAN <- as.factor(train_balanced_over1$CARAVAN) # Converting the target column to factors
# Over Sampling the training data (60%) using ROSE Sampling method
train_balanced_over2 <- ovun.sample(CARAVAN~ Cont_Thirdparty_Private + Cont_Car + Cont_Life + Cont_FamilyAccident + 
                                     Cont_Disability + Cont_Fire + Cont_Boat + Life + FamilyAccident + 
                                     Bicycle, data = train2, method = 'over', seed = 123)$data
train_balanced_over2$CARAVAN <- as.factor(train_balanced_over2$CARAVAN) # Converting the target column to factors
# Creating a training model using Random Forest Algorithm for training data (70%)
train_model <- randomForest(CARAVAN~ Cont_Thirdparty_Private + Cont_Car + Cont_Life + Cont_FamilyAccident + 
                              Cont_Disability + Cont_Fire + Cont_Boat + Life + FamilyAccident + 
                              Bicycle, data = train_balanced_over) 
# Creating a training model using Random Forest Algorithm for training data1 (80%)
train_model1 <- randomForest(CARAVAN~ Cont_Thirdparty_Private + Cont_Car + Cont_Life + Cont_FamilyAccident + 
                              Cont_Disability + Cont_Fire + Cont_Boat + Life + FamilyAccident + 
                              Bicycle, data = train_balanced_over1)
# Creating a training model using Random Forest Algorithm for training data2 (60%)
train_model2 <- randomForest(CARAVAN~ Cont_Thirdparty_Private + Cont_Car + Cont_Life + Cont_FamilyAccident + 
                              Cont_Disability + Cont_Fire + Cont_Boat + Life + FamilyAccident + 
                              Bicycle, data = train_balanced_over2) 
# Testing the prediction model for test data (30%) and creating a Confusion Matrix for the same
test$CARAVAN <- factor(test$CARAVAN) # Converting the target column to factors
p <- predict(train_model, test)
conf_matrix <- confusionMatrix(p, test$CARAVAN, positive = '1')
conf_matrix
p_numeric <- as.numeric(p)
roc_curve <- roc(test$CARAVAN, p_numeric, plot = TRUE, legacy.axes = TRUE, percent = TRUE) # Creating a ROC Curve for the predicted model
auc_value <- auc(roc_curve) #Determining the AUC value for the predicted model
print(paste("AUC Value :", auc_value))
# Testing the prediction model for test data (20%) and creating a Confusion Matrix for the same
test1$CARAVAN <- factor(test1$CARAVAN) # Converting the target column to factors
p1 <- predict(train_model1, test1)
conf_matrix1 <- confusionMatrix(p1, test1$CARAVAN, positive = '1')
conf_matrix1
p1_numeric <- as.numeric(p1)
roc_curve1 <- roc(test1$CARAVAN, p1_numeric, plot = TRUE, legacy.axes = TRUE, percent = TRUE) # Creating a ROC Curve for the predicted model
auc_value1 <- auc(roc_curve1) #Determining the AUC value for the predicted model
print(paste("AUC Value :", auc_value1))
# Testing the prediction model for test data (40%) and creating a Confusion Matrix for the same
test2$CARAVAN <- factor(test2$CARAVAN) # Converting the target column to factors
p2 <- predict(train_model2, test2)
conf_matrix2 <- confusionMatrix(p2, test2$CARAVAN, positive = '1')
conf_matrix2
p2_numeric <- as.numeric(p2)
roc_curve2 <- roc(test2$CARAVAN, p2_numeric, plot = TRUE, legacy.axes = TRUE, percent = TRUE) # Creating a ROC Curve for the predicted model
auc_value2 <- auc(roc_curve2) #Determining the AUC value for the predicted model
print(paste("AUC Value :", auc_value2))