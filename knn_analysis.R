
# SED612, SIT@KMUTT
# Aug. 2022
# By PM

# Use k-nearest neighbors, for classification


# Our 1st dataset
# Use Iris
# Source: https://www.kaggle.com/uciml/iris

irs <- read.csv("data/datasets_19_420_Iris.csv")
# check its dimension
dim(irs)
# split the data into training and testing, use a 80-20 rule.
# take 10 data points for each Iris type and keep them for testing, do random sampling

# set.seed(612) # set the seed for sampling, if needed to replicate, when experimenting

rowID_setosa_test <- sample(1:50,10)
rowID_versicol_test <- sample(51:100,10)
rowID_virgi_test <-sample(101:150,10)
# combine all the row IDs for test data
rowID_all_test <- c(rowID_setosa_test,rowID_versicol_test,rowID_virgi_test)
sort(rowID_all_test)
# keep just the test data points/rows
# We can use indexing to select some rows/records and all columns
# We want to remove the first column, which is the ID, because of its no usefulness.
all_test <- irs[rowID_all_test,-1]
dim(all_test)
# Now for the rest, keep them as the train data points
# We can use indexing to delete some rows/records and all columns
# Here a minus sign indicates a removal of selected rows/columns
# So herein -1 represents our not using 1st column.
all_train <- irs[-(rowID_all_test), -1]
dim(all_train)

# now use the kknn package
library(kknn)

# build the knn model
# Check the data structure
str(all_train)
str(all_test)
# first, change the "Species" variable to be of a factor type
all_train$Species <- as.factor(all_train$Species)
all_test$Species <- as.factor((all_test$Species))
# now, verify the change
str(all_train)
str(all_test)
# run a weighted k-NN
# The ~ is a separator, and the . means using all variables, except Species.
model_knn = train.kknn(Species ~ ., data=all_train, kmax=9)

# see the model's details
model_knn
summary(model_knn)

# Do a prediction on the test data
prediction <- predict(model_knn, all_test[, -5])
prediction

# Or here we can see the probabilities of the predictions
# by setting the type = "prob"
prediction_prob <- predict(model_knn, all_test[, -5], type='prob')
prediction_prob

# Furthermore, we can see in the prediction in another representation
prediction_2 <- as.character(as.numeric(prediction))
prediction_2

# See a confusion matrix. Each column of the matrix represents
# the number of predictions of each class, 
# while each row represents the instances in the actual class
CM <- table(all_test[, 5], prediction)
CM


# ------------ another way to call the kknn function ------------
model_knn_2 <- kknn(Species~., all_train, all_test)
model_knn_2
model_fit <- fitted(model_knn_2)
table(all_test$Species,model_fit)

# ---------------------------------------------------------------

#********************************************Recall********************************************
# calculate Sensitivity of Setosa type

# True positive rate
Setosa_Sensitivity = CM[1,1]/sum(CM[1,])
format(round(Setosa_Sensitivity,digits=2), nsmall = 2)

#********************************************Precision********************************************
Setosa_Precision = CM[1,1]/sum(CM[,1])
format(round(Setosa_Precision,digits=2), nsmall = 2)

#--------------------------------------------Specificity---------------------------------------
# True negative rate
TN <- CM[-1,-1]
Setosa_Specificity = sum(TN)/(sum(TN)+sum(CM[-1,1]))
format(round(Setosa_Specificity,digits=2), nsmall = 2)

#************************************Accuracy*******************

# See the overall accuracy of the model (for all Iris types)
accuracy <- (sum(diag(CM)))/sum(CM)
round(accuracy,2)


# now let's try to query for the classification of a given input
# make up a query first
query <- irs[141,-1]
# slightly adjust its petal length and width
query$PetalLengthCm <- 3.1
query$PetalWidthCm <- 1.45
# now try to predict what class it is for this new changed data point
prediction <- predict(model_knn, query[1, -5])
prediction



# ---------------------Extra-----------------------
# Study this part in your free time.

# Try another way to do k-NN with another dataset
# Your 2nd dataset
# Use German credit risk dataset
# Source: https://datahub.io/machine-learning/credit-g#resource-credit-g_arff

cred <- read.csv("data/credit-g_csv.csv")
# check its dimension
dim(cred)

# split the data into training and testing datasets
train <- cred[1:800,]
test <- cred[801:1000,]

# build the knn model
# first, change the "class" variable to be factor type
train$class <- as.factor(train$class)

# try either one of the follows, with different kernels
#model_knn = kknn(class ~ ., train,test) 
model_knn = kknn(class ~ ., train,test,kernel='gaussian')

# see the model's details
model_knn
summary(model_knn)

# try to fit the model
fit <- fitted(model_knn)
# see the confusion matrix
# in fact, you can compute the accuracy from it.
CM <- table(test$class, fit)
CM
acc <- sum(diag(CM))/sum(CM)
acc

# make prediction
# don't forget to set the type = "prob"
test$class <-as.factor(test$class)
prd <- predict(model_knn,test, type = "prob")
prd


# then apply a threshold to convert these probabilities into class labels
# note that there are two classes: bad and good; we need to select
# the one with higher probability
pre_labels <- ifelse(prd[,1] < prd[,2],"good","bad")

# show the accuracy
# the value should be the same as found from the above confusion matrix
table(pre_labels == test$class)/length(test$class)

# ------------------- END -----------------------





