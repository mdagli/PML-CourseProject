PRACTICAL MACHINE LEARNING PROJECT: HUMAN ACTIVITY RECOGNITION 
========================================================
Six young health participants wearing an accelerometer were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different ways: 
(Class A) exactly according to the specification 
(Class B) throwing the elbows to the front 
(Class C) lifting the dumbbell only halfway 
(Class D) lowering the dumbbell only halfway 
(Class E) throwing the hips to the front

**PROJECT GOALS: **

1. Using the accelerometer data build a predictive classification model which predicts the manner in which they did the exercise ("classe" variable in the training set). 
2. Use the prediction model to predict 20 different test cases. 
3. Create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. 

**STEP 1: Load data**

Read in the data files and load relevant libraries. 
Replace blank values and "#DIV/0!" values with NA.


```r
        library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
        traindata<-read.csv("pml-training.csv", na.strings=c("NA", "#DIV/0!",""))
        testdata<-read.csv("pml-testing.csv", na.strings=c("NA", "#DIV/0!",""))
```

**STEP 2: Clean and process the data**

First remove the variables that have more than 50% NA observations (in reality about 100 of the variables have almost all of the observations missing).  This reduces the number of independent variables from 159 to 59. 

```r
        missing=colSums(is.na(traindata))>10000
        traindata2=traindata[,missing==FALSE]
        testdata2=testdata[,missing==FALSE]
```
Next, remove the variables that clearly don't have anything to do with accelerometer data such as subject number, names etc. This reduces the number of independent variables to 53

```r
        traindata2=traindata2[,8:60]
        testdata2=testdata2[,8:60]
```
**STEP 3: Divide data into training and validation sets**  

The training set will only contain 25% of the observations in order to minimize computational time for this exercise. In a real world situation we would use a greater portion of the data in the training set in order to decrease the variance of the model estimates. 


```r
        set.seed(1321)
        intrain<-createDataPartition(y=traindata2$classe, p=0.25, list=FALSE)
        trainsubset<-traindata2[intrain,]
        trainvalidation<-traindata2[-intrain,]
```

**STEP 4: Model 1 creation: CART model**  


```r
        modCART<-train(classe~., method="rpart", data=trainsubset)
```

```
## Loading required package: rpart
```

```
## Warning: package 'rpart' was built under R version 3.1.2
```

**STEP 5: Test the accuracy of the CART model on the validation set**

```r
        predCART<-predict(modCART, newdata=trainvalidation)
        CARTconfusionmatrix<-confusionMatrix(predCART, trainvalidation$classe)
        CARTaccuracy<-CARTconfusionmatrix$overall[1]
```
The CART model estimated out-of-sample accuracy is 0.4958 
This does not seem to be very good.  So next we will try to build a prediction model using Random Forests. 

**STEP 6: Model 2 creation: Random Forest model**

```r
        rfModel<-train(classe~., data=trainsubset, tuneGrid=data.frame(mtry=3), trControl=trainControl(method="none"))
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

**STEP 7: Test the accuracy of the RF model on the validation set (0.98)**

```r
        predRF<-predict(rfModel, newdata=trainvalidation)
        RFconfusionmatrix<-confusionMatrix(predRF, trainvalidation$classe)
        RFaccuracy<-RFconfusionmatrix$overall[1]
```
The RF model estimated out-of-sample accuracy using the validation set is 0.98
The accuracy of this model seems to be very acceptable.  Thus, we will use the RF model to classify the test set cases. 

**STEP 8: Use the Random Forest model to classify the twenty Test Set cases**

```r
        predRFtest<-predict(rfModel, newdata=testdata2)
        predictions<-data.frame(testdata2$problem_id,predRFtest)
        colnames(predictions)<-c("problem_id", "PredictedClass")
        predictions
```

```
##    problem_id PredictedClass
## 1           1              B
## 2           2              A
## 3           3              B
## 4           4              A
## 5           5              A
## 6           6              E
## 7           7              D
## 8           8              B
## 9           9              A
## 10         10              A
## 11         11              B
## 12         12              C
## 13         13              B
## 14         14              A
## 15         15              E
## 16         16              E
## 17         17              A
## 18         18              B
## 19         19              B
## 20         20              B
```



