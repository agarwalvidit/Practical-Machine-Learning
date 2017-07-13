A. Overview
-----------

    This report explores the relationship of how multiple metrics relate with different effectiveness level of an exercise performed.In its due emphasis is laid on identifying important variables and subsequently a machine learning model has been build to predict exercise effectiveness based on test sample.  

B. Brief Background
-------------------

    There has been recent surge in monitoring devices which measures personal activity bases on reading of various sensors e.g. accerelometers, gyroscopes. Given complexity of various human movements, it entails measurement of each activity across multiple metrics. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
      In this project, we will be using data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants to see if we can help gauge whether a particular exercise i.e  barbell in this case, if being lifted correctly or incorrectly in 5 different ways
      

C. Data Exploration
-------------------

### i. Dataset:

Full data for this project is available at <http://groupware.les.inf.puc-rio.br/har>.

**Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. “Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13)”. Stuttgart, Germany: ACM SIGCHI, 2013.**

A special thanks to authors of this data for allowing its usage for this academic assignment

**The training data**
**Link**: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

**The test data**
**Link** : <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

### ii. Loading Data

``` r
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(corrplot))

url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
filename1 <- "pml-training.csv"
filename2 <- "pml-testing.csv"
download.file(url=url1, destfile=filename1,method="curl")
download.file(url=url2, destfile=filename2,method="curl")
training_raw <- read.csv("pml-training.csv", sep = ",", header = TRUE, na.strings = c("", "NA"))
testing_raw <- read.csv("pml-testing.csv", sep = ",", header = TRUE, na.strings =c("","NA"))
```

### iii. Stucture of data

``` r
dim(training_raw)
```

    ## [1] 19622   160

``` r
dim(testing_raw)
```

    ## [1]  20 160

### iv. Cleaning of dataset

*On checking variables in dataset, we see that test dataset have variable "problem\_id" which is not present in train. lets see it content*

``` r
unique(training_raw$classe)
```

    ## [1] A B C D E
    ## Levels: A B C D E

``` r
unique(testing_raw$problem_id)
```

    ##  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20

*Clearly it is different from classe variable in training set. So, we have to remove it as we cannot predict based on new variable not defined in prediction model ( sourced through train dataset)*

``` r
testing_raw <- testing_raw[ , -(which(names(testing_raw) == "problem_id"))]
```

Both data-sets have numerous variables i.e 160 variables. to check if we can trim, we will run 2 process.
1. Removing variables with less influence on result owing to very low variability in the measurements.
2. Removing variables with large number of NA variables.

``` r
# removing variables with very low variance
nsv <- nearZeroVar(training_raw,saveMetrics=TRUE)
zerovar <- names(training_raw[, nsv$nzv])

training_v1 <- training_raw[,!nsv$nzv]
testing_v1 <- testing_raw[, -which(names(testing_raw) %in% zerovar)]
```

``` r
#removing variables with large NAs
na_filter    <- sapply(training_v1, function(x) mean(is.na(x))) > 0.95
training_v2 <- training_v1[, na_filter==FALSE]
testing_v2  <- testing_v1[, which(names(testing_v1) %in% names(training_v2))]
sum(is.na(training_v2))
```

    ## [1] 0

*Final values shows there are no more NA in trainingdata set*

*We also observed that 1st 6 variables capture reduntant information from exercise point of view. We will remove them as well*

``` r
training_v3 <- training_v2[,-(1:6)]
testing_v3 <- testing_v2[, -(1:6)]
dim(training_v3)
```

    ## [1] 19622    53

With the cleaning process above, the number of variables for the analysis has been reduced to 53 only.

D. Variable relationships
-------------------------

Lets explore correlation among variables before proceeding further

``` r
corMatrix <- cor(training_v3[, -ncol(training_v3)])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

<img src="Practical_Machine_Learning_-_Assignment_files/figure-markdown_github/unnamed-chunk-8-1.png" style="display: block; margin: auto;" /> Since correlations are quite few in nature, we could avoid PCA preprocessing and directly explore modelling

D. Model Building:
------------------

Though random forest method will most probably yield better result, for better confidence in our model we will try to check accuracy for other model methods as well. But first lets split training-set it into
1. training subset
2. validation subset

``` r
inTrain <- createDataPartition(y=training_v3$classe, p=0.7, list=FALSE)
training_clean <- training_v3[inTrain,]
validation_clean <- training_v3[-inTrain,]
```

**Method 1**: **Random forest**

``` r
set.seed(123)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modelRF <- train(classe ~ ., data=training_clean, method="rf",trControl=controlRF, message = FALSE)
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
validation_pred_RF <- predict(modelRF, newdata = validation_clean)
confusionMatrix(validation_pred_RF,validation_clean$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    9    0    0    0
    ##          B    1 1129    2    0    0
    ##          C    1    1 1020    9    0
    ##          D    0    0    4  955    3
    ##          E    0    0    0    0 1079
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9949          
    ##                  95% CI : (0.9927, 0.9966)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9936          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9912   0.9942   0.9907   0.9972
    ## Specificity            0.9979   0.9994   0.9977   0.9986   1.0000
    ## Pos Pred Value         0.9946   0.9973   0.9893   0.9927   1.0000
    ## Neg Pred Value         0.9995   0.9979   0.9988   0.9982   0.9994
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1918   0.1733   0.1623   0.1833
    ## Detection Prevalence   0.2856   0.1924   0.1752   0.1635   0.1833
    ## Balanced Accuracy      0.9983   0.9953   0.9959   0.9946   0.9986

Accuracy = **99.3%**.
Lets check the accuracy through another method

**Method 1**: **General boosting method**

``` r
controlGBM <- trainControl(method="repeatedcv", number=5, repeats = 1)
modelGBM <- train(classe ~ ., data=training_clean, method="gbm",trControl=controlGBM, verbose = FALSE)
```

    ## Loading required package: gbm

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.3

    ## Loading required package: plyr

``` r
validation_pred_GBM <- predict(modelGBM, newdata = validation_clean)
confusionMatrix(validation_pred_GBM,validation_clean$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1642   42    0    1    1
    ##          B   23 1066   25    1   13
    ##          C    5   27  991   31    6
    ##          D    3    1    9  928   17
    ##          E    1    3    1    3 1045
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9638          
    ##                  95% CI : (0.9587, 0.9684)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9542          
    ##  Mcnemar's Test P-Value : 4.301e-06       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9809   0.9359   0.9659   0.9627   0.9658
    ## Specificity            0.9896   0.9869   0.9858   0.9939   0.9983
    ## Pos Pred Value         0.9739   0.9450   0.9349   0.9687   0.9924
    ## Neg Pred Value         0.9924   0.9847   0.9927   0.9927   0.9923
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2790   0.1811   0.1684   0.1577   0.1776
    ## Detection Prevalence   0.2865   0.1917   0.1801   0.1628   0.1789
    ## Balanced Accuracy      0.9852   0.9614   0.9758   0.9783   0.9821

Accuracy = **96.3%**.
*Random forest* model looks to be a better model with a very low out-of-sample error is **0.7%**. Also, we don’t need to worry about variables we excluded with this accuracy.

\*\* Performing check on importance of variables \*\*

``` r
varImpPlot(modelRF$finalModel, sort = TRUE, pch = 19, col = 1, cex = 1, main = "Importance of Predictors")
```

<img src="Practical_Machine_Learning_-_Assignment_files/figure-markdown_github/unnamed-chunk-12-1.png" style="display: block; margin: auto;" /> The top 4 most important variables according to the model fit are ‘roll\_belt’, ‘pitch\_forearm’, ‘yaw\_belt’ and ‘roll\_forearm’

E. Predicting on test sample :
------------------------------

``` r
test_pred<- predict( modelRF, newdata= testing_v3)
test_pred
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Results
-------

We used 52 variables to build the random forest model with 3-fold cross validation.
The out-of-sample error is approximately **0.7%**
