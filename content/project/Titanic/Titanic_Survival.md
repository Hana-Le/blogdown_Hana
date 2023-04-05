---
title: "Titanic Survival"
author: "Hana Le"
date: "2023-04-05"
output: 
 html_document:
    keep_md: true
---

<script src="/rmarkdown-libs/htmlwidgets/htmlwidgets.js"></script>
<link href="/rmarkdown-libs/datatables-css/datatables-crosstalk.css" rel="stylesheet" />
<script src="/rmarkdown-libs/datatables-binding/datatables.js"></script>
<script src="/rmarkdown-libs/jquery/jquery-3.6.0.min.js"></script>
<link href="/rmarkdown-libs/dt-core/css/jquery.dataTables.min.css" rel="stylesheet" />
<link href="/rmarkdown-libs/dt-core/css/jquery.dataTables.extra.css" rel="stylesheet" />
<script src="/rmarkdown-libs/dt-core/js/jquery.dataTables.min.js"></script>
<link href="/rmarkdown-libs/crosstalk/css/crosstalk.min.css" rel="stylesheet" />
<script src="/rmarkdown-libs/crosstalk/js/crosstalk.min.js"></script>

# 1. Introduction

The Titanic sinking is famous - it was a new ship that people believed couldn’t sink, but it hit an iceberg on April 15, 1912, and sank. Sadly, not everyone had a lifeboat, and 1502 people died out of 2224 passengers and crew. Although chance played a role, it has been believed that some groups had a better chance of surviving.

This project uses data from the Kaggle competition [“Titanic - Machine Learning from Disater”](https://www.kaggle.com/competitions/titanic/overview). The goal of this project is to use machine learning to build a predictive model that predicts which passengers survived the Titanic shipwreck answers the question: “what sorts of people were more likely to survive?” using passenger data (ie. name, age, gender, socio-economic class, etc) and evaluate the performance of the model using the test data. It looks like a simple project but I aim to learn more about ML and also gain some more knowledge thru analysing the data.

# 2. Overview the data

## 2.1 Loading packages and reading the data

``` r
# Loading R packages
packages <- c("tidyverse","Amelia", "psych","DT", "htmlTable","mice","ranger", "janitor", "randomForest") 
sapply(packages, require, character = TRUE)
```

``` r
# Reading data
train <- read.csv("titanic_data/train.csv")
test <- read.csv("titanic_data/test.csv")
```

## 2.2 Data size

The Titanic train data set has 891 obs and 12 variables with the response variable Survived. The Titanic test data set has 418 obs and 11 variables.

``` r
dim(train); dim(test)
```

    ## [1] 891  12

    ## [1] 418  11

``` r
# Combine 2 data sets to see the structure, and for cleaning & feature engineering later.
test$Survived <- NA
data <- rbind(train, test)
dim(data) 
```

    ## [1] 1309   12

The data now has 12 columns consisting of 11 predictors and the response variable Survived.

## 2.3 Data Structure

``` r
str(data)
```

    ## 'data.frame':	1309 obs. of  12 variables:
    ##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
    ##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
    ##  $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
    ##  $ Sex        : chr  "male" "female" "female" "female" ...
    ##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
    ##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
    ##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
    ##  $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
    ##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
    ##  $ Cabin      : chr  "" "C85" "" "C123" ...
    ##  $ Embarked   : chr  "S" "C" "S" "S" ...

Some categorical variables should be in factor form:

``` r
var_cat <- c("Survived", "Pclass","Sex", "Embarked")
data[, var_cat] <- data.frame(lapply(data[, var_cat], as.factor))
```

## 2.3 Missingness of the data

It seems from the data structure that missing values are not only represented as ‘NA’, but also as empty.

``` r
sort(colSums(is.na(data)| data == ""), decreasing = T)
```

    ##       Cabin    Survived         Age    Embarked        Fare PassengerId 
    ##        1014         418         263           2           1           0 
    ##      Pclass        Name         Sex       SibSp       Parch      Ticket 
    ##           0           0           0           0           0           0

Out of a total of 1309 records, Cabin is the variable with the highest number of missing values, with 1014 records (77.5%) missing. Age is the second variable with the most missing values, with 263 records (20.1%) missing. Embarked has only 2 missing values, while Fare has only 1 missing value.

``` r
data[data ==""] <- NA

missmap(data[,colnames(data) != "Survived"])
```

<img src="/project/Titanic/Titanic_Survival_files/figure-html/unnamed-chunk-6-1.png" width="672" />

## 2.4 Imputing missing data

``` r
# I was wondering if there is any case where Cabin is exclusively for the 1st class passengers?
# Have look on Cabin

# data %>% tabyl(Pclass, Cabin)

# It appears that passengers who had cabin information were predominantly in the 1st class. 
# However, there were still 67 1st-class passengers who had missing values for the cabin,
# making it difficult to draw conclusions. 
# I decided to drop the cabin variable from the analysis

data$Cabin <- NULL
```

``` r
# Imputing missing values for Embarked
data[is.na(data$Embarked), ]
```

    ##     PassengerId Survived Pclass                                      Name
    ## 62           62        1      1                       Icard, Miss. Amelie
    ## 830         830        1      1 Stone, Mrs. George Nelson (Martha Evelyn)
    ##        Sex Age SibSp Parch Ticket Fare Embarked
    ## 62  female  38     0     0 113572   80     <NA>
    ## 830 female  62     0     0 113572   80     <NA>

``` r
data %>% tabyl(Embarked, Pclass)
```

    ##  Embarked   1   2   3
    ##             0   0   0
    ##         C 141  28 101
    ##         Q   3   7 113
    ##         S 177 242 495
    ##      <NA>   2   0   0

``` r
#names(sort(table(data$Embarked), decreasing = T))[1]
# It seems likely the 2 passengers who had Embarked info missing embarked from S = Southampton.
data$Embarked[is.na(data$Embarked)] <- "S"
```

``` r
#Imputing the missing value for Fare
data[is.na(data$Fare),]
```

    ##      PassengerId Survived Pclass               Name  Sex  Age SibSp Parch
    ## 1044        1044     <NA>      3 Storey, Mr. Thomas male 60.5     0     0
    ##      Ticket Fare Embarked
    ## 1044   3701   NA        S

``` r
data$Fare[is.na(data$Fare)] <- median(data$Fare, data$Pclass == 3 & data$Embarded == "S", na.rm = T)
```

``` r
# Imputing missing values for Age
# I'm going to use mice package for Age
# select only the columns needed for imputing Age
age_data <- data[, c('Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')]

# impute missing values using MICE with the "random forest" method

age_mice <- mice(age_data, method = "rf")
```

    ## 
    ##  iter imp variable
    ##   1   1  Age
    ##   1   2  Age
    ##   1   3  Age
    ##   1   4  Age
    ##   1   5  Age
    ##   2   1  Age
    ##   2   2  Age
    ##   2   3  Age
    ##   2   4  Age
    ##   2   5  Age
    ##   3   1  Age
    ##   3   2  Age
    ##   3   3  Age
    ##   3   4  Age
    ##   3   5  Age
    ##   4   1  Age
    ##   4   2  Age
    ##   4   3  Age
    ##   4   4  Age
    ##   4   5  Age
    ##   5   1  Age
    ##   5   2  Age
    ##   5   3  Age
    ##   5   4  Age
    ##   5   5  Age

``` r
age_imputed <- complete(age_mice)
data$Age <- age_imputed$Age
```

``` r
# Checking missing values
colSums(is.na(data[,colnames(data) != "Survived"]))
```

    ## PassengerId      Pclass        Name         Sex         Age       SibSp 
    ##           0           0           0           0           0           0 
    ##       Parch      Ticket        Fare    Embarked 
    ##           0           0           0           0

## 2.5 Descriptive statistics

``` r
data_table <- describe(data)
data_table %>% round(digits = 3) %>%
  DT::datatable(options = list(pageLength = 10)) 
```

<div class="datatables html-widget html-fill-item-overflow-hidden html-fill-item" id="htmlwidget-1" style="width:100%;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"filter":"none","vertical":false,"data":[["PassengerId","Survived*","Pclass*","Name*","Sex*","Age","SibSp","Parch","Ticket*","Fare","Embarked*"],[1,2,3,4,5,6,7,8,9,10,11],[1309,891,1309,1309,1309,1309,1309,1309,1309,1309,1309],[655,1.384,2.295,653.694,1.644,29.654,0.499,0.385,464.604,33.281,3.494],[378.02,0.487,0.838,377.31,0.479,14.027,1.042,0.866,278.039,51.741,0.814],[655,1,3,653,2,28,0,0,460,14.454,4],[655,1.355,2.368,653.619,1.68,29.101,0.275,0.175,465.234,21.568,3.616],[484.81,0,0,484.81,0,11.861,0,0,379.546,10.236,0],[1,1,1,1,1,0.17,0,0,1,0,2],[1309,2,3,1307,2,80,8,9,929,512.329,4],[1308,1,2,1306,1,79.83,8,9,928,512.329,2],[0,0.477,-0.597,0.002,-0.601,0.455,3.835,3.661,-0.009,4.36,-1.125],[-1.203,-1.775,-1.317,-1.202,-1.64,0.292,19.927,21.417,-1.328,26.896,-0.548],[10.448,0.016,0.023,10.429,0.013,0.388,0.029,0.024,7.685,1.43,0.023]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th> <\/th>\n      <th>vars<\/th>\n      <th>n<\/th>\n      <th>mean<\/th>\n      <th>sd<\/th>\n      <th>median<\/th>\n      <th>trimmed<\/th>\n      <th>mad<\/th>\n      <th>min<\/th>\n      <th>max<\/th>\n      <th>range<\/th>\n      <th>skew<\/th>\n      <th>kurtosis<\/th>\n      <th>se<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"pageLength":10,"columnDefs":[{"className":"dt-right","targets":[1,2,3,4,5,6,7,8,9,10,11,12,13]},{"orderable":false,"targets":0}],"order":[],"autoWidth":false,"orderClasses":false}},"evals":[],"jsHooks":[]}</script>

## 3. Feature Engineering

Passenger titles provide some information about their social status, gender, and possibly age. These titles were included in the name of each passenger so we can extract them out.

``` r
data$Title <- sapply(data$Name, function(x) strsplit(x, split = '[,.]')[[1]][[2]])
data$Title <- sub(' ', '', data$Title)
data %>% tabyl(Title, Sex) 
```

    ##         Title female male
    ##          Capt      0    1
    ##           Col      0    4
    ##           Don      0    1
    ##          Dona      1    0
    ##            Dr      1    7
    ##      Jonkheer      0    1
    ##          Lady      1    0
    ##         Major      0    2
    ##        Master      0   61
    ##          Miss    260    0
    ##          Mlle      2    0
    ##           Mme      1    0
    ##            Mr      0  757
    ##           Mrs    197    0
    ##            Ms      2    0
    ##           Rev      0    8
    ##           Sir      0    1
    ##  the Countess      1    0

``` r
data$Title[data$Title %in% c("Mlle", "Ms")] <- "Miss"
data$Title[data$Title == "Mme"] <- "Mrs"
data$Title[data$Title %in% c("the Countess", "Dona", "Lady", "Jonkheer")] <- "Noble"
data$Title[data$Title %in% c("Capt","Col", "Dr", "Rev", "Don", "Major", "Sir")] <- "Officer"
data %>% tabyl(Title, Sex) 
```

    ##    Title female male
    ##   Master      0   61
    ##     Miss    264    0
    ##       Mr      0  757
    ##      Mrs    198    0
    ##    Noble      3    1
    ##  Officer      1   24

## 4. Exploring variables

## 5. Data preparation for modelling

## 6. Modelling

## 7. Conclusion
