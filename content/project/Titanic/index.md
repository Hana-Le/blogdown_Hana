---
title: "Titanic Survival Prediction"
subtitle: "Kaggle competitions"
excerpt: "Using machine learning to create a model that predicts which passengers survived the Titanic shipwreck."
date: "2023-04-05"
author: "Hana Lê"
featured: true
draft: false
tags:
- R
- Kaggle Competition
categories:
- EDA
- Binary Classification 
- Machine learning
# layout options: single or single-sidebar
layout: single
links:
- icon: github
  icon_pack: fab
  name: code
  url: https://github.com/Hana-Le/Titanic-Survival-Prediction_R
---


---

{{< figure src="featured.png" alt="Traditional right sidebar layout" caption="Bettmann//Getty Images" >}}

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

This project uses data from the Kaggle competition [“Titanic - Machine Learning from Disater”](https://www.kaggle.com/competitions/titanic/overview). The goal of this project is to use machine learning to build a predictive model that predicts which passengers survived the Titanic shipwreck, to answers the question: “what sorts of people were more likely to survive?” using passenger data (ie. name, age, gender, socio-economic class, etc).

It seems like a straightforward project that is quite popular among beginners in data science :smiley:, but my objective is to gain more experience working with this type of dataset.

# 2. Overview the data

## 2.1 Loading packages and reading the data

``` r
# Loading R packages
packages <- c("tidyverse","Amelia", "psych","DT","mice","ranger", "janitor","vcd","kableExtra", "stringr","ggplot2", "GGally", "randomForest") 
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
# Visualization of missing values 
data[data ==""] <- NA
missmap(data[,colnames(data) != "Survived"])
```

<img src="/project/Titanic/Titanic_Survival_files/figure-html/unnamed-chunk-6-1.png" width="672" />

## 2.4 Imputing missing data

``` r
# Imputing missing values for Embarked
data[is.na(data$Embarked), ]
```

    ##     PassengerId Survived Pclass                                      Name
    ## 62           62        1      1                       Icard, Miss. Amelie
    ## 830         830        1      1 Stone, Mrs. George Nelson (Martha Evelyn)
    ##        Sex Age SibSp Parch Ticket Fare Cabin Embarked
    ## 62  female  38     0     0 113572   80   B28     <NA>
    ## 830 female  62     0     0 113572   80   B28     <NA>

``` r
data %>% tabyl(Embarked, Pclass) %>%
  kable() %>%
  kable_styling(full_width = FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Embarked
</th>
<th style="text-align:right;">
1
</th>
<th style="text-align:right;">
2
</th>
<th style="text-align:right;">
3
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
C
</td>
<td style="text-align:right;">
141
</td>
<td style="text-align:right;">
28
</td>
<td style="text-align:right;">
101
</td>
</tr>
<tr>
<td style="text-align:left;">
Q
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
113
</td>
</tr>
<tr>
<td style="text-align:left;">
S
</td>
<td style="text-align:right;">
177
</td>
<td style="text-align:right;">
242
</td>
<td style="text-align:right;">
495
</td>
</tr>
<tr>
<td style="text-align:left;">
NA
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
</tbody>
</table>

``` r
#names(sort(table(data$Embarked), decreasing = T))[1]
# These 2 passengers were in 1st class, in the same Cabin
# It seems likely the 2 passengers embarked from S = Southampton.
data$Embarked[is.na(data$Embarked)] <- "S"
```

``` r
# Imputing the missing value for Fare
data[is.na(data$Fare),]
```

    ##      PassengerId Survived Pclass               Name  Sex  Age SibSp Parch
    ## 1044        1044     <NA>      3 Storey, Mr. Thomas male 60.5     0     0
    ##      Ticket Fare Cabin Embarked
    ## 1044   3701   NA  <NA>        S

``` r
## the passenger was in Pclass 3 and embarked from "S"
data$Fare[is.na(data$Fare)] <- median(data$Fare, data$Pclass == 3 & data$Embarked == "S", na.rm = T)
```

``` r
# I was wondering if there is any case where Cabin is exclusively for the 1st class passengers?
# Have look on Cabin

#data %>% tabyl(Pclass, Cabin)
```

It appears that passengers who had cabin information were predominantly in the 1st class. These cabins mostly contain letters from A to E.

According to [‘Wikipedia’](https://en.wikipedia.org/wiki/First-class_facilities_of_the_Titanic) and [‘Titanic fandom’](https://titanic.fandom.com/wiki/Third_Class_cabins), first-class facilities and accommodation was located on the upper decks within the superstructure of the Titanic, which occupied almost the entirety of B and C Decks, but also large sections forward on A, D and E-Decks. While the third-class cabins were located on F Deck, with a few on the forward G Deck. Rather than numbered by the deck they were on, these cabins were numbered separately. This area was the first to flood during the sinking, because of their location in the lowest decks in the bow. It is possible that the location of the accommodation would have been affecting the chance of survival of passengers on the ship.

It is quite obvious to see significant association between Pclass with Cabin labeled from A to E by using Chi-Squared test in this case.

``` r
data$Cabin2 <- ifelse(is.na(data$Cabin), 0, 1)
Pclass_Cabin2 <- table(data$Pclass, data$Cabin2)
chisq.test(Pclass_Cabin2)
```

    ## 
    ## 	Pearson's Chi-squared test
    ## 
    ## data:  Pclass_Cabin2
    ## X-squared = 794.43, df = 2, p-value < 2.2e-16

So I decided to drop the cabin variable from the analysis

``` r
data$Cabin <- NULL
```

``` r
# Imputing missing values for Age
# I'm going to use mice package for Age
# select only the columns needed for imputing Age
age_data <- data[, c('Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')]

# impute missing values using MICE with the "random forest" method
set.seed(1234)
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
    ##       Parch      Ticket        Fare    Embarked      Cabin2 
    ##           0           0           0           0           0

## 2.5 Descriptive statistics

``` r
data_table <- describe(data)
data_table %>% round(digits = 3) %>%
  DT::datatable(options = list(pageLength = 10)) 
```

<div class="datatables html-widget html-fill-item-overflow-hidden html-fill-item" id="htmlwidget-1" style="width:100%;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"filter":"none","vertical":false,"data":[["PassengerId","Survived*","Pclass*","Name*","Sex*","Age","SibSp","Parch","Ticket*","Fare","Embarked*","Cabin2"],[1,2,3,4,5,6,7,8,9,10,11,12],[1309,891,1309,1309,1309,1309,1309,1309,1309,1309,1309,1309],[655,1.384,2.295,653.694,1.644,29.503,0.499,0.385,464.604,33.281,3.494,0.225],[378.02,0.487,0.838,377.31,0.479,14.17,1.042,0.866,278.039,51.741,0.814,0.418],[655,1,3,653,2,28,0,0,460,14.454,4,0],[655,1.355,2.368,653.619,1.68,28.943,0.275,0.175,465.234,21.568,3.616,0.157],[484.81,0,0,484.81,0,11.861,0,0,379.546,10.236,0,0],[1,1,1,1,1,0.17,0,0,1,0,2,0],[1309,2,3,1307,2,80,8,9,929,512.329,4,1],[1308,1,2,1306,1,79.83,8,9,928,512.329,2,1],[0,0.477,-0.597,0.002,-0.601,0.472,3.835,3.661,-0.009,4.36,-1.125,1.313],[-1.203,-1.775,-1.317,-1.202,-1.64,0.312,19.927,21.417,-1.328,26.896,-0.548,-0.276],[10.448,0.016,0.023,10.429,0.013,0.392,0.029,0.024,7.685,1.43,0.023,0.012]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th> <\/th>\n      <th>vars<\/th>\n      <th>n<\/th>\n      <th>mean<\/th>\n      <th>sd<\/th>\n      <th>median<\/th>\n      <th>trimmed<\/th>\n      <th>mad<\/th>\n      <th>min<\/th>\n      <th>max<\/th>\n      <th>range<\/th>\n      <th>skew<\/th>\n      <th>kurtosis<\/th>\n      <th>se<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"pageLength":10,"columnDefs":[{"className":"dt-right","targets":[1,2,3,4,5,6,7,8,9,10,11,12,13]},{"orderable":false,"targets":0}],"order":[],"autoWidth":false,"orderClasses":false}},"evals":[],"jsHooks":[]}</script>

## 3. Exploring variables

### 3.1 Survived

``` r
data %>% 
  filter(!is.na(Survived)) %>%
  tabyl(Survived) %>%
  adorn_pct_formatting() %>%
  kable() %>%
  kable_styling(full_width = FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Survived
</th>
<th style="text-align:right;">
n
</th>
<th style="text-align:left;">
percent
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
0
</td>
<td style="text-align:right;">
549
</td>
<td style="text-align:left;">
61.6%
</td>
</tr>
<tr>
<td style="text-align:left;">
1
</td>
<td style="text-align:right;">
342
</td>
<td style="text-align:left;">
38.4%
</td>
</tr>
</tbody>
</table>

Out of the 891 passengers in the training set, 342 survived, which corresponds to a survival rate of approximately 38.4%.

### 3.2 Exploring potential predcitors of Survived

Passenger titles provide some information about their social status, gender, and possibly age which could be strong predictors for survival. These titles were included in the name of each passenger so we can extract them out.

``` r
data$Title <- sapply(data$Name, function(x) strsplit(x, split = '[,.]')[[1]][[2]])
data$Title <- sub(' ', '', data$Title)
data %>% tabyl(Title, Sex) %>%
  kable() %>%
  kable_styling(full_width = FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Title
</th>
<th style="text-align:right;">
female
</th>
<th style="text-align:right;">
male
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Capt
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
Col
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
4
</td>
</tr>
<tr>
<td style="text-align:left;">
Don
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
Dona
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Dr
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
7
</td>
</tr>
<tr>
<td style="text-align:left;">
Jonkheer
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
Lady
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Major
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
2
</td>
</tr>
<tr>
<td style="text-align:left;">
Master
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
61
</td>
</tr>
<tr>
<td style="text-align:left;">
Miss
</td>
<td style="text-align:right;">
260
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Mlle
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Mme
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Mr
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
757
</td>
</tr>
<tr>
<td style="text-align:left;">
Mrs
</td>
<td style="text-align:right;">
197
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Ms
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Rev
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
8
</td>
</tr>
<tr>
<td style="text-align:left;">
Sir
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
the Countess
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
</tr>
</tbody>
</table>

``` r
data <- data %>% 
  mutate(Title = case_when(
    Title %in% c("Mlle", "Ms") ~ "Miss",
    Title == "Mme" ~ "Mrs",
    Title %in% c("the Countess", "Dona", "Lady", "Jonkheer") ~ "Noble",
    Title %in% c("Capt","Col", "Dr", "Rev", "Don", "Major", "Sir") ~ "Officer",
    TRUE ~ Title
  ))

data %>% tabyl(Title, Sex) %>%
  kable() %>%
  kable_styling(full_width = FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Title
</th>
<th style="text-align:right;">
female
</th>
<th style="text-align:right;">
male
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Master
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
61
</td>
</tr>
<tr>
<td style="text-align:left;">
Miss
</td>
<td style="text-align:right;">
264
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Mr
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
757
</td>
</tr>
<tr>
<td style="text-align:left;">
Mrs
</td>
<td style="text-align:right;">
198
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
Noble
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
Officer
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
24
</td>
</tr>
</tbody>
</table>

### 3.2.1 Potential continuous predictors of Survived

``` r
sub_con <- subset(data[!is.na(data$Survived),], select = c("Survived", "Age", "Fare"))

# Create scatter plot matrix with ggally
ggpairs(sub_con, aes(color = Survived, alpha = 0.7),
        lower = list(combo = "count"))
```

<img src="/project/Titanic/Titanic_Survival_files/figure-html/unnamed-chunk-18-1.png" width="672" />

The median age of the survivors appears to be slightly higher than that of non-survivors, indicating that older individuals had a higher chance of survival. On the other hand, the median fare of the surviving passengers is noticeably higher, suggesting a potential correlation between fare paid and survival rate.

### 3.2.2 Age categories

I read somewhere that priority to secure a place on lifeboats was given to “Women and children first”. So younger passenger should have better surviving odds.

I’m going to create age_cate to check is there was an association between age_cat and Survived.

``` r
data$age_cat <- cut(data$Age, breaks = c(0,18,40,60,95),
                    label = c("under18","middle", "senior", "elder"))
```

``` r
data %>%
  filter(!is.na(Survived)) %>%
  tabyl(age_cat, Survived) %>%
  adorn_percentages("row") %>% 
  adorn_pct_formatting(rounding = "half up", digits = 0) %>%
  adorn_ns(position = "front") %>%
  knitr::kable()
```

<table>
<thead>
<tr>
<th style="text-align:left;">
age_cat
</th>
<th style="text-align:left;">
0
</th>
<th style="text-align:left;">
1
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
under18
</td>
<td style="text-align:left;">
85 (51%)
</td>
<td style="text-align:left;">
81 (49%)
</td>
</tr>
<tr>
<td style="text-align:left;">
middle
</td>
<td style="text-align:left;">
354 (65%)
</td>
<td style="text-align:left;">
193 (35%)
</td>
</tr>
<tr>
<td style="text-align:left;">
senior
</td>
<td style="text-align:left;">
90 (60%)
</td>
<td style="text-align:left;">
61 (40%)
</td>
</tr>
<tr>
<td style="text-align:left;">
elder
</td>
<td style="text-align:left;">
20 (74%)
</td>
<td style="text-align:left;">
7 (26%)
</td>
</tr>
</tbody>
</table>

``` r
#data %>%
  #filter(!is.na(Survived)) %>%
 #chisq.test(x = .$Survived, y =.$age_cat, correct = FALSE)

mosaic(~ age_cat + Survived, data = data[!is.na(data$Survived),], gp = shading_max, rot_labels = 0)
```

<img src="/project/Titanic/Titanic_Survival_files/figure-html/unnamed-chunk-20-1.png" width="672" />

The association is significant (p-value \< 0.05). Passengers who were under 18 had higher survived rate than other groups, whereas the elder had the lowest likelihood of surviving.
\### 3.2.3 Sex

``` r
data %>% 
  filter(!is.na(Survived)) %>%
  tabyl(Sex, Survived) %>%
  adorn_percentages("row") %>% 
  adorn_pct_formatting(rounding = "half up", digits = 0) %>%
  adorn_ns(position = "front") %>%
  knitr::kable()
```

<table>
<thead>
<tr>
<th style="text-align:left;">
Sex
</th>
<th style="text-align:left;">
0
</th>
<th style="text-align:left;">
1
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
female
</td>
<td style="text-align:left;">
81 (26%)
</td>
<td style="text-align:left;">
233 (74%)
</td>
</tr>
<tr>
<td style="text-align:left;">
male
</td>
<td style="text-align:left;">
468 (81%)
</td>
<td style="text-align:left;">
109 (19%)
</td>
</tr>
</tbody>
</table>

``` r
#chisq.test(data$Survived, data$Sex, correct = FALSE)

mosaic(~ Sex + Survived, data = data[!is.na(data$Survived),], gp = shading_max, rot_labels=0)
```

<img src="/project/Titanic/Titanic_Survival_files/figure-html/unnamed-chunk-21-1.png" width="672" />

Female passengers had a noticeably higher rate of survival when compared to male passengers( p-value \< 0.001).

``` r
data %>% 
  filter(!is.na(Survived)) %>%
  tabyl(Title, Survived) %>%
  adorn_percentages("row") %>% 
  adorn_pct_formatting(rounding = "half up", digits = 0) %>%
  adorn_ns(position = "front") %>%
  knitr::kable()
```

<table>
<thead>
<tr>
<th style="text-align:left;">
Title
</th>
<th style="text-align:left;">
0
</th>
<th style="text-align:left;">
1
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Master
</td>
<td style="text-align:left;">
17 (43%)
</td>
<td style="text-align:left;">
23 (58%)
</td>
</tr>
<tr>
<td style="text-align:left;">
Miss
</td>
<td style="text-align:left;">
55 (30%)
</td>
<td style="text-align:left;">
130 (70%)
</td>
</tr>
<tr>
<td style="text-align:left;">
Mr
</td>
<td style="text-align:left;">
436 (84%)
</td>
<td style="text-align:left;">
81 (16%)
</td>
</tr>
<tr>
<td style="text-align:left;">
Mrs
</td>
<td style="text-align:left;">
26 (21%)
</td>
<td style="text-align:left;">
100 (79%)
</td>
</tr>
<tr>
<td style="text-align:left;">
Noble
</td>
<td style="text-align:left;">
1 (33%)
</td>
<td style="text-align:left;">
2 (67%)
</td>
</tr>
<tr>
<td style="text-align:left;">
Officer
</td>
<td style="text-align:left;">
14 (70%)
</td>
<td style="text-align:left;">
6 (30%)
</td>
</tr>
</tbody>
</table>

``` r
mosaic(~ Title + Survived, data = data[!is.na(data$Survived),], gp = shading_max, rot_labels=0)
```

<img src="/project/Titanic/Titanic_Survival_files/figure-html/unnamed-chunk-22-1.png" width="672" />

The title that a passenger held had an impact on their odds of survival (p-value \< 0.001). In line with the previous observation, it appears that women with titles of Miss or Mrs had a higher likelihood of surviving. Additionally, it is noteworthy that passengers with titles of Master or Noble had a better chance of survival when compared to those with titles of Mr or Officer.

## 4. Data preparation for modelling

## 5. Modelling

## 6. Conclusion
