---
title: "House Prices Prediction- Advanced Regression Techniques"
subtitle: "Kaggle competitions"
excerpt: "This is a competition for data science beginners or machine learning students are looking to expand their skill set. The goal of this competition is to predict the sales price for a sample of houses. All the submissions are evaluated on RMSE between the logarithm of the predicted value and the logarithm of the observed sales price."
date: "2023-02-25"
author: "Hana Lê"
featured: true
draft: false
tags:
- R
- Kaggle Competition
categories:
- Linear regression
- Machine learning
# layout options: single or single-sidebar
layout: single
links:
- icon: github
  icon_pack: fab
  name: code
  url: https://github.com/Hana-Le/House-Price-Prediction_R
---


---

{{< figure src="featured.png" alt="Traditional right sidebar layout" caption="Photo by Phil Hearing on Unsplash" >}}

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

This project is based on the Kaggle competition ["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The goal of this project is to predict housing prices in based on the provided training data (train.csv) and evaluate the performance of the model using the test data (test.csv). Through this project, I aim to not only build a robust prediction model but also gain some knowledge and insights on data wrangling and analysis. 

# 2. Overview the data
## 2.1 Loading packages and reading the data


```r
# Loading R packages
packages <- c("tidyverse", "psych","DT", "gridExtra", "GGally", "corrplot", "ggcorrplot", "naniar", "visdat", "moments", "janitor","caret", "reshape2", "xgboost") 
sapply(packages, require, character = TRUE)
```


```r
# Reading data
train <- read.csv("housing_data/train.csv")
test <- read.csv("housing_data/test.csv")
```


## 2.2 Data size
The housing train data set has 1460 obs and 81 variables with the response variable Sale Price. The housing test data set has 1459 obs and 80 variables.


```r
dim(train) ; dim(test)
```

```
## [1] 1460   81
```

```
## [1] 1459   80
```



```r
# Combine 2 data sets to see the structure, and for cleaning & feature engineering later.
# Removing Id as not necessary but keeping the test Id for the final file.
test_labels <- test$Id
test$Id <- NULL
train$Id <- NULL
test$SalePrice <- NA
df <- rbind(train, test)
dim(df) 
```

```
## [1] 2919   80
```
The data now has 80 columns consisting of 79 predictors and reponse variable Sale price.

## 2.3 Missingness of the data

The dataset has 13965 missing values (exclude the missing values for Sale price in the test dataset), happens to be about 6%.


```r
n_miss(df[,colnames(df)!="SalePrice"])
```

```
## [1] 13965
```

```r
pct_miss(df[,colnames(df)!="SalePrice"])
```

```
## [1] 6.055915
```

```r
# Select columns with > 0 missing values
df_miss <- names(df[colSums(is.na(df[,colnames(df)!="SalePrice"])) > 0])
cat("There are", length(df_miss), "columns with missing values")
```

```
## There are 34 columns with missing values
```


```r
vis_miss(df[,df_miss], sort_miss = TRUE) # visualizing missing data
```

![](house_price_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

- The predictors having the most missing values which is about 50% or more are: PoolQC, MiscFeature, Alley, Fence, FireplaceQu. They are all categorical variables. As described in the data_description.txt file, the NA value reflects the houses didn't have these features. 
- Followed by LotFrontage (16.7%), Garage related (5.x%) and basement related variables (2.x%).

I will leave dropping/imputing missing values for later after exploring variables.

## 2.4 Data enginering

### 2.4.1 Data struture


```r
# Data structure 
str(df)
```

```
## 'data.frame':	2919 obs. of  80 variables:
##  $ MSSubClass   : int  60 20 60 70 60 50 20 60 50 190 ...
##  $ MSZoning     : chr  "RL" "RL" "RL" "RL" ...
##  $ LotFrontage  : int  65 80 68 60 84 85 75 NA 51 50 ...
##  $ LotArea      : int  8450 9600 11250 9550 14260 14115 10084 10382 6120 7420 ...
##  $ Street       : chr  "Pave" "Pave" "Pave" "Pave" ...
##  $ Alley        : chr  NA NA NA NA ...
##  $ LotShape     : chr  "Reg" "Reg" "IR1" "IR1" ...
##  $ LandContour  : chr  "Lvl" "Lvl" "Lvl" "Lvl" ...
##  $ Utilities    : chr  "AllPub" "AllPub" "AllPub" "AllPub" ...
##  $ LotConfig    : chr  "Inside" "FR2" "Inside" "Corner" ...
##  $ LandSlope    : chr  "Gtl" "Gtl" "Gtl" "Gtl" ...
##  $ Neighborhood : chr  "CollgCr" "Veenker" "CollgCr" "Crawfor" ...
##  $ Condition1   : chr  "Norm" "Feedr" "Norm" "Norm" ...
##  $ Condition2   : chr  "Norm" "Norm" "Norm" "Norm" ...
##  $ BldgType     : chr  "1Fam" "1Fam" "1Fam" "1Fam" ...
##  $ HouseStyle   : chr  "2Story" "1Story" "2Story" "2Story" ...
##  $ OverallQual  : int  7 6 7 7 8 5 8 7 7 5 ...
##  $ OverallCond  : int  5 8 5 5 5 5 5 6 5 6 ...
##  $ YearBuilt    : int  2003 1976 2001 1915 2000 1993 2004 1973 1931 1939 ...
##  $ YearRemodAdd : int  2003 1976 2002 1970 2000 1995 2005 1973 1950 1950 ...
##  $ RoofStyle    : chr  "Gable" "Gable" "Gable" "Gable" ...
##  $ RoofMatl     : chr  "CompShg" "CompShg" "CompShg" "CompShg" ...
##  $ Exterior1st  : chr  "VinylSd" "MetalSd" "VinylSd" "Wd Sdng" ...
##  $ Exterior2nd  : chr  "VinylSd" "MetalSd" "VinylSd" "Wd Shng" ...
##  $ MasVnrType   : chr  "BrkFace" "None" "BrkFace" "None" ...
##  $ MasVnrArea   : int  196 0 162 0 350 0 186 240 0 0 ...
##  $ ExterQual    : chr  "Gd" "TA" "Gd" "TA" ...
##  $ ExterCond    : chr  "TA" "TA" "TA" "TA" ...
##  $ Foundation   : chr  "PConc" "CBlock" "PConc" "BrkTil" ...
##  $ BsmtQual     : chr  "Gd" "Gd" "Gd" "TA" ...
##  $ BsmtCond     : chr  "TA" "TA" "TA" "Gd" ...
##  $ BsmtExposure : chr  "No" "Gd" "Mn" "No" ...
##  $ BsmtFinType1 : chr  "GLQ" "ALQ" "GLQ" "ALQ" ...
##  $ BsmtFinSF1   : int  706 978 486 216 655 732 1369 859 0 851 ...
##  $ BsmtFinType2 : chr  "Unf" "Unf" "Unf" "Unf" ...
##  $ BsmtFinSF2   : int  0 0 0 0 0 0 0 32 0 0 ...
##  $ BsmtUnfSF    : int  150 284 434 540 490 64 317 216 952 140 ...
##  $ TotalBsmtSF  : int  856 1262 920 756 1145 796 1686 1107 952 991 ...
##  $ Heating      : chr  "GasA" "GasA" "GasA" "GasA" ...
##  $ HeatingQC    : chr  "Ex" "Ex" "Ex" "Gd" ...
##  $ CentralAir   : chr  "Y" "Y" "Y" "Y" ...
##  $ Electrical   : chr  "SBrkr" "SBrkr" "SBrkr" "SBrkr" ...
##  $ X1stFlrSF    : int  856 1262 920 961 1145 796 1694 1107 1022 1077 ...
##  $ X2ndFlrSF    : int  854 0 866 756 1053 566 0 983 752 0 ...
##  $ LowQualFinSF : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ GrLivArea    : int  1710 1262 1786 1717 2198 1362 1694 2090 1774 1077 ...
##  $ BsmtFullBath : int  1 0 1 1 1 1 1 1 0 1 ...
##  $ BsmtHalfBath : int  0 1 0 0 0 0 0 0 0 0 ...
##  $ FullBath     : int  2 2 2 1 2 1 2 2 2 1 ...
##  $ HalfBath     : int  1 0 1 0 1 1 0 1 0 0 ...
##  $ BedroomAbvGr : int  3 3 3 3 4 1 3 3 2 2 ...
##  $ KitchenAbvGr : int  1 1 1 1 1 1 1 1 2 2 ...
##  $ KitchenQual  : chr  "Gd" "TA" "Gd" "Gd" ...
##  $ TotRmsAbvGrd : int  8 6 6 7 9 5 7 7 8 5 ...
##  $ Functional   : chr  "Typ" "Typ" "Typ" "Typ" ...
##  $ Fireplaces   : int  0 1 1 1 1 0 1 2 2 2 ...
##  $ FireplaceQu  : chr  NA "TA" "TA" "Gd" ...
##  $ GarageType   : chr  "Attchd" "Attchd" "Attchd" "Detchd" ...
##  $ GarageYrBlt  : int  2003 1976 2001 1998 2000 1993 2004 1973 1931 1939 ...
##  $ GarageFinish : chr  "RFn" "RFn" "RFn" "Unf" ...
##  $ GarageCars   : int  2 2 2 3 3 2 2 2 2 1 ...
##  $ GarageArea   : int  548 460 608 642 836 480 636 484 468 205 ...
##  $ GarageQual   : chr  "TA" "TA" "TA" "TA" ...
##  $ GarageCond   : chr  "TA" "TA" "TA" "TA" ...
##  $ PavedDrive   : chr  "Y" "Y" "Y" "Y" ...
##  $ WoodDeckSF   : int  0 298 0 0 192 40 255 235 90 0 ...
##  $ OpenPorchSF  : int  61 0 42 35 84 30 57 204 0 4 ...
##  $ EnclosedPorch: int  0 0 0 272 0 0 0 228 205 0 ...
##  $ X3SsnPorch   : int  0 0 0 0 0 320 0 0 0 0 ...
##  $ ScreenPorch  : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ PoolArea     : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ PoolQC       : chr  NA NA NA NA ...
##  $ Fence        : chr  NA NA NA NA ...
##  $ MiscFeature  : chr  NA NA NA NA ...
##  $ MiscVal      : int  0 0 0 0 0 700 0 350 0 0 ...
##  $ MoSold       : int  2 5 9 2 12 10 8 11 4 1 ...
##  $ YrSold       : int  2008 2007 2008 2006 2008 2009 2007 2009 2008 2008 ...
##  $ SaleType     : chr  "WD" "WD" "WD" "WD" ...
##  $ SaleCondition: chr  "Normal" "Normal" "Normal" "Abnorml" ...
##  $ SalePrice    : int  208500 181500 223500 140000 250000 143000 307000 200000 129900 118000 ...
```

**Observation**:

There are 2 types of data, integer and character. I will change categorical variables into factors later so modelling would treat them correctly.

There are some variables should be in categorical form:

- MSsubClass: should be categorical variable as it indicated the type of dwelling involved in the sale. 
- MoSold should be a categorical rather than numeric variable as high values are not better than low values. The movement of house prices is observed not having a monthly trend  (i.e. sold in December is not always better or worse than in Januray)
- Same as MoSold for YrSold and YearBuilt. However, these 2 predictors can create a new numeric predictor age which is likely affecting the Sale price. So I'll leave them for data type converting for later.

Others:

- YearBuild of some houses were the same with YearRemodAdd, so not all houses were remoded. 
- GarageQual and GaraCond look the similar, their decriptions in the data_description.txt sounds the similar too. Need to check the association between them.


```r
# Categorical variables
vars_cat <- which(sapply(df, is.character))

# Change character variables into factor
df[,vars_cat] <- data.frame(lapply(df[,vars_cat], as.factor))

#Convert MSSubClass and MoSold variables into factor
df$MSSubClass <- as.factor(df$MSSubClass)
df$MoSold <- as.factor(df$MoSold)
```

Some variables should be in ordinal form:

- Some catergorical variables related to quality should be in ordinal form.
- While OveralQual and OveralCond also should be treated as ordinal variable but since they are have 10 levels which are in numbers so in this case I would leave them as they are and treat them as numeric variables.


```r
# OverallQual and OverallCond
df$OverallQual <- factor(df$OverallQual, levels = c(1:10), ordered = TRUE)
df$OverallCond <- factor(df$OverallCond, levels = c(1:10), ordered = TRUE)

# KitchenQual
df$KitchenQual <- factor(df$KitchenQual, levels = c("Po","Fa","TA","Gd","Ex"), ordered = TRUE)

# GarageFinish ,GarageQual, GarageCond
df$GarageFinish <- factor(df$GarageFinish, levels = c("None", "Unf","RFn","Fin"), ordered = TRUE)

df$GarageQual <- factor(df$GarageQual, levels = c("None","Po","Fa","TA","Gd","Ex"), ordered = TRUE)

df$GarageCond <- factor(df$GarageCond, levels = c("None","Po","Fa","TA","Gd","Ex"), ordered = TRUE)

# ExterQual, ExterCond
df$ExterQual <- factor(df$ExterQual,levels = c("Po","Fa","TA","Gd","Ex"), ordered = TRUE)

df$ExterCond <- factor(df$ExterCond,levels = c("Po","Fa","TA","Gd","Ex"), ordered = TRUE)

# BsmtQual, BsmtCont ,BsmtExposure ,BsmtFinType1
df$BsmtQual <- factor(df$BsmtQual, levels = c("None","Po","Fa","TA","Gd","Ex"), ordered = TRUE)

df$BsmtCond <- factor(df$BsmtCond, levels = c("None","Po","Fa","TA","Gd","Ex"), ordered = TRUE)

df$BsmtExposure <- factor(df$BsmtExposure, levels = c("None","Po","Fa","TA","Gd","Ex"), ordered = TRUE)

df$BsmtFinType1 <- factor(df$BsmtFinType1, levels = c("None","Unf","LwQ","Rec","BLQ","ALQ","GLQ"), ordered = TRUE)

# FireplaceQu
df$FireplaceQu <- factor(df$FireplaceQu, levels = c("None","Po","Fa","TA","Gd","Ex"), ordered = TRUE)

# Electrical
df$Electrical <- factor(df$Electrical, levels = c("FuseP","Mix","FuseF","FuseA","SBrkr"), ordered = TRUE)

# Fence
df$Fence <- factor(df$Fence, levels = c("None","MnWw","MnPrv","GdWo","GdPrv"), ordered = TRUE)

# PoolQC
df$PoolQC <- factor(df$PoolQC, levels = c("None","Fa","Gd","Ex"), ordered =  TRUE)
```

### 2.4.2 Data clearning

This part is mainly fixing typos


```r
df <- df %>% 
  mutate(YearRemodAdd = ifelse(YearRemodAdd > YrSold, YrSold, YearRemodAdd), #typo
         GarageYrBlt = ifelse(GarageYrBlt == 2207, 2007, GarageYrBlt)) # typo
```


## 2.5 Descriptive statistics


```r
df_table <- describe(df)
df_table %>% round(digits = 3) %>% 
datatable(options = list(pageLength = 10),width = "50%") 
```

```{=html}
<div id="htmlwidget-a3ab1898f98c994aaf1d" style="width:50%;height:auto;" class="datatables html-widget "></div>
<script type="application/json" data-for="htmlwidget-a3ab1898f98c994aaf1d">{"x":{"filter":"none","vertical":false,"data":[["MSSubClass*","MSZoning*","LotFrontage","LotArea","Street*","Alley*","LotShape*","LandContour*","Utilities*","LotConfig*","LandSlope*","Neighborhood*","Condition1*","Condition2*","BldgType*","HouseStyle*","OverallQual*","OverallCond*","YearBuilt","YearRemodAdd","RoofStyle*","RoofMatl*","Exterior1st*","Exterior2nd*","MasVnrType*","MasVnrArea","ExterQual*","ExterCond*","Foundation*","BsmtQual*","BsmtCond*","BsmtExposure*","BsmtFinType1*","BsmtFinSF1","BsmtFinType2*","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","Heating*","HeatingQC*","CentralAir*","Electrical*","X1stFlrSF","X2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual*","TotRmsAbvGrd","Functional*","Fireplaces","FireplaceQu*","GarageType*","GarageYrBlt","GarageFinish*","GarageCars","GarageArea","GarageQual*","GarageCond*","PavedDrive*","WoodDeckSF","OpenPorchSF","EnclosedPorch","X3SsnPorch","ScreenPorch","PoolArea","PoolQC*","Fence*","MiscFeature*","MiscVal","MoSold*","YrSold","SaleType*","SaleCondition*","SalePrice"],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80],[2919,2915,2433,2919,2919,198,2919,2919,2917,2919,2919,2919,2919,2919,2919,2919,2919,2919,2919,2919,2919,2919,2918,2918,2895,2896,2919,2919,2919,2838,2837,276,2840,2918,2839,2918,2918,2918,2919,2919,2919,2918,2919,2919,2919,2919,2917,2917,2919,2919,2919,2919,2918,2919,2917,2919,1499,2762,2760,2760,2918,2918,2760,2760,2919,2919,2919,2919,2919,2919,2919,10,571,105,2919,2919,2919,2918,2919,1460],[5.267,4.028,69.306,10168.114,1.996,1.394,2.948,3.777,1,4.056,1.054,13.321,3.04,3.002,1.506,4.027,6.089,5.565,1971.313,1984.263,2.396,2.063,10.623,11.335,2.765,102.201,3.397,3.086,2.393,4.577,4.003,5,4.64,441.423,5.678,49.582,560.772,1051.778,2.025,2.534,1.933,4.889,1159.582,336.484,4.694,1500.76,0.43,0.061,1.568,0.38,2.86,1.045,3.511,6.452,6.76,0.597,4.443,3.284,1978.041,2.815,1.767,472.875,3.962,3.971,2.831,93.71,47.487,23.098,2.602,16.062,2.252,3.2,3.588,2.876,50.826,6.213,2007.793,8.491,4.779,180921.196],[4.345,0.659,23.345,7886.996,0.064,0.49,1.41,0.704,0.019,1.604,0.249,5.822,0.874,0.209,1.207,1.913,1.41,1.113,30.291,20.893,0.821,0.539,3.199,3.551,0.608,179.334,0.58,0.372,0.727,0.699,0.295,0,2.058,455.611,1.003,169.206,439.544,440.766,0.246,1.743,0.25,0.41,392.362,428.701,46.397,506.051,0.525,0.246,0.553,0.503,0.823,0.214,0.662,1.569,0.936,0.646,0.766,1.79,25.206,0.82,0.762,215.395,0.253,0.237,0.537,126.527,67.575,64.244,25.188,56.184,35.664,0.789,0.836,0.474,567.402,2.715,1.315,1.595,1.078,79442.503],[5,4,68,9453,2,1,4,4,1,5,1,13,3,3,1,3,6,5,1973,1993,2,2,13,14,3,0,3,3,2,5,4,5,5,368.5,6,0,467,989.5,2,1,2,5,1082,0,0,1444,0,0,2,0,3,1,3,6,7,1,5,2,1979,3,2,480,4,4,3,0,26,0,0,0,0,3,3,3,0,6,2008,9,5,163000],[4.723,4.072,68.435,9499.492,2,1.369,3.059,3.997,1,4.319,1,13.306,3,3,1.161,4.013,6.069,5.47,1974.191,1985.62,2.246,2,10.925,11.645,2.735,61.414,3.34,3.009,2.454,4.521,4,5,4.675,382.444,5.972,1.905,512.459,1034.984,2,2.418,2,5,1127.139,274.21,0,1453.447,0.394,0,1.559,0.34,2.836,1,3.456,6.34,7,0.517,4.531,3.115,1980.693,2.769,1.768,468.42,4,4,3,71.154,33.804,4.94,0,0,0,3.25,3.512,3,0,6.153,2007.741,8.916,5,170783.291],[5.93,0,17.791,3023.021,0,0,0,0,0,0,0,7.413,0,0,0,0,1.483,0,37.065,20.756,0,0,1.483,2.965,0,0,0,0,1.483,1.483,0,0,2.965,546.338,0,0,415.128,350.635,0,0,0,0,348.411,0,0,464.054,0,0,0,0,0,0,0,1.483,0,1.483,1.483,0,31.135,1.483,0,183.842,0,0,0,0,38.548,0,0,0,0,1.483,0,0,0,2.965,1.483,0,0,56338.8],[1,1,21,1300,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1872,1950,1,1,1,1,1,0,2,1,1,3,2,5,2,0,1,0,0,0,1,1,1,1,334,0,0,334,0,0,0,0,0,0,2,2,1,0,2,1,1895,2,0,0,2,2,1,0,0,0,0,0,0,2,2,1,0,1,2006,1,1,34900],[16,5,313,215245,2,2,4,4,2,5,3,25,9,8,5,8,10,9,2010,2010,6,8,15,16,4,1600,5,5,6,6,5,5,7,5644,6,1526,2336,6110,6,5,2,5,5095,2065,1064,5642,3,2,4,2,8,3,5,15,7,4,6,6,2010,4,5,1488,6,6,3,1424,742,1012,508,576,800,4,5,4,17000,12,2010,9,6,755000],[15,4,292,213945,1,1,3,3,1,4,2,24,8,7,4,7,9,8,138,60,5,7,14,15,3,1600,3,4,5,3,3,0,5,5644,5,1526,2336,6110,5,4,1,4,4761,2065,1064,5308,3,2,4,2,8,3,3,13,6,4,4,5,115,2,5,1488,4,4,2,1424,742,1012,508,576,800,2,3,3,17000,11,4,8,5,720100],[0.738,-1.75,1.501,12.816,-15.492,0.431,-0.617,-3.115,53.954,-1.196,4.973,-0.01,2.982,12.054,2.191,0.317,0.197,0.57,-0.599,-0.451,1.553,8.703,-0.731,-0.68,-0.061,2.6,0.786,1.315,0.008,0.252,-0.332,null,-0.168,1.424,-3.397,4.143,0.919,1.162,12.073,0.486,-3.457,-4.792,1.469,0.861,12.083,1.269,0.623,3.928,0.168,0.694,0.326,4.3,0.438,0.758,-4.052,0.733,-0.969,0.747,-0.661,0.353,-0.218,0.241,-2.156,-3.641,-2.978,1.841,2.534,4.002,11.37,3.945,16.89,-0.293,0.673,-3.046,21.936,0.196,0.132,-3.723,-2.787,1.879],[-0.476,5.907,11.259,264.313,238.089,-1.824,-1.589,8.387,2910.002,-0.441,26.508,-1.029,15.666,307.799,3.187,-0.955,0.063,1.472,-0.514,-1.347,0.869,76.672,-0.311,-0.561,-0.14,9.228,0.065,6.269,0.751,-0.363,11.35,null,-1.619,6.884,10.886,18.787,0.399,9.125,167.802,-1.513,9.956,29.348,6.936,-0.425,174.51,4.108,-0.738,14.808,-0.541,-1.035,1.933,19.726,-0.255,1.162,16.203,0.072,1.46,-1.312,-0.383,-1.427,0.234,0.933,20.131,36.746,7.105,6.721,10.907,28.306,149.048,17.73,297.914,-1.504,-0.893,9.413,562.719,-0.457,-1.156,13.598,7.208,6.497],[0.08,0.012,0.473,145.98,0.001,0.035,0.026,0.013,0,0.03,0.005,0.108,0.016,0.004,0.022,0.035,0.026,0.021,0.561,0.387,0.015,0.01,0.059,0.066,0.011,3.332,0.011,0.007,0.013,0.013,0.006,0,0.039,8.434,0.019,3.132,8.137,8.16,0.005,0.032,0.005,0.008,7.262,7.935,0.859,9.366,0.01,0.005,0.01,0.009,0.015,0.004,0.012,0.029,0.017,0.012,0.02,0.034,0.48,0.016,0.014,3.987,0.005,0.005,0.01,2.342,1.251,1.189,0.466,1.04,0.66,0.249,0.035,0.046,10.502,0.05,0.024,0.03,0.02,2079.105]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th> <\/th>\n      <th>vars<\/th>\n      <th>n<\/th>\n      <th>mean<\/th>\n      <th>sd<\/th>\n      <th>median<\/th>\n      <th>trimmed<\/th>\n      <th>mad<\/th>\n      <th>min<\/th>\n      <th>max<\/th>\n      <th>range<\/th>\n      <th>skew<\/th>\n      <th>kurtosis<\/th>\n      <th>se<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"pageLength":10,"columnDefs":[{"className":"dt-right","targets":[1,2,3,4,5,6,7,8,9,10,11,12,13]},{"orderable":false,"targets":0}],"order":[],"autoWidth":false,"orderClasses":false}},"evals":[],"jsHooks":[]}</script>
```

# 3 Exploring variables


```r
# Using data from now on, keep df untouched just in case of checking back
data <- df
```

## 3.1 Sale price


```r
summary(data$SalePrice)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##   34900  129975  163000  180921  214000  755000    1459
```
The min Sale price was 34,900 (my dream!). On the other hand, the max Sale price was 755,000, which is over 20 times more than the min sale price. It sounds ok to me as I don't see any unusual at the moment.


```r
ggplot(data = data[!is.na(data$SalePrice),], aes(x = SalePrice)) + 
  geom_histogram(fill = "steelblue", color = "white") +
labs(x = "Sale Price", y = "Count") +
   theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'),
        panel.border = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.line = element_line( linewidth = 1, colour = "black")) 
```

<img src="house_price_files/figure-html/saleprice_hist-1.png" width="70%" style="display: block; margin: auto;" />
The Sale price obviously looks right skewed. We need to normalize it to meet normality assumption of linear regression. Log transformation can solve the issue. It looks normally distributed now.


```r
skewness(data$SalePrice, na.rm = T)
```

```
## [1] 1.880941
```

```r
# using data from now
data <- data %>% mutate(log_SalePrice = log(SalePrice))
skew(data$log_SalePrice, na.rm= T)
```

```
## [1] 0.1210859
```



```r
ggplot(data[!is.na(data$log_SalePrice),], aes(x = log_SalePrice)) + 
  geom_histogram(fill = "steelblue", color = "white") +
labs(x = "Log(Sale Price)", y = "Count") +
   theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'),
        panel.border = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.line = element_line(linewidth = 1, colour = "black")) 
```

<img src="house_price_files/figure-html/saleprice1-1.png" width="70%" style="display: block; margin: auto;" />

```r
# Remove SalePrice
data$SalePrice <- NULL
```

## 3.2 Exploring predictors of Sale Price

I wanted to quickly figure out which predictor variables were important. I tried several tools, but they couldn't handle missing data. Since there were many variables with missing values, I decided to wait before imputing the missing data and check first if certain variables were worth completing.

So I tried party package and it worked. It is a popular package for constructing decision trees and random forests.

### 3.2.1 Finding important predicitors



```r
library(party)
set.seed(4321)
# Fit a cforest model
fit <- cforest(log_SalePrice ~., data = data[1:1460,], control = cforest_unbiased(mtry = 2,ntree = 50))

# Compute variable importance measures
vi <- varimp(fit)

# Create a data frame with the variable names and importance measures
vi_df <- data.frame(variable = names(vi), importance = vi)

# Sort the data frame by importance measures in descending order
vi_df <- vi_df[order(-vi_df$importance),]

# Create a barplot with ggplot
ggplot(vi_df[1:10,], aes(x = reorder(variable, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Variables", y = "Importance", title = "Variable Importance Plot") +
  theme_bw() +
   coord_flip()
```

<img src="house_price_files/figure-html/unnamed-chunk-10-1.png" width="80%" style="display: block; margin: auto;" />

- The most important variables are Neighborhood, GrLivArea and OverallQual. That makes a lot of sense to me.


### 3.3.2 Visualizing relationship of Log_SalePrice with most important variables.

**Log_saleprice vs. Neighborhood**


```r
data_fullPrice <- data[!is.na(data$log_SalePrice),]
ggplot(data=data_fullPrice, aes(x = reorder(Neighborhood,log_SalePrice, FUN = median), y = log_SalePrice)) + 
  labs(x="Neighborhood") +
  geom_boxplot(fill =  "steelblue") + 
  coord_flip() +
  theme_bw() +
  geom_hline(yintercept= median(data_fullPrice$log_SalePrice), linetype="dashed", color = "red") # median log_SalePrice
```

<img src="house_price_files/figure-html/unnamed-chunk-11-1.png" style="display: block; margin: auto;" />


**Log_SalePrice vs. OverallQual (r = 0.81)**

OverallQual: rating the overall material and finish of the house on a scale from very poor (1) to very excellent (10)


```r
ggplot(data=data_fullPrice, aes(x=factor(OverallQual), y=log_SalePrice)) +
        geom_boxplot(fill = "steelblue") +
  labs(x="Overall Quality") +
  theme_bw()
```

<img src="house_price_files/figure-html/unnamed-chunk-12-1.png" width="70%" style="display: block; margin: auto;" />

Graph shows the positive linear relationship between Log_SalePrice with Overal Quality. There are a few extreme points below housed with grade 3,4,7 and 10, and 1 point above house with grade 4.


**Log_SalePrice vs. GrLivArea (r = 0.7)**

GrLivArea: Above Grade Living Area


```r
library(ggrepel)
data_fullPrice$name <- rownames(data_fullPrice)
ggplot(data=data_fullPrice, aes(x=GrLivArea, y=log_SalePrice)) +
        geom_point(color = "steelblue") + 
  geom_smooth(method = "lm", se = FALSE) +
  geom_text_repel(data = subset(data_fullPrice, GrLivArea > 4550), aes(label = name)) +
  theme_bw()
```

<img src="house_price_files/figure-html/unnamed-chunk-13-1.png" width="70%" style="display: block; margin: auto;" />

### 3.2.3 Correlation matrix



```r
# Selecting numeric variables
vars_num <- which(sapply(data, is.numeric))
data_varNum <- data[, vars_num] 

# Correlation of numeric variables
data_corr <- cor(data_varNum, use="pairwise.complete.obs")
#data_corr <-  vars_num %>% drop_na() %>% cor()

ggcorrplot(data_corr, type = "full", lab = TRUE, lab_size = 1.5, show.legend = TRUE, tl.cex = 5, ggtheme = ggplot2::theme_dark(), title = "Correlation of numeric predictors")
```

<img src="house_price_files/figure-html/correlation-1.png" width="120%" style="display: block; margin: auto;" />


```r
# Select high correlation (> 0.7) to detect multicollinear
corr_table <- melt(data_corr) %>% arrange(desc(value)) %>%
  mutate(value = round(value, digits = 4))%>%
  filter(value !=1)
  
(corr_high <- corr_table %>% filter(abs(value) > 0.7))
```

```
##             Var1          Var2  value
## 1     GarageArea    GarageCars 0.8897
## 2     GarageCars    GarageArea 0.8897
## 3    GarageYrBlt     YearBuilt 0.8438
## 4      YearBuilt   GarageYrBlt 0.8438
## 5   TotRmsAbvGrd     GrLivArea 0.8084
## 6      GrLivArea  TotRmsAbvGrd 0.8084
## 7      X1stFlrSF   TotalBsmtSF 0.8017
## 8    TotalBsmtSF     X1stFlrSF 0.8017
## 9  log_SalePrice     GrLivArea 0.7009
## 10     GrLivArea log_SalePrice 0.7009
```

**Observation**:

- OverallQual and  GrLivArea are hightly correlated with Log_SalePrice like we have found out in the previous session. 

- Some of the predictor variables are highly correlated (r > 0.7) with each other, such as GarageArea vs. GarageCars, GarageYrBlt vs. YearBuilt, GrLivArea vs. TotalRmsAbvGrd, and TotalBsmtSF vs. X1stFlrSF. This presents a problem with multicollinearity that needs to be addressed.

- Beside, YearBuilt and YearRemodAdd are also highly correlated to each other and have high correlction with Log_SalePrice (r > 0.5). 


## 4. Data processing
### 4.1 Imputing missing data

#### Predictors which have the most missing values

As found out in 2.3, five features having the most missing values which is about 50% or more are: PoolQC, MiscFeature, Alley, Fence, FireplaceQu. The NA value reflects the houses didn't have these features. So I replace with "None" to indicate absence of the feature.


```r
# Imputing top missing value predictors
missing_data_top <- c("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu")
data <- data %>% mutate(across(missing_data_top,
    ~ fct_explicit_na(., na_level = "None")
  ))
```


#### Other missing values


```r
# Imputing missing values
# Creating function to find the mode first

findMode <- function(x) {
names(table(x))[table(x)==max(table(x))]
}

# LotFontage
data$LotFrontage <- data$LotFrontage %>% replace_na(median(data$LotFrontage, na.rm = TRUE))

# Garage
data <- data %>% mutate(across(
    c(GarageFinish, GarageQual, GarageCond, GarageType),
    ~ fct_explicit_na(., na_level = "None")
  ))

data$GarageYrBlt <- replace(data$GarageYrBlt, is.na(data$GarageYrBlt), 0)


data$GarageCars <- data$GarageCars %>%
  replace_na(as.integer(findMode(data$GarageCars)))

data$GarageArea <- data$GarageArea %>% 
  replace_na(median(data$GarageArea, na.rm = T))

# Basement
data <- data %>% mutate(across(
  c(BsmtExposure, BsmtCond, BsmtQual, BsmtFinType1, BsmtFinType2),
    ~ fct_explicit_na(., na_level = "None")))


  
data <- data %>%  replace_na(list(BsmtFullBath =  0,
                                  BsmtHalfBath = 0,
                                  BsmtFinSF1 =  0,
                                  BsmtFinSF2 = 0,
                                  BsmtUnfSF = 0,
                                  TotalBsmtSF = 0))
# Exteriorior
data <- data %>% 
  replace_na(list(Exterior1st = findMode(data$Exterior1st),
                  Exterior2nd =  findMode(data$Exterior2nd)))

# Electrical
data$Electrical <- data$Electrical %>% replace_na(findMode(data$Electrical))

# Kitchen
data$KitchenQual <- data$KitchenQual %>% replace_na(findMode(data$KitchenQual))

# MasVnrType and masVnrArea
data <- data %>% 
  replace_na(list(MasVnrType = "None", MasVnrArea = 0))

# MsZoning
data <- data %>% 
  group_by(Neighborhood) %>% 
  mutate(MSZoning = fct_explicit_na(MSZoning, na_level =   
  findMode(MSZoning)))

# Functional
data$Functional <- data$Functional %>% replace_na(findMode(data$Functional))

# Utilities
data$Utilities <- data$Utilities %>%
  replace_na(findMode(data$Utilities))

# Sale Type
data$SaleType <- data$SaleType %>% replace_na(findMode(data$SaleType))

# check missing values again if they are all imputed except the targeted values, Sale price.
colnames(data)[colSums(is.na(data)) > 0]
```

```
## [1] "log_SalePrice"
```

### 4.2 Feature engineering

#### 4.2.1 Adding Age, Remod (yes/no), Basement (yes/no) varibles

Based on observations from the 2.4 (Data structure) and 3.2.3 (Correlation matrix) So I create Age, Remod (yes/no) to reflect its age and if the house was remodeled.

I realized not all houses having basement which could add more value to Sale price. So I create a new feature basement/none to replace TotalBsmtSF.


```r
#data %>% select(YearBuilt, YearRemodAdd, YrSold) %>% head(10)
data$Age <- data$YrSold - data$YearRemodAdd
data$Remod <- ifelse(data$YearBuilt == data$YearRemodAdd, 0 , 1)

# Convert YrSold to factor after creating Age avariable
data$YrSold <- as.factor(data$YrSold)

#sum(data1$TotalBsmtSF == 0) # 79 houses without basement
data <- data %>% mutate(Basement = case_when(TotalBsmtSF == 0 ~ 0, TRUE ~ 1))

# Checking correlation of new variables with Sale price
# it's obvious the house price is negatively correlated with Age
cor(data$log_SalePrice, data$Age, use = "pairwise.complete.obs")
```

```
## [1] -0.5681486
```


#### 4.2.2 Groupping variables: Bathroom and  Porcharea

Among these numeric features, there are some features I think can be group together to create stronger predictors such as Bathrooms and Porch area rather than breaking them down.


```r
data <- data %>% mutate(
Bathrooms = FullBath + HalfBath*0.5 + BsmtHalfBath*0.5 + BsmtFullBath,
PorchArea = ScreenPorch + X3SsnPorch + OpenPorchSF + EnclosedPorch 
)

# Checking correlation with Sale price again
# It's obvious bathroom now becomes a stronger predictor while Porcharea seems remaining the same.
cor(data$log_SalePrice, data$Bathrooms, use = "pairwise.complete.obs")
```

```
## [1] 0.6730106
```

```r
cor(data$log_SalePrice, data$PorchArea, use = "pairwise.complete.obs")
```

```
## [1] 0.1951663
```


## 5. Data Preparation for modelling
### 5.1 Dropping highly correlated/associated variables

Upon the observations from 2.4 (data structure) and data_description, GarageCond and GarageQual appear likely hightly associated. The cross-tabulation suggests a strong likelihood of high association. The Chi Squared test result also confirms it. So I'm going to drop GarageCond.



```r
data %>% tabyl(GarageCond, GarageQual) %>%
  adorn_totals("row") 
```

```
##  GarageCond None Po  Fa   TA Gd Ex
##        None  159  0   0    0  0  0
##          Po    0  4   8    2  0  0
##          Fa    0  1  39   34  0  0
##          TA    0  0  77 2559 17  1
##          Gd    0  0   0    9  6  0
##          Ex    0  0   0    0  1  2
##       Total  159  5 124 2604 24  3
```

```r
chisq.test(data$GarageQual, data$GarageCond, correct = FALSE)
```

```
## 
## 	Pearson's Chi-squared test
## 
## data:  data$GarageQual and data$GarageCond
## X-squared = 5752.3, df = 25, p-value < 2.2e-16
```


Based on the result from the 3.2.3 (Correlation matrix), I'm dropping variables that are highly correlated with another variables and having lower correlation with Sale price which including TotalBsmtSF, GarageArea, TotalRmsAbvGrd, GarageYrBlt, and YearRemodAdd.


```r
# highly correlated/associated  variables 
high_corr_vars <- c("GarageCond", "TotalBsmtSF","GarageArea", "TotalRmsAbvGrd", "GarageYrBlt", "YearRemodAdd")

data <- data[, !(names(data) %in% high_corr_vars)]
```



### 5.2 Removing outliers

In order to detect outliers, I used Cook's distance, a statistical measure that identifies influential observations in a regression analysis. As a rule of thumb, obs with a Cook's distance value greater than 1 should be removed. To assess the level of influence of these observations, a commonly used threshold is 4 times the mean Cook's distance. In this case, the threshold of 0.5 Cook's distance is used. As a result, three points (822, 524, and 826) were identified as having an outstanding level of influence.


```r
mod <- lm(log_SalePrice ~ ., data = na.omit(data))

cooksd <- cooks.distance(mod)
```



```r
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot Cook's distance
abline(h = 4*mean(cooksd, na.rm = T), col = "red")  # Influential level cut off 
text(x = 1:length(cooksd)+1, y = cooksd, labels = ifelse(cooksd > 0.5,names(cooksd),""), col= "red") # add labels for threshold at 0.5
```

![](house_price_files/figure-html/unnamed-chunk-22-1.png)<!-- -->


These 3 outliers having certain unusual values for important predictors. # 822 had small LotArea and GrLivArea, low OverallQua, located in low value Neighborhood, and very high Age but had a log_SalePrice value that was close to the mean. Meanwhile # 524 had very large LotArea & GrLivArea,  and # 826 was located in high_end Neighborhood, both had high OverallQual but both low values for log_SalePrice.


```r
data[c(822,524,826),] %>% 
  select(LotArea, GrLivArea, OverallQual, Neighborhood, log_SalePrice,MSSubClass, KitchenQual, Age) %>% 
  knitr::kable()
```



| LotArea| GrLivArea|OverallQual |Neighborhood | log_SalePrice|MSSubClass |KitchenQual | Age|
|-------:|---------:|:-----------|:------------|-------------:|:----------|:-----------|---:|
|    6000|       936|4           |OldTown      |      11.44035|20         |TA          |  56|
|   40094|      4676|10          |Edwards      |      12.12676|60         |Ex          |   0|
|   14803|      2084|10          |NridgHt      |      12.86100|20         |Ex          |   0|



```r
# Removing outliers
data <- data[-c(822,524,826),]
```

### 5.3 PreProcessing predictor variables


```r
vars_numNames <- names(vars_num)

vars_numNames <- vars_numNames[!(vars_numNames %in% c("MSSubClass",  "MoSold", "YrSold", "log_SalePrice", "OverallQual", "OverallCond"))]

vars_numNames <- append(vars_numNames, c("Age", "Bathrooms", "PorchArea"))

data_varNum <- data[, names(data) %in% vars_numNames]

data_varFac <- data[,!(names(data) %in% vars_numNames)]
data_varFac <- data_varFac[, names(data_varFac) != "log_SalePrice"]

cat("There are", length(data_varNum), "numeric variable, and", length(data_varFac)," factor variables")
```

```
## There are 30 numeric variable, and 49  factor variables
```

#### 5.3.1 Fixing skewness and normalizing the numeric predictors

In order to fix skewness, I'm going to log all numeric predictors that have absolute skewness > 0.8.


```r
for (i in 1:ncol(data_varNum)) {
  if (abs(skew(data_varNum[,i])) > 0.8){
              data_varNum[,i] <- log(data_varNum[,i] + 1) # log + 1 to avoid 0 values
  }
}
```


```r
pre_varNum <- preProcess(data_varNum, method = c("center", "scale"))
print(pre_varNum)
```

```
## Created from 2916 samples and 30 variables
## 
## Pre-processing:
##   - centered (30)
##   - ignored (0)
##   - scaled (30)
```


```r
data_norm <- predict(pre_varNum, data_varNum)
dim(data_norm)
```

```
## [1] 2916   30
```


#### 5.3.2 One hot encoding the categorical predictors

One hot encoding is a common technique for encoding categorical variables as numeric variables in machine learning. I'm going to use model.matrix() to perform one hot encoding for the dataset.

```r
data_dummies <- as.data.frame(model.matrix (~.-1, data_varFac)) # use all variables in the data except for intercept
dim(data_dummies)
```

```
## [1] 2916  274
```
#### Removing levels with few obs in the data


```r
# Check levels that has a few obs (less than 10) in the train set
sparse_train <- which(colSums(data_dummies[1:nrow(data[!is.na(data$log_SalePrice),]),]) < 10)
colnames(data_dummies[sparse_train])
```

```
##  [1] "MSSubClass40"         "MSSubClass150"        "UtilitiesNoSeWa"     
##  [4] "LotConfigFR3"         "NeighborhoodBlueste"  "NeighborhoodNPkVill" 
##  [7] "Condition1PosA"       "Condition1RRNe"       "Condition1RRNn"      
## [10] "Condition2Feedr"      "Condition2PosA"       "Condition2PosN"      
## [13] "Condition2RRAe"       "Condition2RRAn"       "Condition2RRNn"      
## [16] "HouseStyle2.5Fin"     "OverallQual.Q"        "OverallQual.C"       
## [19] "OverallQual^6"        "OverallQual^9"        "OverallCond.Q"       
## [22] "OverallCond.C"        "OverallCond^5"        "OverallCond^6"       
## [25] "OverallCond^9"        "RoofStyleMansard"     "RoofStyleShed"       
## [28] "RoofMatlMembran"      "RoofMatlMetal"        "RoofMatlRoll"        
## [31] "RoofMatlWdShake"      "RoofMatlWdShngl"      "Exterior1stAsphShn"  
## [34] "Exterior1stBrkComm"   "Exterior1stCBlock"    "Exterior1stImStucc"  
## [37] "Exterior1stStone"     "Exterior2ndAsphShn"   "Exterior2ndBrk Cmn"  
## [40] "Exterior2ndCBlock"    "Exterior2ndOther"     "Exterior2ndStone"    
## [43] "ExterQual.Q"          "ExterQual.C"          "ExterCond.Q"         
## [46] "ExterCond.C"          "FoundationStone"      "FoundationWood"      
## [49] "BsmtQual.Q"           "BsmtQual.C"           "BsmtQual^4"          
## [52] "BsmtCond.Q"           "BsmtCond.C"           "BsmtExposure.L"      
## [55] "BsmtExposure.C"       "BsmtExposure^5"       "BsmtFinType1^4"      
## [58] "BsmtFinType1^6"       "HeatingGrav"          "HeatingOthW"         
## [61] "HeatingWall"          "HeatingQCPo"          "KitchenQual.Q"       
## [64] "KitchenQual.C"        "FunctionalMaj2"       "FunctionalSev"       
## [67] "FireplaceQu.L"        "FireplaceQu.C"        "GarageTypeCarPort"   
## [70] "GarageFinish.Q"       "GarageQual.Q"         "GarageQual.C"        
## [73] "PoolQC.L"             "PoolQC.C"             "Fence.L"             
## [76] "Fence.C"              "MiscFeatureOthr"      "MiscFeatureTenC"     
## [79] "SaleTypeCon"          "SaleTypeConLD"        "SaleTypeConLI"       
## [82] "SaleTypeConLw"        "SaleTypeCWD"          "SaleTypeOth"         
## [85] "SaleConditionAdjLand"
```


```r
# Removing levels that has a few obs in the train set
data_dummies <- data_dummies[, -sparse_train]
dim(data_dummies)
```

```
## [1] 2916  189
```



```r
# Check levels that has a few obs (less than 10) in the test set
sparse_test <- which(colSums(data_dummies[(nrow(data[!is.na(data$log_SalePrice),]) +1 ): nrow(data),]) < 10)
colnames(data_dummies[sparse_test])
```

```
##  [1] "MSSubClass45"       "MSSubClass75"       "MSSubClass180"     
##  [4] "LotShapeIR3"        "LandSlopeSev"       "HouseStyle1.5Unf"  
##  [7] "OverallCond.L"      "Exterior2ndImStucc" "HeatingGasW"       
## [10] "FireplaceQu^5"
```

```r
# Removing levels that has a few obs in the test set
data_dummies <- data_dummies[, -sparse_test]
dim(data_dummies)
```

```
## [1] 2916  179
```
In total, 95 one hot encoded predictors with a few obs have been removed.

Now getting the ready data for modelling.


```r
data_combo <- cbind(data_dummies, data_norm)

data_combo$log_SalePrice <- data$log_SalePrice
```

#### 5.3.3 Split training data into train and test sets.

Now it's time to split the combined dataset above into train set and test set again


```r
train1 <- data_combo[!is.na(data$log_SalePrice),]
test1 <- data_combo[is.na(data$log_SalePrice),]
```

## 6. Modelling


```r
# Split the train1 data set into train2 and test2 subsets
set.seed(20231)
trainIndex <- createDataPartition(train1$log_SalePrice, p = 0.8, list = FALSE)
train2 <- train1[trainIndex, ]
test2 <- train1[-trainIndex, ]
```


```r
# Define the tuning grid for XGBoost
xgb_grid <- expand.grid(
            nrounds = 400,
            max_depth = c(3,4,5),
            eta = seq(0.05,0.1, by= 0.025),
            gamma = 0,
            colsample_bytree = 1,
            min_child_weight = 4,
            subsample = 1)

# Train an XGBoost model on the train2 subset
set.seed(12345)
xgb_model <- train(log_SalePrice ~ .,
                   data = train2,
                   method = "xgbTree",
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = xgb_grid)

xgb_model$bestTune
```

```
##   nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
## 2     400         4 0.05     0                1                4         1
```


```r
# Evaluate the performance of the XGBoost model on the test2 subset
xgb_pred <- predict(xgb_model, newdata = test2)
xgb_rmse <- sqrt(mean((xgb_pred - test2$log_SalePrice)^2))
xgb_rmse
```

```
## [1] 0.1207274
```

```r
# Use the trained XGBoost model to predict the house prices in the test1 data set
pred_test <- predict(xgb_model, newdata = test1)

# Apply inverse log transformation to the predicted house prices
pred_SalePrice<- exp(pred_test)
```


```r
# Create a data frame with Id and predicted SalePrice columns
pred_df <- data.frame(Id =  test_labels, SalePrice = pred_SalePrice)
head(pred_df)
```

```
##     Id SalePrice
## 1 1461  120875.4
## 2 1462  160283.0
## 3 1463  182510.3
## 4 1464  190364.8
## 5 1465  184605.9
## 6 1466  172300.7
```


```r
# Write the data frame to a CSV file
write.csv(pred_df, file = "predicted_sale_prices.csv", row.names = FALSE)
```


## 7. Conclusion

My submission for the Kaggle competition using XGBoost achieved a score of 0.13097, which I was satisfied with for my first attempt, even it is way to far to reach top 5 :D. However, I learned that there are other methods that could be used to improve the score, such as trying different feature engineering techniques and experimenting with model parameters or other algorithms like Lasso or glmnet. Overall, this experience has given me knowledge and insights on data wrangling and a valuable introduction to using XGBoost for machine learning projects and has motivated me to continue exploring different methods to improve my performance.

