---
title: "House Prices Prediction- Advanced Regression Techniques"
subtitle: "Kaggle competitions"
excerpt: "This is a competition for data science beginners or machine learning students are looking to expand their skill set. The goal of this competition is to predict the sales price for a sample of houses. All the submissions are evaluated on RMSE between the logarithm of the predicted value and the logarithm of the observed sales price."
date: "2023-01-29"
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

This project is based on the Kaggle competition [“House Prices: Advanced Regression Techniques”](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The goal of this project is to predict housing prices in based on the provided training data (train.csv) and evaluate the performance of the model using the test data (test.csv). Through this project, I aim to not only build a robust prediction model but also gain some knowledge and insights on data wrangling and analysis.

# 2. Overview the data

## 2.1 Loading packages and reading the data

``` r
# Loading R packages
packages <- c("tidyverse", "psych","DT", "gridExtra", "GGally", "corrplot", "ggcorrplot", "naniar", "visdat", "moments", "mice", "reshape2", "xgboost") 
sapply(packages, require, character = TRUE)
```

<!--more-->

``` r
# Reading data
train <- read.csv("housing_data/train.csv")
test <- read.csv("housing_data/test.csv")
```

## 2.2 Data size and structure

The housing train data set has 1460 obs and 81 variables with the response variable Sale Price. The housing test data set has 1459 obs and 80 variables.

``` r
dim(train) ; dim(test)
```

    ## [1] 1460   81

    ## [1] 1459   80

``` r
# Combine 2 data sets to see the structure, and for cleaning & feature engineering later.
# Removing Id as not necessary but keeping the test Id for the final file.
test_labels <- test$Id
test$Id <- NULL
train$Id <- NULL
test$SalePrice <- NA
df <- rbind(train, test)
dim(df) 
```

    ## [1] 2919   80

The data now has 80 columns consisting of 79 predictors and reponse variable Sale price.

``` r
# Data structure 
str(df)
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

**Observation**:

There are 2 types of data, integer and character. I will change categorical variables into factors later so modelling would treat them correctly.

There are some variables should be in categorical form:

- MSsubClass: should be categorical variable as it indicated the type of dwelling involved in the sale.
- MoSold should be a categorical rather than numeric variable as high values are not better than low values (i.e. sold in December is not better than in Januray)
- Same as MoSold for YrSold and YearBuilt. However, these 2 predictors can create a new numeric predictor age which is likely affect Sale price.

``` r
# Categorical variables
vars_cat <- which(sapply(df, is.character))

# Change data type to factor
df[,vars_cat] <- data.frame(lapply(df[,vars_cat], as.factor))

#Convert MSSubClass and MoSold variables into factor
df$MSSubClass <- as.factor(df$MSSubClass)
df$MoSold <- as.factor(df$MoSold)
```

Some variables should be in ordinal form:

- Some catergorical variables related to quality should be in ordinal form.
- While OveralQual and OveralCond also should be treated as ordinal variables but since they are have 10 levels which are in numbers so in this case I would leave them as they are and treat them as numeric variable.

``` r
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

df$BsmtFinType1 <- factor(df$BsmtFinType1, levels = c("None","Po","Fa","TA","Gd","Ex"), ordered = TRUE)

# FireplaceQu
df$FireplaceQu <- factor(df$FireplaceQu, levels = c("None","Po","Fa","TA","Gd","Ex"), ordered = TRUE)

# Electrical
df$Electrical <- factor(df$Electrical, levels = c("FuseP","Mix","FuseF","FuseA","SBrkr"), ordered = TRUE)

# Fence
df$Fence <- factor(df$Fence, levels = c("None","MnWw","MnPrv","GdWo","GdPrv"), ordered = TRUE)

# PoolQC
df$PoolQC <- factor(df$PoolQC, levels = c("None","Fa","Gd","Ex"), ordered =  TRUE)
```

## 2.3 Missingness of the data

The dataset has 13965 missing values (exclude the missing values for Sale price in the test dataset), happens to be about 6%.

``` r
n_miss(df[,colnames(df)!="SalePrice"])
```

    ## [1] 19366

``` r
pct_miss(df[,colnames(df)!="SalePrice"])
```

    ## [1] 8.398056

``` r
# Select columns with > 0 missing values
df_miss <- names(df[colSums(is.na(df[,colnames(df)!="SalePrice"])) > 0])
cat("There are", length(df_miss), "columns with missing values")
```

    ## There are 34 columns with missing values

``` r
vis_miss(df[,df_miss], sort_miss = TRUE) # visualizing missing data
```

<img src="/project/Housing/house_price_files/figure-html/unnamed-chunk-5-1.png" width="672" />

- The predictors having the most missing values which is about 50% or more are: PoolQC, MiscFeature, Alley, Fence, FireplaceQu. They are all categorical variables. As described in the data_description.txt file, the NA value reflects the houses didn’t have these features.
- Followed by LotFrontage (16.7%), Garage related (5.x%) and basement related variables (2.x%).

I leave imputing missing values later after exploring variables.

## 2.4 Descriptive statistics

``` r
df_table <- describe(df)
df_table %>% round(digits = 3) %>% 
datatable(options = list(pageLength = 10),width = "50%") 
```

<div class="datatables html-widget html-fill-item-overflow-hidden html-fill-item" id="htmlwidget-1" style="width:100%;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"filter":"none","vertical":false,"data":[["MSSubClass*","MSZoning*","LotFrontage","LotArea","Street*","Alley*","LotShape*","LandContour*","Utilities*","LotConfig*","LandSlope*","Neighborhood*","Condition1*","Condition2*","BldgType*","HouseStyle*","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle*","RoofMatl*","Exterior1st*","Exterior2nd*","MasVnrType*","MasVnrArea","ExterQual*","ExterCond*","Foundation*","BsmtQual*","BsmtCond*","BsmtExposure*","BsmtFinType1*","BsmtFinSF1","BsmtFinType2*","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","Heating*","HeatingQC*","CentralAir*","Electrical*","X1stFlrSF","X2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual*","TotRmsAbvGrd","Functional*","Fireplaces","FireplaceQu*","GarageType*","GarageYrBlt","GarageFinish*","GarageCars","GarageArea","GarageQual*","GarageCond*","PavedDrive*","WoodDeckSF","OpenPorchSF","EnclosedPorch","X3SsnPorch","ScreenPorch","PoolArea","PoolQC*","Fence*","MiscFeature*","MiscVal","MoSold*","YrSold","SaleType*","SaleCondition*","SalePrice"],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80],[2919,2915,2433,2919,2919,198,2919,2919,2917,2919,2919,2919,2919,2919,2919,2919,2919,2919,2919,2919,2919,2919,2918,2918,2895,2896,2919,2919,2919,2838,2837,276,0,2918,2839,2918,2918,2918,2919,2919,2919,2918,2919,2919,2919,2919,2917,2917,2919,2919,2919,2919,2918,2919,2917,2919,1499,2762,2760,2760,2918,2918,2760,2760,2919,2919,2919,2919,2919,2919,2919,10,571,105,2919,2919,2919,2918,2919,1460],[5.26652963343611,4.02778730703259,69.3057953144266,10168.1140801644,1.99588900308325,1.39393939393939,2.94758478931141,3.77697841726619,1.00034281796366,4.05584104145255,1.05378554299418,13.3206577595067,3.04042480301473,3.00205549845838,1.50565262076053,4.02672147995889,6.08907159986297,5.56457690990065,1971.31277834875,1984.26447413498,2.3963686193902,2.06303528605687,10.623372172721,11.3351610692255,2.76511226252159,102.201312154696,3.3967112024666,3.08564576909901,2.39328537170264,4.57681465821001,4.00281988015509,5,null,441.423235092529,5.67840789010215,49.5822481151474,560.772104180946,1051.77758738862,2.02535114765331,2.53374443302501,1.93285371702638,4.88930774503084,1159.58170606372,336.483727303871,4.69441589585472,1500.75984926345,0.429893726431265,0.061364415495372,1.56800274066461,0.380267214799589,2.86022610483042,1.04453579993148,3.51096641535298,6.45152449468996,6.75968460747343,0.597122302158273,4.44296197464977,3.28385228095583,1978.1134057971,2.81485507246377,1.76662097326936,472.8745716244,3.96231884057971,3.97065217391304,2.83076396026036,93.7098321342926,47.4868105515587,23.0983213429257,2.60226104830425,16.0623501199041,2.25179856115106,3.2,3.58844133099825,2.87619047619048,50.8259677971907,6.21308667351833,2007.79273723878,8.4910897875257,4.77903391572456,180921.195890411],[4.34490712659322,0.658805836163688,23.3449047069274,7886.99635910555,0.0639960910523377,0.489860268446704,1.40972103277746,0.704390653312761,0.0185153440060205,1.60447219293564,0.248749903819906,5.82241963183029,0.874046993646341,0.209431217563372,1.20651315778852,1.91293670618384,1.40994720661469,1.11313074663773,30.2914415341212,20.8943442338407,0.820906048226983,0.539210166466461,3.19930389764543,3.55119338409643,0.607564711406854,179.33425303776,0.580293356463571,0.372360971164021,0.727061459881234,0.698645248405488,0.294506257077762,0,null,455.610825870297,1.00270711864073,169.205611099981,439.543659423441,440.766258115938,0.245678213251896,1.74254764659886,0.25031804891471,0.409539320628125,392.36207866659,428.701455518118,46.3968245165089,506.05104511834,0.524735633698244,0.245686916449331,0.552969259587454,0.502871600235756,0.822693100671624,0.214462001223478,0.662204863324699,1.56937914364375,0.935604766498271,0.646129358956219,0.766397625027567,1.79029738512212,25.5742847239051,0.81983187263873,0.76162432259936,215.394814993521,0.252627482573003,0.23673417272248,0.537298792015698,126.526589310476,67.5754933916901,64.2442455926333,25.1881693311639,56.1843651106945,35.663945965488,0.788810637746616,0.835908700614153,0.474245110579355,567.402210550197,2.7147617741579,1.3149644889049,1.59460742076364,1.07824090672141,79442.5028828866],[5,4,68,9453,2,1,4,4,1,5,1,13,3,3,1,3,6,5,1973,1993,2,2,13,14,3,0,3,3,2,5,4,5,null,368.5,6,0,467,989.5,2,1,2,5,1082,0,0,1444,0,0,2,0,3,1,3,6,7,1,5,2,1979,3,2,480,4,4,3,0,26,0,0,0,0,3,3,3,0,6,2008,9,5,163000],[4.72314933675646,4.07243891984569,68.4350282485876,9499.49208386818,2,1.36875,3.05905006418485,3.99743260590501,1,4.31878476679504,1,13.3063756953358,3,3,1.16089002995293,4.01283697047497,6.06889174154898,5.47026101839966,1974.19084296106,1985.61959777493,2.24561403508771,2,10.9250856164383,11.6446917808219,2.73457056538628,61.4141501294222,3.34017971758666,3.00855798031665,2.45442875481386,4.52112676056337,4,5,null,382.444349315062,5.97228332600088,1.90453767123285,512.458904109595,1034.9841609589,2,2.41762943945233,2,5,1127.13863928113,274.209670517751,0,1453.44715447155,0.39443254817988,0,1.55926401369277,0.33975181857083,2.83611467693625,1,3.45590753424657,6.34017971758662,7,0.516902011125384,4.53122398001666,3.11538461538459,1980.69338768116,2.76856884057973,1.76797945205479,468.419520547943,4,4,3,71.1536157466841,33.804450149765,4.9400941377837,0,0,0,3.25,3.51203501094092,3,0,6.15318784766796,2007.74112109542,8.91566780821918,5,170783.291095891],[5.9304,0,17.7912,3023.0214,0,0,0,0,0,0,0,7.413,0,0,0,0,1.4826,0,37.065,20.7564,0,0,1.4826,2.9652,0,0,0,0,1.4826,1.4826,0,0,null,546.3381,0,0,415.128,350.6349,0,0,0,0,348.411,0,0,464.0538,0,0,0,0,0,0,0,1.4826,0,1.4826,1.4826,0,31.1346,1.4826,0,183.8424,0,0,0,0,38.5476,0,0,0,0,1.4826,0,0,0,2.9652,1.4826,0,0,56338.8],[1,1,21,1300,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1872,1950,1,1,1,1,1,0,2,1,1,3,2,5,null,0,1,0,0,0,1,1,1,1,334,0,0,334,0,0,0,0,0,0,2,2,1,0,2,1,1895,2,0,0,2,2,1,0,0,0,0,0,0,2,2,1,0,1,2006,1,1,34900],[16,5,313,215245,2,2,4,4,2,5,3,25,9,8,5,8,10,9,2010,2010,6,8,15,16,4,1600,5,5,6,6,5,5,null,5644,6,1526,2336,6110,6,5,2,5,5095,2065,1064,5642,3,2,4,2,8,3,5,15,7,4,6,6,2207,4,5,1488,6,6,3,1424,742,1012,508,576,800,4,5,4,17000,12,2010,9,6,755000],[15,4,292,213945,1,1,3,3,1,4,2,24,8,7,4,7,9,8,138,60,5,7,14,15,3,1600,3,4,5,3,3,0,null,5644,5,1526,2336,6110,5,4,1,4,4761,2065,1064,5308,3,2,4,2,8,3,3,13,6,4,4,5,312,2,5,1488,4,4,2,1424,742,1012,508,576,800,2,3,3,17000,11,4,8,5,720100],[0.738342223086808,-1.74982216639282,1.50142471242018,12.8158428439246,-15.492168716917,0.430836927308458,-0.617372454544532,-3.11509132081283,53.953725128329,-1.19567137263436,4.97260044860579,-0.00974800680561387,2.98158164174491,12.0538958554955,2.1911347336578,0.316933346847224,0.197009167290338,0.570019006502035,-0.599497349057127,-0.450788629007491,1.55250873153804,8.70329268383117,-0.731247393320375,-0.679935402689662,-0.0611575398020273,2.59989308127073,0.785977586632969,1.31523711583217,0.00757387103740258,0.252228895441162,-0.332300197297635,null,null,1.42425683553317,-3.39737673859201,4.14319267074755,0.918878241729549,1.16168718659861,12.0725813864484,0.486405941138226,-3.45724513709174,-4.79246744500907,1.46884928920505,0.8612319939896,12.0825494280388,1.26870545340664,0.623420526025012,3.92797523267679,0.167519612310872,0.694209605143622,0.32615667385233,4.30004373966202,0.43773351555747,0.757977233976828,-4.05204165067148,0.733117706990301,-0.968516124007526,0.74661403583706,-0.381734880693549,0.352771940363206,-0.218148308494456,0.241052492603059,-2.1561691553204,-3.6407648161024,-2.97755344787575,1.84148611276007,2.53381111046927,4.00183389870891,11.3702193126747,3.94466577099656,16.8896450381553,-0.2933892627129,0.673304589508143,-3.04593002269886,21.9359176674511,0.195783291899439,0.132330784029226,-3.72273488113925,-2.78660613271373,1.87900860360558],[-0.476117974855987,5.90741122090735,11.2585267728837,264.313383829956,238.088857118634,-1.82351280874008,-1.58879356558952,8.38698274997069,2910.00205655523,-0.440935743232344,26.5075824143602,-1.02877745934972,15.6663633020742,307.799353832829,3.18668433214177,-0.954887865274705,0.06294977108305,1.47179408919692,-0.514200729426967,-1.3473139158451,0.868925679546427,76.6720340391318,-0.311440277336161,-0.560602807780714,-0.139877416283842,9.22785311866979,0.0647750388082753,6.26879216824093,0.750610466151918,-0.362620705058108,11.3500255820428,null,null,6.88417272197202,10.8855799246531,18.7872825538659,0.398539580743025,9.1250559848258,167.80243538589,-1.51336183890352,9.95595503169773,29.3476570199628,6.93570298995635,-0.425357479267518,174.509570114517,4.10761996328787,-0.73804088489242,14.8083680385259,-0.540948552214265,-1.03507890643019,1.93264366957674,19.7264407154592,-0.255159065640423,1.16215401071867,16.2029942961395,0.0721321982753156,1.45979181776006,-1.31239096712348,1.80091373580091,-1.42666252852748,0.233516968690023,0.933420502426853,20.1311124749754,36.7463900758771,7.10467455507513,6.72128910481376,10.9070384236174,28.3058077713787,149.047744258177,17.7300026247504,297.913519000927,-1.50397959183674,-0.892577586467937,9.41317161090797,562.718967479078,-0.457356544482714,-1.1564873952393,13.5975912991346,7.20762565171686,6.4967893326974],[0.0804198854657326,0.012202200540559,0.473283158934305,145.980414629769,0.00118450364132845,0.0348128533823296,0.0260925264203493,0.01303756651456,0.000342817963661288,0.0296971756194956,0.00460411193886531,0.107767164241405,0.0161777356966938,0.00387636237983324,0.0223313518874523,0.0354065452568423,0.0260967126718744,0.0206029368510763,0.560664287587665,0.386733414825502,0.0151941499445361,0.0099802409040655,0.059226061235767,0.0657402996262205,0.0112919320786131,3.33245251261437,0.0107406496626124,0.00689202916896388,0.0134571804705162,0.0131144630594934,0.00552923257978711,0,null,8.4343455751521,0.018818777258837,3.13236322808165,8.13690744038562,8.15954039659897,0.00454725801852404,0.0322528141741755,0.00463313673623,0.00758146198609854,7.26222966524136,7.93483518686673,0.858758819205633,9.36649873115717,0.00971566077014039,0.00454897777583788,0.0102349079569307,0.00930765038622084,0.0152272264980117,0.00396947715397245,0.0122588497500208,0.0290476383745165,0.0173230441053884,0.0119592082245608,0.0197949009891111,0.0340653880286221,0.486798190166156,0.0156052329967227,0.0140993197933204,3.98742567471057,0.00480868194870053,0.00450615796593776,0.00994486327459405,2.34188062581134,1.25075479878235,1.18909673368391,0.466207823017829,1.03991640684613,0.660103971439649,0.249443825784929,0.0349816903762591,0.0462815838009878,10.5020474445051,0.0502475253402502,0.024338677561597,0.0295196454512063,0.0199571608084044,2079.10532396724]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th> <\/th>\n      <th>vars<\/th>\n      <th>n<\/th>\n      <th>mean<\/th>\n      <th>sd<\/th>\n      <th>median<\/th>\n      <th>trimmed<\/th>\n      <th>mad<\/th>\n      <th>min<\/th>\n      <th>max<\/th>\n      <th>range<\/th>\n      <th>skew<\/th>\n      <th>kurtosis<\/th>\n      <th>se<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"pageLength":10,"columnDefs":[{"className":"dt-right","targets":[1,2,3,4,5,6,7,8,9,10,11,12,13]},{"orderable":false,"targets":0}],"order":[],"autoWidth":false,"orderClasses":false}},"evals":[],"jsHooks":[]}</script>

# 3 Exploring variables

``` r
# Using data from now on, keep df untouched just in case of checking back
data <- df
```

## 3.1 Sale price

``` r
summary(data$SalePrice)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
    ##   34900  129975  163000  180921  214000  755000    1459

The min Sale price was 34,900 (my dream!). On the other hand, the max Sale price was 755,000, which is over 20 times more than the min sale price. It sounds ok to me as I don’t see any unusual at the moment.

``` r
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

<img src="/project/Housing/house_price_files/figure-html/saleprice_hist-1.png" width="60%" style="display: block; margin: auto;" />
The Sale price obviously looks right skewed. We need to normalize it to meet normality assumption of linear regression. Log transformation can solve the issue. It looks normally distributed now.

``` r
skewness(data$SalePrice, na.rm = T)
```

    ## [1] 1.880941

``` r
# using data1 from now
data <- data %>% mutate(log_SalePrice = log(SalePrice))
skewness(data$log_SalePrice, na.rm= T)
```

    ## [1] 0.1212104

``` r
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

<img src="/project/Housing/house_price_files/figure-html/saleprice1-1.png" width="60%" style="display: block; margin: auto;" />

``` r
# Remove SalePrice
data$SalePrice <- NULL
```

## 3.2 Exploring predictors of Sale Price

I wanted quickly see which predictor variables were important. I tried several packages including randomForest, earth, Step-wise Regression but failed as they can’t handle missing values. I haven’t done the imputing missing values at this stage as there are quite a lot variables (35) having missing values. I want to see if certain variables are worth to get them full :D.

So I tried party and it worked. It is a popular package for constructing decision trees and random forests.

### 3.2.1 Finding important predicitors

``` r
library(party)
set.seed(1234)
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

<img src="/project/Housing/house_price_files/figure-html/unnamed-chunk-9-1.png" width="80%" style="display: block; margin: auto;" />

- The most important variables are Neighborhood, GrLivArea and OverallQual. That makes sense to me.

### 3.3.2 Visualizing relationship of Log_SalePrice with most important variables.

**Log_saleprice vs. Neighborhood**

``` r
data_fullPrice <- data[!is.na(data$log_SalePrice),]
ggplot(data=data_fullPrice, aes(x = reorder(Neighborhood,log_SalePrice, FUN = median), y = log_SalePrice)) + 
  labs(x="Neighborhood") +
  geom_boxplot(fill =  "steelblue") + 
  coord_flip() +
  theme_bw() +
  geom_hline(yintercept= median(data_fullPrice$log_SalePrice), linetype="dashed", color = "red") # median log_SalePrice
```

<img src="/project/Housing/house_price_files/figure-html/unnamed-chunk-10-1.png" width="672" style="display: block; margin: auto;" />

**Log_SalePrice vs. OverallQual (r = 0.81)**

OverallQual: rating the overall material and finish of the house on a scale from very poor (1) to very excellent (10).

``` r
ggplot(data=data_fullPrice, aes(x=factor(OverallQual), y=log_SalePrice)) +
        geom_boxplot(fill = "steelblue") +
  labs(x="Overall Quality") +
  theme_bw()
```

<img src="/project/Housing/house_price_files/figure-html/unnamed-chunk-11-1.png" width="60%" style="display: block; margin: auto;" />

Graph shows the positive linear relationship between Log_SalePrice with Overal Quality. There are a few extreme points below housed with grade 3, 4, 7 and 10, and 1 point above of houses with grade 4.

**Log_SalePrice vs. GrLivArea (r = 0.7)**

GrLivArea: Above Grade Living Area

``` r
library(ggrepel)
data_fullPrice$name <- rownames(data_fullPrice)
ggplot(data=data_fullPrice, aes(x=GrLivArea, y=log_SalePrice)) +
        geom_point(color = "steelblue") + 
  geom_smooth(method = "lm", se = FALSE) +
  geom_text_repel(data = subset(data_fullPrice, GrLivArea > 4550), aes(label = name)) +
  theme_bw()
```

<img src="/project/Housing/house_price_files/figure-html/unnamed-chunk-12-1.png" width="60%" style="display: block; margin: auto;" />

### 3.2.3 Correlation matrix

``` r
# Selecting numeric variables
vars_num <- which(sapply(data, is.numeric))
data_varNum <- data[, vars_num] 

# Correlation of numeric variables
data_corr <- cor(data_varNum, use="pairwise.complete.obs")
#data_corr <-  vars_num %>% drop_na() %>% cor()

ggcorrplot(data_corr, type = "full", lab = TRUE, lab_size = 1.5, show.legend = TRUE, tl.cex = 5, ggtheme = ggplot2::theme_dark(), title = "Correlation of numeric predictors")
```

<img src="/project/Housing/house_price_files/figure-html/correlation-1.png" width="120%" style="display: block; margin: auto;" />

``` r
# Select high correlation (> 0.7) to detech multicollinear
corr_table <- melt(data_corr) %>% arrange(desc(value)) %>%
  mutate(value = round(value, digits = 4))%>%
  filter(value !=1)
  
(corr_high <- corr_table %>% filter(abs(value) > 0.7))
```

    ##             Var1          Var2  value
    ## 1     GarageArea    GarageCars 0.8897
    ## 2     GarageCars    GarageArea 0.8897
    ## 3    GarageYrBlt     YearBuilt 0.8348
    ## 4      YearBuilt   GarageYrBlt 0.8348
    ## 5  log_SalePrice   OverallQual 0.8172
    ## 6    OverallQual log_SalePrice 0.8172
    ## 7   TotRmsAbvGrd     GrLivArea 0.8084
    ## 8      GrLivArea  TotRmsAbvGrd 0.8084
    ## 9      X1stFlrSF   TotalBsmtSF 0.8017
    ## 10   TotalBsmtSF     X1stFlrSF 0.8017
    ## 11 log_SalePrice     GrLivArea 0.7009
    ## 12     GrLivArea log_SalePrice 0.7009

**Observation**:

- OverallQual and GrLivArea are hightly correlated with Log_SalePrice like we have found out in the previous session.

- There are some predictor variables are highly correlated to each other (r \> 0.7) including: GarageArea vs GarageCars; GarageYrBlt vs YearBuilt; GrLivArea vs TotalRmsAbvGrd; TotalBsmtSF vs X1stFlrSF. So there is multicollinear issue here that needs to be solved.

- Beside, YearBuilt and YearRemodAdd are also highly correlated to each other and have high correlction with Log_SalePrice (r \> 0.5).

continue….

## Data processing

### Imputing missing data

### label encoding

### Feature engineering

## Data Preparation for modelling

## Modelling




---

{{< figure src="featured.png" alt="Traditional right sidebar layout" caption="Photo by Phil Hearing on Unsplash" >}}

---




