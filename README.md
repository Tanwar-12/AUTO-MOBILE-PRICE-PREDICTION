# *MACHINE LEARNING PROJECT*
## **AUTO-MOBILE PRICE PREDICTION**
### **1. BUSINESS CASE: THE AIM IS TO PREDICT THE PRICE OF CAR USING ALL THE GIVEN FEATURES**
### 2. **IMPORTING THE PYTHON LIBRARIES**
### 3. **LOADING THE DATASET**
### 4. **DOMAIN ANALYSIS**
  1) **symboling**:            This rating corresponds to the degree to which the auto is more risky than its price indicates.
      * Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less), this symbol is
      adjusted by moving it up (or down) the scale.
      * Actuarians call this process "symboling".  A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe. (-3, -2, -1, 0, 1, 2, 3.)<br>
        
  2) **normalized-losses:** This factor is the relative average loss payment per insured vehicle year.  This value is normalized for all autos within a
      particular size classification (two-door small, station wagons,sports/speciality, etc...), and represents the average loss per car
      per year. (continuous from 65 to 256)<br>
  3) **make**:  This represents the maker/make of auto. (alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo)<br>
  4) **fuel-type:**                Type of fuel used in engine. (diesel, gas)<br> 
  5) **aspiration:**               If the auto has turbo or natually aspirated engine. (std, turbo)<br>
  6) **num-of-doors:**             Number of doors in auto. (four, two)<br>
  7) **body-style:**               Type/style of auto body. (hardtop, wagon, sedan, hatchback, convertible)<br>
  8) **drive-wheels:**             The wheel drive system which transmits force causing the auto to move. (4wd, fwd, rwd)<br>
  9) **engine-location:**          Placement of engine in auto. (front, rear)<br>
 10) **wheel-base:**               The distance between the wheel axles - centers of front and rear wheels. (continuous from 86.6 120.9)<br>
 11) **length:**                   Length of auto. (continuous from 141.1 to 208.1)<br>
 12) **width:**                    Width of auto. (continuous from 60.3 to 72.3)<br>
 13) **height:**                   Height of auto. (continuous from 47.8 to 59.8)<br>
 14) **curb-weight:**              Weight of the car with standard components. (continuous from 1488 to 4066)<br>
 15) **engine-type:**              dohc(double overhead camp), dohcv, l, ohc, ohcf, ohcv(overhead camp valve), rotor.<br>
 16) **num-of-cylinders:**         Number of cylinders used in auto. (eight, five, four, six, three, twelve, two)<br>
 17) **engine-size:**              Size of auto engine. (continuous from 61 to 326)<br>
 18. **fuel-system:**              The system that helps transfer fuel to the engine. (1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi)<br>
 19) **bore:**                     Hollow part inside engine/inner diameter of the cylinder. (continuous from 2.54 to 3.94)<br>
 20) **stroke:**                   The full travel of piston along the cylinder. (continuous from 2.07 to 4.17)<br>
 21) **compression-ratio:**        The ratio of maximum to minimum volume in the cylinder of an internal combustion engine. (continuous from 7 to 23<br>
 22) **horsepower:**               Power that an engine produces. (continuous from 48 to 288)<br>
 23) **peak-rpm:**                 RPM that the engine produces at highest horsepower. (continuous from 4150 to 6600)<br>
 24) **city-mpg:**                 Lowest mpg rating for an auto. (continuous from 13 to 49)<br>
 25) **highway-mpg:**              Highest/average mpg rating of an auto while driving on an open stretch road. (continuous from 16 to 54)<br>
 26) **price:**                    Cost of auto. (continuous from 5118 to 45400)
###  *BASIC CHECKS*
## 5.**EXPLORATORY DATA ANALYSIS(EDA)**
### 5.1 UNIVARIATE ANALYSIS:
### DATA INSIGHTS:
   - More than 30% of cars have a risk value of 0 which means that its averagly safe to buy.<br>
   - 15% of the cars are of the make Toyota.<br>
   -  90% of cars have gas engine and only 10% have diesel fuel engine.<br>
   - 82% of cars have naturally aspirated engine and 12% have turbo engines.<br>
   - 56% of cars have 4 doors, 43% have 2 doors.<br>
   - More than 45% of cars are Sedan and around 34% are hatchbacks.<br>
   - Almost 60% of cars have forward wheel drive.<br>
   - Just 1% of cars have engine located in the rear end of car.<br>
   - For most of the cars, wheel base is around 95 to 100.<br>
   - Around 25% of cars are of length 170units. This feature shows almost normal distribution.<br>
   - Around 25% of cars are of width range 66 to 68units.<br>
   - Most of the cars are of height range 55 to 56units.<br>
   - 25% of cars have curb weight of 2500units.<br>
   -  More than 70% of cars have ohc (OverHead Camp) kind of engine.<br>
   - Almost 80% of cars have 4 cylinders.<br>
   - 45% of car's engine size is 100.<br>
   - Around 45% of cars have mpfi kind of fuel system.<br>
   - Most of the car's bore is more than 3.8units.<br>
   - Most of the car's stroke length is more than 3.5units.<br>
   - 65% of cars have compression ratio of 10<br>
   - More than 18% of cars have a horsepower of 68hp.<br>
   - Most of the cars have peak rpm of 5500 or 4800.<br>
   - Around 30% of cars give a milege of 25 in city.<br>
   - More than 20% of cars give a milege in the range of 25 to 35 in highway.<br>
   - 40% of cars have a price range of 5 to 10k.
  ### 5.2 BIVARIATE ANALYSIS:
  ![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/fb04fc94-d3b7-4226-8c16-bd03c6f5341b)
  #### DATA INSIGHTS:
   - Price of Jaguar cars is the highest.<br>
 - The price of diesel engine cars is higher than gas engine cars.<br>
 - Turbo engined cars are pricey than the naturally aspirated cars.<br>
 - Cars with 4 doors are costlier than 2 door cars.<br>
 - Convertible and Hardtop type of cars have higher price than other types.<br>
 - Reverse wheel drive cars have higher price than forward or 4 wheel drive cars.<br>
 - Rear engine cars are costlier than front engine cars.<br>
 - OverHead Camp Valve (ohcv) engine have higher price.<br>
 - 8 cylinder engine cars have highest price.<br>
 - mpfi fuel system car has the highest price.
![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/4f3d86c1-2729-4358-ab43-3e6cf0eda470)
#### DATA INSIGHTS:
 - At an average normalized loss of 142, the price of cars is 35k.<br>
 - Cars with bore measuring 3.8units has the highest price.<br>
 - Cars with stroke length of 2.76units has the highest price.<br>
 - A horsepower of 184 is paid the highest price.<br>
 - Car with peak RPM of 5900 is the costliest.<br>
 - A car with symboling value of -1 has higher price than that with value of 3.<br>
 - Car with wheel base of 112 is paid highest.<br>
 - Car with length of 199.2, width of 72 and height of 55.4 is highly paid.<br>
 - Car with curb weight of 3715 is highly paid.<br>
 - Engine size of 304 has highest price.<br>
 - Compression ratio of 11.5 is paid more.<br>
 - A city mpg of 14 is sold at highest price.<br>
 - A highway mpg of 16 is sold at highest price.
   ### 5.3  MULTIVARIATE ANALYSIS:
   ![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/8fa90d5e-bd40-4910-9334-233d3516338b)
   

   ![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/7f729201-9ddd-4163-90c7-fe0bc9d0eeee)
   * Data Insight:Marcedes-benz with gas fuel type has highest price
     ![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/dbd6e192-e45e-4deb-8f7f-9da3b8d2ce7c)
   * Data Insight: With increase in weight of vehicle, city mpg decreases.
  
   ##  6.**FEATURE ENGINERING/ DATA PREPROCESSING**
   #### 6.1 Checking for missing or null values
   #### 6.2  Converting categorical data to numerical data
   #### 6.3 Using Label Encoder
   ####  6.4 **HANDLING OUTLIERS**
   ![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/4ac41b78-d470-4fc2-a448-407eae2072ea)

   ## 7.**FEATURE SELECTION**
   ### 7.1 Checking for correlation
![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/a0ed1f9e-ef71-448c-9dd2-7e7c994558ea)

* USE OF HEAT MAP:
 ![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/15ae55e0-3c77-4e95-80d6-8f05625c3497)
## 8.**MODEL BUILDING & EVALUATION**
## **LINEAR REGRESSION**:
- Training R2 accuracy using Linear Regression is:  86.53144036257537
- Testing R2 accuracy using Linear Regression is:  76.72938159080243
- Testing Adjusted R2 score is:  48.28751464622761
- MSE score is:  28470850.70003938
- RMSE score is:  5335.808345512363
- MAE score is:  3406.7596001032775


## **K-Neighbors Regressor**:

- Training R2 accuracy using KNN Regression is:  82.50133136918122
- Testing R2 accuracy using KNN Regression is:  64.93533173518867
- Testing Adjusted R2 score is:  22.078514967085916
- MSE score is:  42900490.114146344
- RMSE score is:  6549.846571801995
- MAE score is:  3946.4146341463415

## SVM:
- Training R2 accuracy using SVM Regression is:  -9.315945709087071
- Testing R2 accuracy using SVM Regression is:  -21.91990376994277
- Testing Adjusted R2 score is:  -170.9331194887617
- MSE score is:  149165068.0080445
- RMSE score is:  12213.315193183402
- MAE score is:  7943.122270031737

## DECISION TREE REGRESSION:
- Training R2 accuracy using Decision tree Regression is:  99.89831862870213
- Testing R2 accuracy using Decision tree Regression is:  89.4528958266559
- Testing Adjusted R2 score is:  76.56199072590202
- MSE score is:  12904041.609756097
- RMSE score is:  3592.219593754827
- MAE score is:  1950.3902439024391

## GRADIENT BOOSTING:
- The training R2 accuracy using Gradient Boosting is: 99.22053302005716
- The testing R2 accuracy using Gradient Boosting is: 94.9681530034094
- Testing Adjusted R2 score is:  88.81811778535422
- MSE score is:  6156302.426786809
- RMSE score is:  2481.1897200308586
- MAE score is:  1656.7798100138448

## XGBOOST REGRESSOR:
- The training R2 accuracy using XG Boost is: 99.89824021512386
- The testing R2 accuracy using XB Boost is: 92.59952491032328
- Testing Adjusted R2 score is:  83.5544998007184
- MSE score is:  9054242.465007683
- RMSE score is:  3009.026830223965
- MAE score is:  1908.5585580221036


## RANDOM FOREST REGRESSOR
- The training R2 accuracy using Random Forest is: 96.98578103166696
- The testing R2 accuracy using Random Forest is: 86.21079174165337
- Testing Adjusted R2 score is:  69.35731498145194
- MSE score is:  16870651.33773875
- RMSE score is:  4107.389844869701
- MAE score is:  2489.5789674796747

## AFTER HPYERPARAMETER TUNING
- Training R2 score after Hyperparameter tuning on Random forest is: 99.89831862870213
- Testing R2 score after Hyperparameter tuning on Random forest is: 94.451462613851
- Testing Adjusted R2 score is:  87.6699169196689
- MSE score is:  6788456.445239862
- RMSE score is:  2605.466646349529
- MAE score is:  1553.0564634146342

**So far, Gradient Boosting and Random Forest algorithms have given the best scores:**
 #### GRADIENT BOOSTING
* The training R2 accuracy using Gradient Boosting is: 99.22053302005716<br>
* The testing R2 accuracy using Gradient Boosting is: 94.9681530034094<br>
* Testing Adjusted R2 score is:  88.81811778535422<br>
* MSE score is:  6156302.426786809<br>
* RMSE score is:  2481.1897200308586<br>
* MAE score is:  1656.7798100138448<br>
  #### RANDOM FOREST  
* Training R2 score after Hyperparameter tuning on Random forest is: 99.89831862870213<br>
* Testing R2 score after Hyperparameter tuning on Random forest is: 94.451462613851<br>
* Testing Adjusted R2 score is:  87.6699169196689<br>
* MSE score is:  6788456.445239862<br>
* RMSE score is:  2605.466646349529<br>
* MAE score is:  1553.0564634146342

  ## CROSS VALIDATION SCORES:
  [ 0.6154544   0.36697642 -0.85743858]
- Cross validation score of Linear Regression model is: 0.041664079961660404
[ 0.30571883  0.55311535 -0.69954503]
- Cross validation score of KNN model is: 0.05309638233246209
[-0.20935183 -0.2448066  -0.01494369]
- Cross validation score of SVR model is: -0.15636737650881907
[0.35543453 0.4146508  0.12383546]
- Cross validation score of Decision Tree model is: 0.2979735956409533
[0.56014578 0.65703396 0.39638363]
- Cross validation score of Gradient Boost model is: 0.5378544577413296
[0.62451125 0.61963064 0.65658258]
- Cross validation score of XG Boost model is: 0.6335748203051849
[0.76522458 0.62552531 0.63047936]
- Cross validation score of Random Forest model is: 0.6737430824341969

- Training R2 score after Hyperparameter tuning on Random forest is: 99.89831862870213
- Testing R2 score after Hyperparameter tuning on Random forest is: 94.451462613851
- Testing Adjusted R2 score is: 87.6699169196689
- MSE score is: 6788456.445239862
- RMSE score is: 2605.466646349529
- MAE score is: 1553.0564634146342
- Cross validation score of Random Forest model is: 0.6737430824341969

  ![image](https://github.com/Tanwar-12/AUTO-MOBILE-PRICE-PREDICTION/assets/110081008/82b1128c-d858-473e-827e-8a190611f2eb)

  ### **RESULT:** **Since Random Forest model comparitively has higher train, test and Cross validation scores and lower MSE, RMSE, MAE score, we choose this model for this problem**









 
