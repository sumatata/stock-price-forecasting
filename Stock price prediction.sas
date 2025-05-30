LIBNAME RA '/home/u63992188/STAT674/RA';

PROC EXPORT DATA= RA.Final_Presenatation_dataset
    OUTFILE="/home/u63992188/STAT674/RA/Project.csv"
    DBMS=CSV
    REPLACE;
RUN;


proc contents data= RA.reduced_dataset_renamed;
run;

PROC PRINT DATA=RA.reduced_dataset_renamed (OBS=10); 
RUN;

PROC SGSCATTER DATA=RA.reduced_dataset_renamed; /* Replace with your dataset name */
    MATRIX close Capital_Expenditures Current_Ratio
           Gross_Margin Gross_Profit Inventory Long_Term_Debt
           Net_Cash_Flow Pre_Tax_Margin Profit_Margin
           Research_and_Development Retained_Earnings
           high low open volume /
           DIAGONAL=(histogram); /* Adds histograms on the diagonal */
    TITLE "Scatterplot Matrix: All Variables in the Dataset";
RUN;

/* Boxplot: Response Variable (close) by Categorical Variable (GICS_Sector) */
PROC SGPLOT DATA=RA.reduced_dataset_renamed; /* Replace with your dataset name */
    VBOX close / CATEGORY=GICS_Sector;
    TITLE "Boxplot of Close (Response Variable) by GICS Sector";
RUN;

/* Histogram for Close */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM close / BINWIDTH=0.5; /* Adjust BINWIDTH as needed */
    DENSITY close; /* Adds a density curve */
    TITLE "Histogram of Close (Response Variable)";
RUN;

/* Histogram for Current_Ratio */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Current_Ratio / BINWIDTH=0.1; /* Adjust BINWIDTH for granularity */
    DENSITY Current_Ratio;
    TITLE "Histogram of Current Ratio";
RUN;

/* Histogram for Gross_Margin */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Gross_Margin / BINWIDTH=5; /* Adjust BINWIDTH as needed */
    DENSITY Gross_Margin;
    TITLE "Histogram of Gross Margin";
RUN;

/* Histogram for Gross_Profit */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Gross_Profit / BINWIDTH=1000000000; /* Adjust BINWIDTH */
    DENSITY Gross_Profit;
    TITLE "Histogram of Gross Profit";
RUN;

/* Histogram for Inventory */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Inventory / BINWIDTH=500000000; /* Adjust BINWIDTH */
    DENSITY Inventory;
    TITLE "Histogram of Inventory";
RUN;

/* Histogram for Long_Term_Debt */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Long_Term_Debt / BINWIDTH=500000000; /* Adjust BINWIDTH */
    DENSITY Long_Term_Debt;
    TITLE "Histogram of Long Term Debt";
RUN;

/* Histogram for Net_Cash_Flow */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Net_Cash_Flow / BINWIDTH=500000000; /* Adjust BINWIDTH */
    DENSITY Net_Cash_Flow;
    TITLE "Histogram of Net Cash Flow";
RUN;

/* Histogram for Pre_Tax_Margin */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Pre_Tax_Margin / BINWIDTH=5; /* Adjust BINWIDTH */
    DENSITY Pre_Tax_Margin;
    TITLE "Histogram of Pre Tax Margin";
RUN;

/* Histogram for Profit_Margin */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Profit_Margin / BINWIDTH=5; /* Adjust BINWIDTH */
    DENSITY Profit_Margin;
    TITLE "Histogram of Profit Margin";
RUN;

/* Histogram for Retained_Earnings */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM Retained_Earnings / BINWIDTH=500000000; /* Adjust BINWIDTH */
    DENSITY Retained_Earnings;
    TITLE "Histogram of Retained Earnings";
RUN;

/* Histogram for High */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM high / BINWIDTH=0.5; /* Adjust BINWIDTH */
    DENSITY high;
    TITLE "Histogram of High Prices";
RUN;

/* Histogram for Low */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM low / BINWIDTH=0.5; /* Adjust BINWIDTH */
    DENSITY low;
    TITLE "Histogram of Low Prices";
RUN;

/* Histogram for Open */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM open / BINWIDTH=0.5; /* Adjust BINWIDTH */
    DENSITY open;
    TITLE "Histogram of Open Prices";
RUN;

/* Histogram for Volume */
PROC SGPLOT DATA=RA.reduced_dataset_renamed;
    HISTOGRAM volume / BINWIDTH=1000000; /* Adjust BINWIDTH */
    DENSITY volume;
    TITLE "Histogram of Volume";
RUN;

PROC CORR DATA=RA.reduced_dataset_renamed;
    VAR close; /* Response variable */
    WITH Current_Ratio
         Gross_Margin Gross_Profit Inventory
         Net_Cash_Flow Pre_Tax_Margin Profit_Margin
         high low open volume; /* Predictors */
    TITLE "Correlation Between Response Variable (close) and Predictors";
RUN;

/*ME*/

DATA work.reduced_dataset_with_price_range;
    SET RA.reduced_dataset_renamed; 
    /* Calculate Price_Range as the relative price fluctuation */
    Price_Range = (high - low) / open;
RUN;

data RA.Final_Presenatation_dataset;
	set work.reduced_dataset_with_price_range;
	keep close Current_Ratio
         Gross_Margin Gross_Profit Inventory
         Pre_Tax_Margin Price_Range
         volume GICS_Sector;
run;
	
/* Summary statistics using PROC MEANS */
proc means data=RA.Final_Presenatation_dataset n mean std min max;
	var close Current_Ratio Gross_Margin Gross_Profit 
        Inventory Pre_Tax_Margin volume;
run;

/* Summary statistics for categorical variables using PROC FREQ */
proc freq data=RA.Final_Presenatation_dataset;
	tables Price_Range GICS_Sector / nocum nopercent;
run;


/*transformations*/
/*box-cox*/
DATA scaled_dataset;
    SET transformed_neg025;
    scaled_Current_Ratio = Current_Ratio;
    scaled_Gross_Margin = Gross_Margin;
    scaled_Gross_Profit = Gross_Profit / 1E6; /* Scale down by 1 million */
    scaled_Inventory = Inventory / 1E6; /* Scale down by 1 million */
    scaled_Pre_Tax_Margin = Pre_Tax_Margin;
    scaled_Price_Range = Price_Range;
    scaled_volume = volume / 1E6; /* Scale down by 1 million */
RUN;


PROC TRANSREG DATA=scaled_dataset;
    MODEL BOXCOX(transformed_close) = IDENTITY(scaled_Current_Ratio scaled_Gross_Margin 
                                   scaled_Gross_Profit scaled_Inventory 
                                   scaled_Pre_Tax_Margin scaled_Price_Range 
                                   scaled_volume);
RUN;

DATA transformed_dataset;
    SET RA.Final_Presenatation_dataset;
    sqrt_close = SQRT(close); /* Square root transformation for close */
RUN;

DATA transformed_neg025;
    SET RA.Final_Presenatation_dataset;
    transformed_close = (close**(-0.25) - 1) / -0.25;
RUN;

DATA transformed_neg025;
    SET RA.Final_Presenatation_dataset;
    transformed_close = (close**2.25 - 1) / 2.25;
RUN;


/*inverse*/
	
PROC REG DATA=transformed_neg025 PLOTS(MAXPOINTS= none);
    MODEL transformed_close = Current_Ratio
         					  Gross_Margin Gross_Profit Inventory
         					  Pre_Tax_Margin Price_Range
        					  volume;
    OUTPUT OUT=predicted_results P=predicted_values R=residuals;
RUN;

PROC SGPLOT DATA=predicted_results;
    SCATTER X=predicted_values Y=residuals / MARKERATTRS=(SYMBOL=CircleFilled);
    REG X=predicted_values Y=residuals / DEGREE=2; /* Quadratic trendline */
    TITLE "Inverse Fitted Plot for Overall Model Residuals";
RUN;

/*individual inverse*/

/* Generate Inverse Fitted Plots for Each Predictor Individually */
ods graphics on;
proc reg data=transformed_neg025 PLOTS(MAXPOINTS= none);
    /* Model with all predictors included */
    model transformed_close = Current_Ratio
         					  Gross_Margin Gross_Profit Inventory
         					  Pre_Tax_Margin Price_Range
        					  volume;
    /* Create inverse fitted plots for individual predictors */
    output out=reg_out predicted=predicted_close residual=residual_close;
run;
quit;

/* For each predictor, plot inverse fitted plot */
proc sgplot data=reg_out;
    scatter x=Price_Range y=residual_close / markerattrs=(symbol=circlefilled);
    reg x=Price_Range y=residual_close / degree=1;
    title "Inverse Fitted Plot for Price_Range";
run;

proc sgplot data=reg_out;
    scatter x=Volume y=residual_close / markerattrs=(symbol=circlefilled);
    reg x=Volume y=residual_close / degree=1;
    title "Inverse Fitted Plot for Volume";
run;

proc sgplot data=reg_out;
    scatter x=Pre_Tax_Margin y=residual_close / markerattrs=(symbol=circlefilled);
    reg x=Pre_Tax_Margin y=residual_close / degree=1;
    title "Inverse Fitted Plot for Pre_Tax_Margin";
run;

proc sgplot data=reg_out;
    scatter x=Current_Ratio y=residual_close / markerattrs=(symbol=circlefilled);
    reg x=Current_Ratio y=residual_close / degree=1;
    title "Inverse Fitted Plot for Current_Ratio";
run;

proc sgplot data=reg_out;
    scatter x=Inventory y=residual_close / markerattrs=(symbol=circlefilled);
    reg x=Inventory y=residual_close / degree=1;
    title "Inverse Fitted Plot for Inventory";
run;

proc sgplot data=reg_out;
    scatter x=Gross_Margin y=residual_close / markerattrs=(symbol=circlefilled);
    reg x=Gross_Margin y=residual_close / degree=1;
    title "Inverse Fitted Plot for Gross_Margin";
run;


proc sgplot data=reg_out;
    scatter x=Gross_Profit y=residual_close / markerattrs=(symbol=circlefilled);
    reg x=Gross_Profit y=residual_close / degree=1;
    title "Inverse Fitted Plot for Gross_Profit";
run;

/* For GICS_Sector, check residuals grouped by category */
proc sgplot data=reg_out;
    vbox residual_close / category=GICS_Sector;
    title "Residual Distribution by GICS_Sector";
run;
ods graphics off;


/*inverse transformation*/

DATA RA.Final_Transformed_Dataset;
    SET reg_out; /* Replace with your dataset */
    
    /* Handle zero and negative values for log transformations */
    IF Price_Range > 0 THEN log_Price_Range = LOG(Price_Range);
    ELSE log_Price_Range = .; /* Set to missing if invalid */
    
    IF Volume > 0 THEN log_Volume = LOG(Volume);
    ELSE log_Volume = .; /* Set to missing if invalid */
    
    IF Inventory > 0 THEN log_Inventory = LOG(Inventory);
    ELSE log_Inventory = .; /* Set to missing if invalid */
    
    IF Gross_Profit > 0 THEN log_Gross_Profit = LOG(Gross_Profit);
    ELSE log_Gross_Profit = .; /* Set to missing if invalid */
    
    /* Safe transformations for other predictors */
    sqrt_Pre_Tax_Margin = SQRT(Pre_Tax_Margin); /* Square root is valid for zero */
    sqrt_Gross_Margin = SQRT(Gross_Margin); /* Square root is valid for zero */
RUN;
	
proc contents data=RA.Final_Transformed_Dataset;
run;
	
/*split for train and test*/
PROC SURVEYSELECT DATA= Ra.final_transformed_dataset
    OUT=work.split_dataset 
    SAMPRATE=0.7 
    OUTALL;
RUN;

DATA train test;
    SET work.split_dataset;
    IF Selected=1 THEN OUTPUT train;
    ELSE OUTPUT test;
RUN;

/*training*/

/* Stepwise Regression */

PROC GLMSELECT DATA=train;
    CLASS GICS_Sector; /* Declare GICS_Sector as a categorical variable */
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin volume Price_Range 
                  GICS_Sector
                  / SELECTION=STEPWISE(STEPS=20) STATS=ALL SHOWPVALUES;
    OUTPUT OUT=stepwise_out PREDICTED=predicted RESIDUAL=residual;
RUN;


PROC GLMSELECT DATA=train;
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin 
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume
                  / SELECTION=FORWARD(STEPS=20) STATS=ALL SHOWPVALUES;
    TITLE "Forward Selection Model";
RUN;

/* LASSO Regression */
PROC GLMSELECT DATA=train;
    CLASS GICS_Sector; /* Specify GICS_Sector as categorical */
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin 
                  log_Gross_Profit sqrt_Pre_Tax_Margin volume Price_Range
                  GICS_Sector
                  / SELECTION=LASSO(STEPS=10 CHOOSE=CV) CVMETHOD=RANDOM(5) STATS=ALL;
    OUTPUT OUT=lasso_out P=predicted R=residual;
RUN;
QUIT;

/*model diagnosis*/

/* 1. Residual Plots for Stepwise Regression */
PROC SGPLOT DATA=stepwise_out;
    SCATTER X=predicted Y=residual / MARKERATTRS=(SYMBOL=circlefilled);
    REFLINE 0 / AXIS=Y LINEATTRS=(PATTERN=SHORTDASH COLOR=RED);
    TITLE "Residuals vs Predicted Values for Stepwise Regression";
RUN;

/* 2. Test for Curvature in Stepwise Regression */
PROC REG DATA=stepwise_out PLOTS(MAXPOINTS= none);
    MODEL residual = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin 
                     log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume;
    TITLE "Curvature Test for Stepwise Model";
RUN;

/* 3. Test for Non-constant Variance in Stepwise Regression */
PROC GLM DATA=stepwise_out PLOTS(MAXPOINTS= none);
    MODEL residual = predicted / SOLUTION;
    OUTPUT OUT=heterosced_test_stepwise RSTUDENT=std_residual;
    TITLE "Test for Non-constant Variance in Stepwise Model";
RUN;

/* 4. QQ Plot for Stepwise Residuals */
PROC UNIVARIATE DATA=stepwise_out;
    VAR residual;
    QQPLOT / NORMAL(MU=EST SIGMA=EST COLOR=RED);
    TITLE "QQ Plot for Stepwise Residuals";
RUN;

/* 5. Cook's Distance for Stepwise Regression */

PROC REG DATA=stepwise_out PLOTS(MAXPOINTS=none ONLY)=(COOKSD RSTUDENTBYPREDICTED);
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin volume Price_Range;
    OUTPUT OUT=cooks_dist_stepwise R=residual P=predicted;
    TITLE "Cook's Distance for Stepwise Model";
RUN;

/*lasso*/

/* 1. Residual Plots for LASSO Regression */
PROC SGPLOT DATA=lasso_out;
    SCATTER X=predicted Y=residual / MARKERATTRS=(SYMBOL=circlefilled);
    REFLINE 0 / AXIS=Y LINEATTRS=(PATTERN=SHORTDASH COLOR=RED);
    TITLE "Residuals vs Predicted Values for LASSO Regression";
RUN;

/* 2. Test for Curvature in LASSO Regression */
PROC REG DATA=lasso_out PLOTS(MAXPOINTS=none);
    MODEL residual = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin 
                     log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume;
    TITLE "Curvature Test for LASSO Model";
RUN;

/* 3. Test for Non-constant Variance in LASSO Regression */
PROC GLM DATA=lasso_out PLOTS(MAXPOINTS=none);
    MODEL residual = predicted / SOLUTION;
    OUTPUT OUT=heterosced_test_lasso RSTUDENT=std_residual;
    TITLE "Test for Non-constant Variance in LASSO Model";
RUN;

/* 4. QQ Plot for LASSO Residuals */
PROC UNIVARIATE DATA=lasso_out;
    VAR residual;
    QQPLOT / NORMAL(MU=EST SIGMA=EST COLOR=RED);
    TITLE "QQ Plot for LASSO Residuals";
RUN;

/* 5. Cook's Distance for LASSO Regression */
PROC REG DATA=lasso_out PLOTS(MAXPOINTS=none ONLY)=(COOKSD RSTUDENTBYPREDICTED);
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin volume Price_Range;
    OUTPUT OUT=cooks_dist_lasso R=residual P=predicted;
    TITLE "Cook's Distance for LASSO Model";
RUN;

/*selection tools*/

/* 1. Stepwise Regression */
PROC GLMSELECT DATA=train;
    CLASS GICS_Sector; /* Declare GICS_Sector as categorical */
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin volume Price_Range 
                  GICS_Sector
                  / SELECTION=STEPWISE(STEPS=20) STATS=ALL SHOWPVALUES;
    OUTPUT OUT=stepwise_out PREDICTED=predicted RESIDUAL=residual; /* Include Residual */
    STORE OUT=stepwise_model; /* Store the Stepwise Model */
RUN;

/* 2. LASSO Regression */
PROC GLMSELECT DATA=train;
    CLASS GICS_Sector; /* Declare GICS_Sector as categorical */
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin volume Price_Range 
                  GICS_Sector
                  / SELECTION=LASSO(STEPS=10 CHOOSE=CV) CVMETHOD=RANDOM(5) STATS=ALL;
    OUTPUT OUT=lasso_out PREDICTED=predicted RESIDUAL=residual; /* Include Residual */
    STORE OUT=lasso_model; /* Store the LASSO Model */
RUN;

/* 3. Summarize Residuals for Stepwise */
PROC MEANS DATA=stepwise_out N MEAN STD MAX MIN;
    VAR residual;
    TITLE "Stepwise Model Residual Summary";
RUN;

/* 4. Summarize Residuals for LASSO */
PROC MEANS DATA=lasso_out N MEAN STD MAX MIN;
    VAR residual;
    TITLE "LASSO Model Residual Summary";
RUN;

/* 5. Compare Predicted vs Residual */
PROC MEANS DATA=stepwise_out N MEAN STD MAX MIN;
    VAR predicted residual;
    TITLE "Stepwise Model Predicted and Residual Summary";
RUN;

PROC MEANS DATA=lasso_out N MEAN STD MAX MIN;
    VAR predicted residual;
    TITLE "LASSO Model Predicted and Residual Summary";
RUN;

/*result*/

PROC SGSCATTER DATA=RA.Final_Transformed_Dataset;
    MATRIX close log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
           log_Gross_Profit sqrt_Pre_Tax_Margin volume /
           DIAGONAL=(histogram);
    TITLE "Scatterplot Matrix: Final Predictors and Response Variable";
RUN;

PROC CORR DATA=RA.Final_Transformed_Dataset;
    VAR close; /* Response Variable */
    WITH log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin 
         log_Gross_Profit sqrt_Pre_Tax_Margin volume; /* Predictors */
    TITLE "Correlation Matrix: Response Variable and Predictors";
RUN;

/* Extreme Value Analysis for All Predictors */

libname RA '/home/u63992188/STAT674/RA';

/* 1. Scatter Plot: log_Price_Range vs Close */
PROC SGPANEL DATA=RA.Final_Transformed_Dataset;
    PANELBY GICS_Sector / NOVARNAME;
    SCATTER X=log_Price_Range Y=close / MARKERATTRS=(SYMBOL=CircleFilled);
    TITLE "Scatterplot of Close vs log_Price_Range by GICS Sector";
RUN;

/* 2. Scatter Plot: log_Inventory vs Close */
PROC SGPANEL DATA=RA.Final_Transformed_Dataset;
    PANELBY GICS_Sector / NOVARNAME;
    SCATTER X=log_Inventory Y=close / MARKERATTRS=(SYMBOL=CircleFilled);
    TITLE "Scatterplot of Close vs log_Inventory by GICS Sector";
RUN;

/* 3. Scatter Plot: sqrt_Gross_Margin vs Close */
PROC SGPANEL DATA=RA.Final_Transformed_Dataset;
    PANELBY GICS_Sector / NOVARNAME;
    SCATTER X=sqrt_Gross_Margin Y=close / MARKERATTRS=(SYMBOL=CircleFilled);
    TITLE "Scatterplot of Close vs sqrt_Gross_Margin by GICS Sector";
RUN;

/* 4. Scatter Plot: log_Gross_Profit vs Close */
PROC SGPANEL DATA=RA.Final_Transformed_Dataset;
    PANELBY GICS_Sector / NOVARNAME;
    SCATTER X=log_Gross_Profit Y=close / MARKERATTRS=(SYMBOL=CircleFilled);
    TITLE "Scatterplot of Close vs log_Gross_Profit by GICS Sector";
RUN;

/* 5. Scatter Plot: sqrt_Pre_Tax_Margin vs Close */
PROC SGPANEL DATA=RA.Final_Transformed_Dataset;
    PANELBY GICS_Sector / NOVARNAME;
    SCATTER X=sqrt_Pre_Tax_Margin Y=close / MARKERATTRS=(SYMBOL=CircleFilled);
    TITLE "Scatterplot of Close vs sqrt_Pre_Tax_Margin by GICS Sector";
RUN;

/* 6. Scatter Plot: volume vs Close */
PROC SGPANEL DATA=RA.Final_Transformed_Dataset;
    PANELBY GICS_Sector / NOVARNAME;
    SCATTER X=volume Y=close / MARKERATTRS=(SYMBOL=CircleFilled);
    TITLE "Scatterplot of Close vs Volume by GICS Sector";
RUN;

/* 7. Scatter Plot: log_Volume vs Close */
PROC SGPANEL DATA=RA.Final_Transformed_Dataset;
    PANELBY GICS_Sector / NOVARNAME;
    SCATTER X=log_Volume Y=close / MARKERATTRS=(SYMBOL=CircleFilled);
    TITLE "Scatterplot of Close vs log_Volume by GICS Sector";
RUN;

/* 8. Boxplot: Close by GICS_Sector */
PROC SGPLOT DATA=RA.Final_Transformed_Dataset;
    VBOX close / CATEGORY=GICS_Sector;
    TITLE "Boxplot of Close by GICS Sector";
RUN;

/*Anova table*/

PROC GLM DATA=train;
    CLASS GICS_Sector; /* Specify the categorical variable */
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin 
                  log_Gross_Profit sqrt_Pre_Tax_Margin volume Price_Range 
                  GICS_Sector / SOLUTION;
    OUTPUT OUT=final_model_out P=predicted R=residual;
    TITLE "Final Model: Parameter Estimates and ANOVA Table";
RUN;

/* Pairwise Comparisons for GICS_Sector */
PROC GLM DATA=train PLOTS(MAXPOINTS= none );
    CLASS GICS_Sector; /* Specify the categorical variable */
    MODEL close = GICS_Sector;
    LSMEANS GICS_Sector / PDIFF=ALL CL ADJUST=TUKEY;
    TITLE "Pairwise Comparisons for GICS_Sector Levels";
RUN;

PROC PLM RESTORE=final_model_store;
    LSMEANS GICS_Sector / DIFF=ALL CL;
    TITLE "Pairwise Comparisons for GICS_Sector";
RUN;

/*Effectplots*/

PROC GLMSELECT DATA=train;
    CLASS GICS_Sector; /* Specify categorical variable */
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin 
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume 
                  GICS_Sector
                  / SELECTION=STEPWISE(STEPS=20) STATS=ALL SHOWPVALUES;
    STORE OUT=final_model_store; /* Save the model */
    TITLE "Stepwise Model with Stored Output";
RUN;

/* Effect Plots for Each Predictor */

PROC PLM RESTORE=final_model_store;
    EFFECTPLOT SLICEFIT(X=log_Price_Range) / CLM;
    TITLE "Effect Plot for log_Price_Range";

    EFFECTPLOT SLICEFIT(X=log_Volume) / CLM;
    TITLE "Effect Plot for log_Volume";

    EFFECTPLOT SLICEFIT(X=log_Inventory) / CLM;
    TITLE "Effect Plot for log_Inventory";

    EFFECTPLOT SLICEFIT(X=sqrt_Gross_Margin) / CLM;
    TITLE "Effect Plot for sqrt_Gross_Margin";

    EFFECTPLOT SLICEFIT(X=log_Gross_Profit) / CLM;
    TITLE "Effect Plot for log_Gross_Profit";

    EFFECTPLOT SLICEFIT(X=sqrt_Pre_Tax_Margin) / CLM;
    TITLE "Effect Plot for sqrt_Pre_Tax_Margin";

    EFFECTPLOT SLICEFIT(X=Price_Range) / CLM;
    TITLE "Effect Plot for Price_Range";

RUN;

PROC PLM RESTORE=final_model_store;
    EFFECTPLOT INTERACTION(X=GICS_Sector) / CLM;
    TITLE "Effect Plot for GICS_Sector Levels";
RUN;

/* Include Interaction Terms */

PROC GLMSELECT DATA=train;
    CLASS GICS_Sector;
    MODEL close = log_Price_Range|log_Volume|log_Inventory|sqrt_Gross_Margin|
                  log_Gross_Profit|sqrt_Pre_Tax_Margin|Price_Range|volume|GICS_Sector @2
                  / SELECTION=BACKWARD(SLSTAY=0.05 CHOOSE=BIC) STATS=ALL SHOWPVALUES;
    OUTPUT OUT=interaction_out PREDICTED=predicted RESIDUAL=residual;
    TITLE "Backward Elimination Based on BIC with Interaction Terms";
RUN;


/*intermediate*/

/*VIF*/

PROC REG DATA=interaction_out PLOTS(MAXPOINTS= none) ;
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume
                  / VIF;
    TITLE "Variance Inflation Factor (VIF) for Final Model (Excluding GICS_Sector)";
RUN;

/*recheck model assumption*/

/* Residual Diagnostics */
PROC UNIVARIATE DATA=interaction_out;
    VAR residual;
    QQPLOT / NORMAL(MU=EST SIGMA=EST COLOR=RED);
    TITLE "QQ Plot for Residuals of Final Model";
RUN;

/* Test for Non-Constant Variance */
PROC GLM DATA=interaction_out PLOTS(MAXPOINTS= none );
    MODEL residual = predicted;
    OUTPUT OUT=heterosced_test_final RSTUDENT=std_residual;
    TITLE "Test for Non-Constant Variance in Final Model";
RUN;

/* Cook's Distance for Outliers */

PROC REG DATA=interaction_out PLOTS(MAXPOINTS=none ONLY)=(COOKSD RSTUDENTBYPREDICTED);
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume
                  /* Include significant interaction terms here */;
    OUTPUT OUT=cooks_dist_final R=residual P=predicted;
    TITLE "Cook's Distance for Final Model";
RUN;

/*with interactive terms*/

/* Include the best 4 interaction terms along with the main predictors */

PROC GLMSELECT DATA=train;
    CLASS GICS_Sector;
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume
                  log_Price_Range*log_Volume log_Price_Range*sqrt_Gross_Margin
                  log_Volume*log_Inventory sqrt_Gross_Margin*sqrt_Pre_Tax_Margin
                  GICS_Sector
                  / SELECTION=NONE STATS=ALL SHOWPVALUES;
    OUTPUT OUT=final_model_out PREDICTED=predicted RESIDUAL=residual;
    TITLE "Final Model with Interaction Terms";
RUN;

/* Diagnostics for the final model */
/* Variance Inflation Factor */
DATA RA.final_model_with_interactions;
    SET final_model_out; /* Start with the dataset containing the predictors */
    interaction1 = log_Price_Range * log_Volume;
    interaction2 = log_Price_Range * sqrt_Gross_Margin;
    interaction3 = log_Volume * log_Inventory;
    interaction4 = sqrt_Gross_Margin * sqrt_Pre_Tax_Margin;
RUN;

/* Final Model with Interaction Terms */
PROC REG DATA= RA.final_model_with_interactions PLOTS(MAXPOINTS= none);
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume
                  interaction1 interaction2 interaction3 interaction4 / VIF;
    TITLE "Variance Inflation Factor (VIF) for Final Model with Interaction Terms";
RUN;


/* Residual Diagnostics */
PROC UNIVARIATE DATA=RA.final_model_with_interactions;
    VAR residual;
    QQPLOT / NORMAL(MU=EST SIGMA=EST COLOR=RED);
    TITLE "QQ Plot for Residuals of Final Model";
RUN;

/* Cook's Distance */
PROC REG DATA=RA.final_model_with_interactions PLOTS(MAXPOINTS=none ONLY)=(COOKSD RSTUDENTBYPREDICTED);
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume
                  interaction1 interaction2 interaction3 interaction4;
    OUTPUT OUT=cooks_dist_final R=residual P=predicted;
    TITLE "Cook's Distance for Final Model";
RUN;


/*splitting the transformed dataset*/


/* Step 1: Identify Outliers and Influential Points */

/* Optimized REG Procedure: Skip Plots */

PROC ROBUSTREG DATA=train;
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume;
    OUTPUT OUT=robust_out RESIDUAL=residual WEIGHTS=weight;
    TITLE "Robust Regression: Downweighting Influential Points";
RUN;

proc contents data=train;
run;

/* Step 2: Flag Influential Points */
DATA diagnostic_flagged;
    SET diagnostic_out;
    /* Flag Cook's Distance > 4/n (assume n=number of observations in train) */
    IF CookD > 0.0001 THEN Flag_CookD = 1;
    ELSE Flag_CookD = 0;
    
    /* Flag Standardized Residuals beyond ±3 */
    IF ABS(StdResidual) > 4 THEN Flag_StdResidual = 1;
    ELSE Flag_StdResidual = 0;

    /* Create a final flag for influential points */
    Flag_Influential = (Flag_CookD OR Flag_StdResidual);
RUN;

PROC FREQ DATA=diagnostic_flagged;
    TABLES Flag_CookD Flag_StdResidual Flag_Influential;
    TITLE "Frequency of Flags for Outliers";
RUN;

/* Step 3: Subset Data Without Influential Points */
DATA train_no_outliers;
    SET diagnostic_flagged;
    IF Flag_Influential = 0; /* Remove flagged observations */
RUN;

PROC MEANS DATA=train_no_outliers N;
    TITLE "Check Number of Observations in train_no_outliers";
RUN;

/* Step 4: Fit the Model Without Outliers */

PROC REG DATA=train_no_outliers PLOTS(MAXPOINTS= none);
    MODEL close = log_Price_Range log_Volume log_Inventory sqrt_Gross_Margin
                  log_Gross_Profit sqrt_Pre_Tax_Margin Price_Range volume;
    OUTPUT OUT=final_model_no_outliers P=predicted_no_outliers;
    TITLE "Final Model Without Influential Points";
RUN;

/* Step 5: Compare Prediction Intervals With and Without Outliers */
PROC MEANS DATA=diagnostic_out MEAN STD MIN MAX;
    VAR predicted;
    TITLE "Prediction Results Including Outliers";
RUN;

PROC MEANS DATA=final_model_no_outliers MEAN STD MIN MAX;
    VAR predicted_no_outliers;
    TITLE "Prediction Results Excluding Outliers";
RUN;
