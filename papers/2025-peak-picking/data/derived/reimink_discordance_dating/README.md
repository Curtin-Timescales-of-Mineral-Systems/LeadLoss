# discordance-dating
public data and code corresponding to submitted paper "Discordance Dating: A New Approach for Dating Alteration Events" by Reimink et al. 2024

Store all files in a single folder and set that folder as your working directory in the R environment. 

There are five main datasets in this repository. 
The main data containing the U-Pb and the TE analysis resides in the file "Sample_2_All_LabeledPoints.csv". 
Reference material data is in the file "RMDataAll.csv". 
The U-Pb data used for discordance dating import is "Sample_2_R.csv"
The U-Pb data corrected for common Pb is in "Sample_2_204_Corr.csv" and using the 207Pb correction method is "Sample_2_207_Corr.csv"


The main bootstrapped resampling code is run in the "ResamplingCodeWorking_public.R" file, which requires sourcing of the "UPb_Constants_Functions_Libraries.R" and "UPb_Reduction_Resample.R" files.
This R script will perform the discordance dating method, and the resampling process. Increase the iteration numbers after testing to achieve a higher N for resampling. 
This R script can be used to run the common-Pb corrected data through discordance dating as well. Just change the input file from "Sample_2_R.csv" and point it towards either "Sample_2_204_Corr.csv" etc. 

The code for peforming synthetic data reduction testing can be found in three separate files:
"SyntheticUPbDataModeling_PerfectLine_public.R" creates a synthtic single discordia line dataset
"SyntheticUPbDataModeling_UpperMultiple_public.R" creates a synthtic multiple discordia line dataset
"SyntheticUPbDataModeling_UpperRange_public.R" creats a synthetic data set with a range of upper intercepts






