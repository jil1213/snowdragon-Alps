# snowdragon-Alps CHANGELOG

This file documents all the steps of changes that need to be made to modify the Snowdragon for a different training dataset.

## Setup

First of all some general settings must be done to run the code.

-   `data_handling/data_parameters.py`: The folder structure for the SMP_LOC and EXP_LOC must be change in the own folder structure. For a variable path those calls can be used:

```
from pathlib import Path
# default paths for raw smp data and preprocessed data
#change for different folder stucture
parentdir = Path(__file__).parent.parent.as_posix()
SMP_LOC = parentdir + "/data/smp_pnt_files/"
EXP_LOC = parentdir + "/data/smp_profiles/"
```

-   To process normal SMP files there is no Temperatur given so you can delete the variable T_LOC and their calls.

-   `data_handling/data_parameters.py`: when different grain types classification are used the `labels` and `anti_labels` and `anti_labels_long` and `colors`.
-   `data_handling/data_parameters.py`: the used colours must be checked, colour 3 must be updated to `3: "#00FF00",` (# was missing before)
-   `data_handling/data_parameters.py`: in `SNOW_TYPES_SELECTION = []` the right grain types must be written
-   `requirements.txt`: the version can be deleted or updated when there are conflicts
-   `setup.py`: there could be problems with the python_requires, an update might be necessary.changed for Snowdragon-Alps

```
python_requires=">=3.6, <3.12"
```

-   Package `graphviz` must be added

## Preprocessing

### Data_preprocessing.py

In the script `data_handling/data_preprocessing.py` some changes has to be done:

-   The progressbar:The following lines must be changed so that the progress bar fits the size of the data set:
    Before declaring the `file_generator` the `file_generator_size` has to be declared

```
file_generator_size = len(list(glob.iglob(match_pnt, recursive=True)))
```

Then update the `with tqdm(total=3825) as pbar:` at two parts of the script

```
with tqdm(total=file_generator_size) as pbar:
```

-   in the def `idx_to_int` the first 4 numbers of the SMP data used must be added in an elif. For example in Snowdragon-Alps:

```
 elif "S36M" in string_idx:
        return int("5" + string_idx[-4:].zfill(6))
```

-   the same change has to be done in the def `int_to_idx`
    WARNING! This function is stored in: `models/helper_funcs.py` Add the following statement adapted to your SMP Profile names:

```
 elif smp_device == 5:
        return "S36M" + int_idx[3:7]
```

-   in the def `idx_to_int` the return statement has to be changed so that the program terminates if the correct data cannot be found. For this you can delete the `return 0` and write instead:

```
    raise ValueError("SMP naming convention is unknown. Please add another elif line in idx_to_int to handle your SMP naming convention.")
```

### Run_models.py

-   update def `preprocess_dataset`
    After step 1. Load dataframe, you should check again if there are any nans or negative values (except for the gradient negative values are ok here!) For testing you should uncomment the following section

```
    ##Check for nans and negative values -only for new dataset and if youre unsure
    # # save dataframe as csv for check for nans and - values
    # smp_org.to_csv('panda.csv', sep=',', encoding='utf-8')

    # #check for nans
    # check_nan_in_df = smp_org.isnull().values.any()
    # print (check_nan_in_df) #output is 'false' -> so there are no nans

    # #check for negative
    # count_nan_in_df = (smp_org<0).sum().sum()
    # print ('Count of negative: ' + str(count_nan_in_df))

    # # Print negative values per column
    # print (smp_org[smp_org<0].count())
    #exit(0)
```

-   If there is no wrong data you can comment this section again, if there is wrong data you should uncomment the following step `remove_nans_mosaic()` and update the def `remove_nans_mosaic()` yor your SMP dataset

```
# #remove nans
    #smp_org = remove_nans_mosaic(smp_org)
```

## Visualization

### plot_data.py

-   update the def `all_in_one_plot()` with a profile of your SMP data, change the section in Line 154 add plot in plot your profile name into the if statement

```
if profile_name == "S36M0335":
```

-   the seaborn plot has been udated, so a new syntax for lineplot is needed. Update the following statements

```
sns.lineplot(data=(raw_profile["distance"], raw_profile["force"]), ax=ax_in_plot, color="darkgrey")
```

```
sns.lineplot(data=(smp_profile["distance"], smp_profile["mean_force"]), ax=ax_in_plot)# , color="darkslategrey"
```

### plot_profile.py

-   the seaborn plot has been udated, so a new syntax for lineplot is needed. Update all `sns.lineplot()` statements (8 statements)
    update:

```
    sns.lineplot(x,y,...)
```

into:

```
sns.lineplot(data=(x,y),...)
```

### run_visualization

-   import `LABELS`

```
from data_handling.data_parameters import LABELS
```

-   change in def `def visualize_original_data(smp):` the smp_profile_name to a profile name of your dataset you want to plot
-   in def `def visualize_original_data(smp):` calling the heatmap must be updated to:

```
# SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    cleaned_labels = list(LABELS.values())
    cleaned_labels.remove(0) # remove not labelled
    cleaned_labels.remove(1) # remove surface
    cleaned_labels.remove(2) # remove ground
    corr_heatmap(smp, labels=cleaned_labels, file_name=path+"corr_heatmap_all.png")# Correlation does not help for categorical + continuous data - use ANOVA instead
```
