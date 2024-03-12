# snowdragon-Alps CHANGELOG

This file documents all the steps of changes that need to be made to modify the Snowdragon for a different training dataset.

## Setup

First of all some general settings must be done to run the code.

### data_parameters.py

in `data_handling/data_parameters.py` there have to be a few changes:

-   The folder structure for the SMP_LOC and EXP_LOC must be change in the own folder structure. For a variable path those calls can be used:

```
from pathlib import Path
# default paths for raw smp data and preprocessed data
#change for different folder stucture
parentdir = Path(__file__).parent.parent.as_posix()
SMP_LOC = parentdir + "/data/smp_pnt_files/"
EXP_LOC = parentdir + "/data/smp_profiles/"
```

-   To process normal SMP files there is no Temperatur given so you can delete the variable T_LOC and their calls.

-   when different grain types classification are used the `labels` and `anti_labels` and `anti_labels_long` and `colors`.
-   the used colours must be checked, colour 3 must be updated to `3: "#00FF00",` (# was missing before)
-   in `SNOW_TYPES_SELECTION = []` the right grain types must be written
-   the dictonary `USED_LABELS` must be added. Fill in the labels you want to use for xour classification
-   the dictonary `RARE_LABELS` must be added. Fill in the labels you want to add for a rare label

### requirements.txt

-   the version can be deleted or updated when there are conflicts
-

### setup.py

-   there could be problems with the python_requires, an update might be necessary.changed for Snowdragon-Alps

```
python_requires=">=3.6, <3.12"
```

-   Package `graphviz` must be added

## Labeling

The data is labeled in `data_handling/data_parameters.py`. If new labels are needed you have to update the script. You also have to make changes in:

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

## Models

First you have to preprocess your data for training

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

the path to the `raw_file` should be also updated. When data is stored in `data/` you can use a simmilar path to:

```
raw_file = Profile.load("./data/smp_pnt_files/" + profile_name + ".pnt")
```

-   the seaborn plot has been udated, so a new syntax for lineplot is needed. Update the following statements

```
sns.lineplot(data=(raw_profile["distance"], raw_profile["force"]), ax=ax_in_plot, color="darkgrey")
```

```
sns.lineplot(data=(smp_profile["distance"], smp_profile["mean_force"]), ax=ax_in_plot)# , color="darkslategrey"
```

-   update your labels you want to visualize in def `visualize_tree()` in line 474.

```
 anti_labels = [ANTI_LABELS[label] for label in y_train.unique()]
```

You can also comment the direct decision which label gets which decision tree label and the defintion of class names (line 467)

```
 # deciding directly which label gets which decision tree label
        #y_train[y_train==6.0] = 0
        #y_train[y_train==3.0] = 1
        #y_train[y_train==5.0] = 2
        #y_train[y_train==12.0] = 3
        #y_train[y_train==4.0] = 4
        #y_train[y_train==17.0] = 5
        #y_train[y_train==16.0] = 6
        anti_labels = [ANTI_LABELS[label] for label in y_train.unique()]
        #class_names = [anti_labels[label] for label in y_train.unique()]
```

### plot_profile.py

-   import `USED_LABELS`:
    ```
      from data_handling.data_parameters import USED_LABELS
    ```

```


-   the seaborn plot has been udated, so a new syntax for lineplot is needed. Update all `sns.lineplot()` statements (8 statements)
    update:

```

    sns.lineplot(x,y,...)

```

into:

```

sns.lineplot(data=(x,y),...)

```

-   update your labels for your legend in def `ini_bogplots()` in Line 102

```

all_labels = USED_LABELS

```

-   update your labels you want to visualize in def ´compare_bogplots()´ in line 290

```

all_labels = USED_LABELS

```

-   update your labels for your legend in def ´compare_model_and_profiles()´ in line 400

```

all_labels = USED_LABELS

```

### run_visualization

-   import `LABELS`

```

from data_handling.data_parameters import LABELS

```

-   change in def `def visualize_original_data(smp):` the smp_profile_name to a profile name of your dataset you want to plot

-   update the labels you want to plot in the corr_heatmap in line 46
    ´´´
    corr_heatmap(smp, labels=[3, 4, 5, 6, 12], file_name=path+"corr_heatmap_all.png") #removed , 16, 17
    ´´´

-   in def `def visualize_original_data(smp):` calling the heatmap must be updated to:

```

    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)

    cleaned_labels = list(LABELS.values())
    cleaned_labels.remove(0) # remove not labelled
    cleaned_labels.remove(1) # remove surface
    cleaned_labels.remove(2) # remove ground
    corr_heatmap(smp, labels=cleaned_labels, file_name=path+"corr_heatmap_all.png")# Correlation does not help for categorical + continuous data - use ANOVA instead

````

- in the def `visualize_normalized_data()` update file_name for following plots:
    Line 63
    ```
    pca(smp, n=24, biplot=False, file_name=path + "pca")
    ```
    Line 64
    ```
    tsne(smp, file_name=path + "tsne")
    ```
    Line 66
    ```
    tsne_pca(smp, n=5, file_name=path + "tsne_pca")
    ```

### plot_dim_reduction

-   the gca() statement was updated. Change the following statements.
    Line 81, 145, 210

````

    ax = plt.figure(figsize=(16,10)).gca(projection="3d")

```

change to

```

        fig = plt.figure(figsize=(16,10))
        ax = fig.add_subplot(projection='3d')

```

```
