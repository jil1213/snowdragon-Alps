# snowdragon-Alps CHANGELOG

This file documents all the steps of changes that need to be made to modify the Snowdragon for a different training dataset.

## Setup

First of all some general settings must be done to run the code.

### data_parameters.py

in `data_handling/data_parameters.py` there have to be a few changes:

-   The folder structure for the SMP_LOC , EXP_LOC and EVAL_LOC (path of output of evaluation) must be change to the own folder structure. For a variable path those calls can be used:

```
from pathlib import Path
# default paths for raw smp data and preprocessed data
#change for different folder stucture
parentdir = Path(__file__).parent.parent.as_posix()
SMP_LOC = parentdir + "/data/smp_pnt_files/"
EXP_LOC = parentdir + "/data/smp_profiles/"
EVAL_LOC = parentdir + "/output/evaluation/"
```

-   Also the folder structure for the stored data (normalized and preprocessed) should be added here. For example:

```
# Example profiles - one for plotting and the folder where they live
EXAMPLE_SMP_NAME = "S36M0337"
EXAMPLE_SMP_PATH = "data/smp_pnt_files/"

# filenames where data, normalized data and preprocessed data is stored
SMP_ORIGINAL_NPZ = "data/all_smp_profiles.npz"
SMP_NORMALIZED_NPZ = "data/all_smp_profiles_updated_normalized.npz" #dont need this one
SMP_PREPROCESSED_TXT = "data/preprocess_data.txt"
```

-   To process normal SMP files there is no Temperatur given so you can delete the variable T_LOC and their calls.

-   when different grain types classification are used the `labels` and `anti_labels` and `anti_labels_long` and `colors`.
-   the used colours must be checked, colour 3 must be updated to `3: "#00FF00",` (# was missing before)
-   in `SNOW_TYPES_SELECTION = []` the right grain types must be written
-   the dictonary `USED_LABELS` must be added. Fill in the labels you want to use for your classification
-   the dictonary `RARE_LABELS` must be added. Fill in the labels you want to add for a rare label

### requirements.txt

-   the version can be deleted or updated when there are conflicts

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

-   import Paths were data is stored: `from data_handling.data_parameters import SMP_ORIGINAL_NPZ, SMP_NORMALIZED_NPZ, SMP_PREPROCESSED_TXT, EXAMPLE_SMP_NAME`
    Add the right file arguments in the praser:
    ```
            # File arguments
        parser.add_argument("--smp_npz", default=SMP_ORIGINAL_NPZ, type=str, help="Name of the united npz file")
        parser.add_argument("--preprocess_file", default=SMP_PREPROCESSED_TXT, type=str, help="Name of the txt file where the preprocessed data is stored.")
    ```
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

-   in def `preprocess_dataset()` a flag `preprocess_dataset(ignore_unlabelled=False)`
    add in section 6. an if-statement:
    the old code is in the else part
    ```
    if ignore_unlabelled:
            unlabelled_x = None
            unlabelled_y = None
            unlabelled_smp_x = None
        else:
            # set unlabelled_smp label to -1
            unlabelled_smp.loc[:, "label"] = -1
            unlabelled_smp_x = unlabelled_smp.drop(["label", "smp_idx"], axis=1)
            unlabelled_smp_y = unlabelled_smp["label"]
            # sample in order to make it time-wise possible
            # OBSERVATION: the more data we include the worse the scores for the models become
            unlabelled_x = unlabelled_smp_x.sample(sample_size_unlabelled) # complete data: 650 326
            unlabelled_y = unlabelled_smp_y.sample(sample_size_unlabelled) # we can do this, because labels are only -1 anyway
    ```
    When calling the function in the main() also add the flag:
    In Line 852
    ```
    data = preprocess_dataset(
            smp_file_name=args.smp_npz, output_file=args.preprocess_file, visualize=False, ignore_unlabelled=True)
    ```
-   in the def train_and_store_models() in Line 541
    ```
     fitted_model.save("models/stored_models/" + model_type + ".keras")
    ```
-   in def `evaluate_all models()` change so that one single model can be evaluated:
    Add the input parameter models and model_names

    ```
    def evaluate_all_models(data, models=["all"], model_names=None, file_scores=None, file_scores_lables=None, overwrite_tables=True, **params):
    ```

    Add the following if statement:

    ```
    if models == ["all"]:
        all_models = ["baseline", "kmeans", "gmm", "bmm",
                      "rf", "rf_bal", "svm", "knn", "easy_ensemble",
                      "self_trainer", "label_spreading",
                      "lstm", "blstm", "enc_dec"]
        all_names = ["Majority Vote", "K-means", "Gaussian Mixture Model", "Bayesian Gaussian Mixture Model",
                     "Random Forest", "Balanced Random Forest", "Support Vector Machine", "K-nearest Neighbors", "Easy Ensemble",
                     "Self Trainer", "Label Propagation",
                     "LSTM", "BLSTM", "Encoder Decoder"]
    else:
        all_models = models

    if model_names is None:
        all_names = all_models
    ```

    And delete line 588-598 where all_models and all names were defined

    -   Add the input parameter profile_name by calling `all_in_one_plot()` Line 639

        ```
        all_in_one_plot(all_smp_trues, show_indices=False, sort=True,
                            title="All Observed SMP Profiles of the Testing Data", file_name=save_file, profile_name=EXAMPLE_SMP_NAME)
        ```

-   in def `validate_all models()` change so that one single model can be validated:
    Add the input parameter models and model_names

    ```
    def validate_all_models(data, intermediate_file=None, models=["all"], model_names=None):
    ```

    -   Add in def `main()` the input arguement model for the evaluation handler and the validation handler

        ```
            # EVALUATION
            if args.evaluate: evaluate_all_models(data, models=args.models, overwrite_tables=False)
        ```

        ```
            # VALIDATION
            if args.validate:
            intermediate_results = "data/validation_results.txt"
            validate_all_models(data, intermediate_results, models=args.models)
        ```

        Add the following if statement:

        ```
        if models == ["all"]:
            all_models = ["baseline", "kmeans", "gmm", "bmm",
                        "rf", "rf_bal", "svm", "knn", "easy_ensemble",
                        "self_trainer", "label_spreading",
                        "lstm", "blstm", "enc_dec"]
            all_names = ["Majority Vote", "K-means", "Gaussian Mixture Model", "Bayesian Gaussian Mixture Model",
                        "Random Forest", "Balanced Random Forest", "Support Vector Machine", "K-nearest Neighbors", "Easy Ensemble",
                        "Self Trainer", "Label Propagation",
                        "LSTM", "BLSTM", "Encoder Decoder"]
        else:
            all_models = models

        if model_names is None:
            all_names = all_models
        ```

        and put the single validation of models into the following statement:

        ```
         for model_type, name in zip(all_models, all_names):
            print("Starting {} Model ...\n".format(name))
            # Baseline - majority class predicition
            if model_type == "baseline":
                baseline_scores = majority_class_baseline(x_train, y_train, cv_stratified)
                all_scores.append(mean_kfolds(baseline_scores))

            # kmeans clustering (does not work well)
            # BEST cluster selection criterion: no difference, you can use either acc or sil (use sil in this case!)
            elif model_type == "kmeans":
                kmeans_scores = kmeans(unlabelled_smp_x, x_train, y_train, cv_stratified, num_clusters=10, find_num_clusters="acc", plot=False)
                all_scores.append(mean_kfolds(kmeans_scores))

            # mixture model clustering ("diag" works best at the moment)
                # BEST cluster selection criterion: bic is slightly better than acc (generalization)
            elif model_type == "gmm":
                print("Starting Gaussian Mixture Model...")
                gm_acc_diag = gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag", find_num_components="acc", plot=False)
                all_scores.append(mean_kfolds(gm_acc_diag))

            elif model_type == "bmm":
                bgm_acc_diag = bayesian_gaussian_mix(unlabelled_smp_x, x_train, y_train, cv_stratified, cov_type="diag")
                all_scores.append(mean_kfolds(bgm_acc_diag))

            # TAKES A LOT OF TIME FOR COMPLETE DATA SET
            # label_spreading and self training model -> different data preparation necessary

            elif model_type == "label_spreading":
                ls_scores = label_spreading(x_train=x_train_all, y_train=y_train_all, cv_semisupervised=cv_semisupervised, name="LabelSpreading_1000")
                all_scores.append(mean_kfolds(ls_scores))

            elif model_type == "self_trainer":
                knn = KNeighborsClassifier(n_neighbors = 20, weights = "distance") # TODO replace with balanced random forest
                st_scores = self_training(x_train=x_train_all, y_train=y_train_all, cv_semisupervised=cv_semisupervised, base_model=knn, name="SelfTraining_1000")
                all_scores.append(mean_kfolds(st_scores))

            elif model_type == "rf":
                rf_scores = random_forest(x_train, y_train, cv_stratified, visualize=False)
                all_scores.append(mean_kfolds(rf_scores))

            elif model_type == "rf_bal":
                rf_scores = random_forest(x_train, y_train, cv_stratified, visualize=False, resample=True, name=name)
                all_scores.append(mean_kfolds(rf_scores))

            # Support Vector Machines
            # works with very high gamma (overfitting) -> "auto" yields 0.75, still good and no overfitting
            elif model_type == "svm":
                svm_scores = svm(x_train, y_train, cv_stratified, gamma="auto")
                all_scores.append(mean_kfolds(svm_scores))

            elif model_type == "knn":
                knn_scores = knn(x_train, y_train, cv_stratified, n_neighbors=20)
                all_scores.append(mean_kfolds(knn_scores))

            elif model_type == "easy_ensemble":
                ada_scores = ada_boost(x_train, y_train, cv_stratified)
                all_scores.append(mean_kfolds(ada_scores))

            elif model_type == "lstm":
                lstm_scores = ann(x_train, y_train, smp_idx_train, ann_type="lstm", cv_timeseries=cv_timeseries, name="LSTM",
                                batch_size=32, epochs=10, rnn_size=25, dense_units=25, dropout=0.2, learning_rate=0.01)
                print(lstm_scores)
                all_scores.append(mean_kfolds(lstm_scores))

            elif model_type == "blstm":
                #  cv can be a float, or a cv split
                blstm_scores = ann(x_train, y_train, smp_idx_train, ann_type="blstm", cv_timeseries=cv_timeseries, name="BLSTM",
                                batch_size=32, epochs=10, rnn_size=25, dense_units=25, dropout=0.2, learning_rate=0.01)
                print(blstm_scores)
                all_scores.append(mean_kfolds(blstm_scores))

            elif model_type == "enc_dec":
                #  cv can be a float, or a cv split
                encdec_scores = ann(x_train, y_train, smp_idx_train, ann_type="enc_dec", cv_timeseries=cv_timeseries, name="ENC_DEC",
                                batch_size=32, epochs=10, rnn_size=25, dense_units=0, dropout=0.2, learning_rate=0.001,
                                attention=True, bidirectional=False)
                print(encdec_scores)
                all_scores.append(mean_kfolds(encdec_scores))

            if intermediate_file is not None: save_results(intermediate_file, all_scores)
            print("...finished {} Model.\n".format(name))
        ```

### cv_handler.py

change line 262 in def `cv_manual()`

```
k_idx = np.resize(np.arange(1, k+1), len(profiles))
```

### evaluation.py

import EXAMPLE-SMP-NAME

```
from data_handling.data_parameters import ANTI_LABELS, EXAMPLE_SMP_NAME
```

in def `plot_testing()` by calling `all_in_one_plot()` add name to the input parameters:
Line336 and Line 344

```
 all_in_one_plot(all_smp_trues, show_indices=False, sort=True,
                        title="All Observed SMP Profiles of the Testing Data", file_name=save_file, profile_name=EXAMPLE_SMP_NAME)
```

When using the `testing()` function to evaluate a new (independent) data set, it may happen that not all labels are predicted. This leads to an error, as not all labels are used. To add probabilities of 0 for missing grain types (labels), the following change in the code is necessary:
In def `testing()` change line 404 to 406:

```
y_pred, prelim_y_pred_prob, fit_time, score_time = predicting(model, x_train,
            y_train, x_test, y_test, smp_idx_train, smp_idx_test,
            unlabelled_data, impl_type, **plot_and_fit_params)
```

Add after calling prediticng the following if statement:

```
# searching for missing grain types in prediction
    missing_grain_types_in_pred = np.array(list(set(np.unique(y_test)) - set(np.unique(y_pred))))

    # when missing grain types in prediction, add probabilities 0 for these
    if len(missing_grain_types_in_pred) > 0:
        print("Missing Grain types in pred:", missing_grain_types_in_pred)
        y_pred_prob = prelim_y_pred_prob
        for missing_grain_type in np.sort(missing_grain_types_in_pred):
            position_grain_type = np.argwhere(labels_order==missing_grain_type)[0][0]
            new_col = np.zeros((prelim_y_pred_prob.shape[0], 1))
            # add new column with zeros for missing grain type
            y_pred_prob = np.hstack((y_pred_prob[:,:position_grain_type], new_col, y_pred_prob[:, position_grain_type:]))
    else:
        y_pred_prob = prelim_y_pred_prob
```

### supervised_models.py

For adaboost is a updated syntax for the spelling in sampeling_strategy. Update in Line 111

```
    def ada_boost(x_train, y_train, cv, n_estimators=100, sampling_strategy="not_majority", name="AdaBoost", only_model=False, **kwargs):
```

to `not majority` (change underscore in blank)

```
    def ada_boost(x_train, y_train, cv, n_estimators=100, sampling_strategy="", name="AdaBoost", only_model=False, **kwargs):
```

### anns.py

-   Update the `def calculate_metrics_ann()` by adding the input parameter cv: in Line 396

    ```
    def calculate_metrics_ann(results, cv, name=None):
    ```

    set the input parameter `cv=cv` in the lines 406 and 407, 409 and 410

    ```
        all_scores.update(calculate_metrics_raw(y_trues=results["y_true_train"], y_preds=results["y_pred_train"], metrics=METRICS, cv=cv, annot="train"))
        all_scores.update(calculate_metrics_raw(y_trues=results["y_true_valid"], y_preds=results["y_pred_valid"], metrics=METRICS, cv=cv, annot="test"))
    ```

-   upadte the `def ann()`
    add the input parameter `cv` when calling the `return calculate_metrics_ann()`
    In Line 618 set `cv=False`
    ```
        return calculate_metrics_ann(results, cv=False , name=name)
    ```
    In Line655 (the else statement) set `cv=True`
    ```
        return calculate_metrics_ann(results, cv=True , name=name)
    ```
    Also update the dictonary `all_results={}` by deleting Line 622 and 633 key_assigned=False, and adding
    ```
    all_results = { # the rest is added on its own
            "score_time": [],
            "fit_time": [],
            "y_true_train": [],
            "y_true_valid": [],
            "y_pred_train": [],
            "y_pred_valid": [],
            "y_pred_prob_train": [],
            "y_pred_prob_valid": [],
         }
    ```
    delete the if not keys_assigned statement and the appendix to the dictonary (Line 641-654) and add instead:
    ```
    all_results[key].append(k_results[key])
    ```

## Visualization

### plot_data.py

-   Import `EXAMPLE-SMP_PATH`

-   update the def `all_in_one_plot()` with a profile of your SMP data, change the section in Line 154 add plot in plot your profile name into the if statement

```
if profile_name:
```

the path to the `raw_file` should be also updated. When data is stored in `data/` you can use a simmilar path to `EXAMPLE_SMP_PATH`:

```
raw_file = Profile.load(EXAMPLE_SMP_PATH + profile_name + ".pnt")
```

change sns.lineplot syntax in Line 164

```
sns.lineplot(data=raw_profile, x="distance", y="force", ax=ax_in_plot, color="darkgrey")
```

-   do the same if statement for adding a smaller plot in Line 159 and change also the seadorn sns.lineplot syntax

    ```
     # add plot within plot
    if profile_name:
        ax = plt.gca()
        fig = plt.gcf()
        ax_in_plot = ax.inset_axes([0.15,0.5,0.4,0.4])
         # retrieve original smp signal
         # load npz in smp_profiles_updated
        raw_file = Profile.load(EXAMPLE_SMP_PATH + profile_name + ".pnt")
        raw_profile = raw_file.samples_within_snowpack(relativize=True)
        sns.lineplot(data=raw_profile, x="distance", y="force", ax=ax_in_plot, color="darkgrey")
        ax_in_plot.set_xlim(left=0, right=raw_profile["distance"].max())
    ```

-   the seaborn plot has been udated, so a new syntax for lineplot is needed. Update the following statements

```
sns.lineplot(data=(raw_profile["distance"], raw_profile["force"]), ax=ax_in_plot, color="darkgrey")
```

change into

```
sns.lineplot(data=(smp_profile["distance"], smp_profile["mean_force"]), ax=ax_in_plot)# , color="darkslategrey"
```

-   also update the xlim statements for the plots appearance
    e.g in Line 178

    ```
    ax_in_plot.set_xlim(left=0, right=smp_profile["distance"].max())
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

-   import `USED_LABELS` and `RARE_LABELS`:
    ```
      from data_handling.data_parameters import USED_LABELS, RARE_LABELS
    ```
-   the seaborn plot has been udated, so a new syntax for lineplot is needed. Update all `sns.lineplot()` statements (8 statements)
    update:

        ```
        sns.lineplot(x,y,...)
        ```

    into:
    `sns.lineplot(data=(x,y) )`

-   update the following seaborn lineplot statements:
    Line351 in `def compare_model_and_profiles()`

    ```
     signalplot = sns.lineplot(data= smp_true, x="distance", y="mean_force", ax=axs[model_i, smp_i])
     signalplot.set_xlim(left=0, right=smp_true["distance"].max())
    ```

    Line468 in `def compare_plot()`

    ```
     signalplot = sns.lineplot(data=smp, x="distance", y="mean_force", ax=ax)
    ```

    For XLIM: also Line 504 in `def compare_plot()`

    ```
     ax.set_xlim(left=0, right=smp["distance"].max())
    ```

    Line 653 in `def smp_pair()`

    ```
     axs[1] = sns.lineplot(data= smp_pred, x= "distance", y= "mean_force")
    ```

    For XLIM: also Line 665 in `def smp_pair()`

    ```
     axs[1].set_xlim(left=0, right=smp_pred["distance"].max())
    ```

    Line722 in `def smp_pair_both()`

    ```
    ax = sns.lineplot(data=smp_profile, x="distance", y="mean_force", ax=ax)
    ```

    For XLIM: also Line 737 in `def smp_pair_both()`

    ```
     ax.set_xlim(left=0, right=smp["distance"].max())
    ```

-   update in def `smp_unlabelled()` in line 554 an in def `smp_labelled()` in line 585 the lineplot statement:

    ```
    ax = sns.lineplot(data=(smp_profile["distance"], smp_profile["mean_force"]))
    ```

    into

    ```
     ax = sns.lineplot(data=smp_profile, x="distance",y ="mean_force")
     ax.set_xlim(left=0, right=smp_profile["distance"].max())
    ```

-   update your labels for your legend in def `ini_bogplots()` in Line 102

    ```
    all_labels = USED_LABELS
    ```

-   update your labels you want to visualize in def ´compare_bogplots()´ in line 290

    ```
    all_labels = USED_LABELS + RARE_LABELS
    ```

-   update your labels for your legend in def ´compare_model_and_profiles()´ in line 400

    ```
    all_labels = USED_LABELS + RARE_LABELS
    ```

### run_visualization

-   import `LABELS, EXAMPLE_SMP_NAME, SMP_ORIGINAL_NPZ, SMP_NORMALIZED_NPZ, SMP_PREPROCESSED_TXT, EVAL_LOC`

    ```
    from data_handling.data_parameters import LABELS, EXAMPLE_SMP_NAME, SMP_ORIGINAL_NPZ, SMP_NORMALIZED_NPZ, SMP_PREPROCESSED_TXT, EVAL_LOC
    ```

-   change in def `def visualize_original_data(smp):` the smp_profile_name to a EXAMPLE_SMP_NAME

-   update the labels you want to plot in the corr_heatmap in line 46

    ```
    corr_heatmap(smp, labels=[3, 4, 5, 6, 12], file_name=path+"corr_heatmap_all.png") #removed , 16, 17
    ```

-   in def `def visualize_original_data(smp):` calling the heatmap must be updated to:

    ```
        # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)

        cleaned_labels = list(LABELS.values())
        cleaned_labels.remove(0) # remove not labelled
        cleaned_labels.remove(1) # remove surface
        cleaned_labels.remove(2) # remove ground
        corr_heatmap(smp, labels=cleaned_labels, file_name=path+"corr_heatmap_all.png")# Correlation does not help for categorical + continuous data - use ANOVA instead
    ```

-   in the def `visualize_normalized_data()` update file_name for following plots:
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

    also update the smp_profile_name in line 76 to EXAMPLE_SMP_NAME

-   in the def `visualize_results()`
    update path of `y_pred_probs` (line134) to EVAL_LOC

    ```
     names, cf_matrices, label_orders, y_trues, y_pred_probs = prepare_evaluation_data(EVAL_LOC)
    ```

    update SMP indices .txt file in line 182

    ```
     with open(SMP_PREPROCESSED_TXT, "rb") as myFile:
    ```

-   in the def `main()` update loaded data in line 200, 201

    ```
     # load dataframe with smp data
     smp = load_data(SMP_ORIGINAL_NPZ)
     smp_preprocessed = load_data(SMP_NORMALIZED_NPZ)
    ```

### plot_dim_reduction

-   the `gca()` statement was updated. Change the following statements.
    Line 81, 145, 210

    ```
        ax = plt.figure(figsize=(16,10)).gca(projection="3d")
    ```

change to

    ```
        fig = plt.figure(figsize=(16,10))
        ax = fig.add_subplot(projection='3d')
    ```

## predict.py

For predictions of raw smp files

Import statements to add:

```
import pandas as pd
from visualization.plot_profile import smp_labelled
from data_handling.data_parameters import ANTI_LABELS, PARAMS, EXP_LOC
```

The paths have to be adapted in Line30. (To add later in configs)

```
# make predictions for all files in this folder

parentdir = Path(**file**).parent.as_posix()
IN_DIR = parentdir + "/data/raw_smp_prediction/"
MARKER_PATH = "data/markers_pred.csv"
```

Update the variable marker_path for use:

```
def predict_all(unlabelled_dir=IN_DIR, marker_path=MARKER_PATH, mm_window=1, overwrite=True):
```

Update the name of the preprocessed_data file: (Line 130):

```
with open("data/preprocess_data.txt", "rb") as handle:
```

When you want to be able to load the markers as an ini (instead of csv), add the function `load_markers_ini`:

```
def load_markers_ini(marker_path_ini, smp_idx_str):
    """ Loads and returns sfc and ground markers with profile name as key
    Parameters:
        marker_path_ini (Path): where the markers are stored in .ini file
        smp_idx_str (string): profile name
    Returns:
        dict < smp_name: (surface, ground) >: marker dictionary
    """
    smp_idx_str = "S45M" +smp_idx_str[3:]
    file_path = os.path.join(marker_path_ini, f"{smp_idx_str}.ini")

    with open(file_path, 'r') as file:
        marker_dic = {}
        for line in file:
            if line.startswith('surface') or line.startswith('ground'):
                key, value = line.split('=')
                marker_dic[key.strip()] = float(value.strip())

    return marker_dic
```

In def `predict_all()` make the following changes:
For adapting the folder structure of prediction (before: output/prediftion), now: output/prediction/ini or output/prediction/plot, change sub_location to sub_location_ini(in Line 127following):

```
sub_location_ini = location + "/" + model_name + "/ini/"
```

Also add new section afterwards where sublocation for plot is created:

```
 sub_location_plot = location + "/" + model_name + "/plot/"
        # make dir if it doesnt exist yet
        if not os.path.exists(sub_location_plot):
            os.makedirs(sub_location_plot)
```

after the prediction, save as dataframe for plot visualization:

```
labelled_smp_df = pd.DataFrame(labelled_smp, columns=["label"])
#delete old labels of unlabelled_smp
unlabelled_smp.drop(columns=["label"], inplace=True)
#merge unlabelled_smp with labelled_smp to combine all informations for visualization
smp = pd.concat([unlabelled_smp, labelled_smp_df], axis=1)
```

When you want to load markers an an ini, add in Line 170 (before prediction starts):

```
# load markers
markers = load_markers_ini(unlabelled_dir, smp_idx_str)
```

Make plot in Line 198 (end of if statement)

```
smp_idx_float = float(smp_idx_str)
smp_labelled(smp, smp_idx_float, file_name=sub_location_plot+smp_idx_str)
```

For markers with ini, change the try: get markers section into(Line 187):

```
try: # get markers
sfc, ground = markers["surface"], markers["ground"]
# save ini
save_as_ini(labelled_smp, sfc, ground, save_file, model_name, git_id)
```

In def `save_as_ini` changes from June7, check if necessary

In def `load_profiles` update the export_dir, for example:

```
export_dir = Path("data/smp_npz_profiles_pred/")
```

also change filter=False if this occurs an error

When you use different data than the mosaic dataset, comment the following section: (Line 282) normalization

```
#all_smp = normalize_mosaic(all_smp) #not for other data than mosaic
```

In def `load_stored_model` update the model file names for keras models: (Line 310)

```
if model_type == "keras":
        if model_name != "enc_dec":
            model_filename = "models/stored_models/" + model_name + ".hdf5"
            loaded_model = keras.models.load_model(model_filename)
        else:
            model_filename = "models/stored_models/" + model_name + ".keras"
            loaded_model = keras.models.load_model(model_filename, custom_objects={"SeqSelfAttention": SeqSelfAttention})
```

## Integration to snowmicropyn

To get the code base ready for an interaction with snowmicropyn some changes have to be done to adjust the calc funtions of both.

The following changes can be found in the Commit 24July (https://github.com/jil1213/snowdragon-Alps/commit/d818415f646d338c08f0c2561c0d1f3482df6385)

The adaption for the new calc function can be found in Commit 26August (https://github.com/jil1213/snowdragon-Alps/commit/a2b4a6a900d06fbcc4e9b6c031e61e9e382f466b)
For this the new def calc_derivatives The function is stored in the branch develop of snowmicropyn(https://github.com/slf-dot-ch/snowmicropyn/blob/develop/snowmicropyn/profile.py#L585)

### data_handling/data_parameters.py

To the arguments for Preprocessing (PARAMS) the has to be adapted, Line 106:

```
    "poisson_cols": ["median_force", "lambda", "delta", "L", "Density", "SSA", "Hand_hardness", "optical_thickness"],
```

### data_handling/data_preprocessing.py

The coulmns Density, SSA and Harndess have to be added to the `poisson_cols` (list) in data_parameters.py.
For this the comments in Line 256, def `rolling_window()` and in Line 386 def `preprocess_profile()` has to be addapted:

```
    List can include: "distance", "median_force", "lambda", "f0", "delta", "L", "Density", "SSA", "Hand_hardness", "optical_thickness
```

also the `Key Error` in Line 288-289:

```
    except KeyError:
                    print("You can only use a (sub)list of the following features for poisson_cols: distance, median_force, lambda, f0, delta, L, Density, SSA, Hand_hardness, optical_thickness")
```

In def `rolling_window()` the List of `poisson_all_colls` in Line 265 has to be adapted:

```
    poisson_all_cols = ["distance", "median_force", "lambda", "f0", "delta", "L", "Density", "SSA", "Hand_hardness", "optical_thickness]
```

In def `calc()` (temporary) the columns must be added as zeros to get the complete dataframe. For this add at the end of the function in Line 320:

```
    result = [row + (0, 0, 0) for row in result]
```

Also add the columns to the return statement:

```
    return pd.DataFrame(result, columns=['distance', 'force_median', 'L2012_lambda', 'L2012_f0',
                                                    'L2012_delta', 'L2012_L', 'Density', 'SSA', 'Hardness'])
```

To replace the old calc function, the new def `calc()` from snowmicropyn is used. This exists only in the branch develop of snowmicropyn.
For this make the following changes:
Add the given profile to the function parameters of `rolling_window()` in Line 246

```
    def rolling_window(df, profile, window_size, rolling_cols, window_type="gaussian", window_type_std=1, poisson_cols=None, \*\*kwargs):
```

Also adapt the function call in Line 437 in def `preprocess_profile():`

```
    # 6. rolling window in order to know distribution of next and past values (+ poisson shot model)
    final_df = rolling_window(df_mm, profile, **params)
```

In def `rolling_window()` replace the old Function call of `calc()` in def `calc_derivatives()` in Line 286:

```
poisson_rolled = profile.calc_derivatives(names_with_units=False, hand_hardness=True, optical_thickness=True)
```
