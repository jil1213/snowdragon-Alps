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

-   For process normal SMP files there is no Temperatur given so you can delete the variable T_LOC and their calls.

-   `data_handling/data_parameters.py`: when different grain types classification are used the `labels` and `anti_labels` and `anti_labels_long` and `colors`.
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

-   in the def `idx_to_string` the first 4 numbers of the SMP data used must be added in an elif. For example in Snowdragon-Alps:

```
 elif "S36M" in string_idx:
        return int("5" + string_idx[-4:].zfill(6))
```

-   in the def `idx_to_string` the return statement has to be changed so that the program terminates if the correct data cannot be found. For this you can delete the `return 0` and write instead:

```
    raise ValueError("SMP naming convention is unknown. Please add another elif line in idx_to_int to handle your SMP naming convention.")
```

### Run_models.py

ongoing changes here...
