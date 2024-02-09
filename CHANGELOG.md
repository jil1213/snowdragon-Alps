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
-   `data_handling/data_parameters.py`: in 'SNOW_TYPES_SELECTION = []' the right grain types must be written
-   `requirements.txt`: the version can be deleted or updated when there are conflicts
-   `setup.py`: there could be problems with the python_requires, an update might be necessary.changed for Snowdragon-Alps

```
python_requires=">=3.6, <3.12"
```

-

## Preprcessing
