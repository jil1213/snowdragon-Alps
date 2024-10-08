# Here are all the important parameters for the data preprocssing

from pathlib import Path
# default paths for raw smp data and preprocessed data
#change for different folder stucture
parentdir = Path(__file__).parent.parent.as_posix()
SMP_LOC = parentdir + "/data/smp_pnt_files/"
EXP_LOC = parentdir + "/data/smp_profiles/"
EVAL_LOC = parentdir + "/output/evaluation/"

# Example profiles - one for plotting and the folder where they live
EXAMPLE_SMP_NAME = "S36M0337"
EXAMPLE_SMP_PATH = "data/smp_pnt_files/"

# filenames where data, normalized data and preprocessed data is stored
SMP_ORIGINAL_NPZ = "data/all_smp_profiles.npz"
SMP_NORMALIZED_NPZ = "data/all_smp_profiles_updated_normalized.npz" #dont need this one 
SMP_PREPROCESSED_TXT = "data/preprocess_data.txt"


# set used labels 
USED_LABELS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
#simply set to zero if you have no rare labels! 
RARE_LABELS = [] #14.0 when using data type rare

LABELS = {
    "not_labelled": 0,
    "surface": 1,
    "ground": 2,
    "pp": 3,
    "ppgp": 4,
    "df": 5,
    "rg": 6,
    "fc": 7,
    "dh": 8,
    "sh": 9,
    "fcxr": 10,
    "mf": 11,
    "mfcr": 12,
    "if": 13,
    "rare":14,
}

ANTI_LABELS = {
    0: "not_labelled",
    1: "surface",
    2: "ground",
    3: "pp",
    4: "ppgp",
    5: "df",
    6: "rg",
    7: "fc",
    8: "dh",
    9: "sh",
    10: "fcxr",
    11: "mf",
    12: "mfcr",
    13: "if",
    14: "rare"
}

ANTI_LABELS_LONG = {
    0: "Not labelled",
    1: "Surface",
    2: "Ground",
    3: "Precipitation Particles",
    4: "Graupel",
    5: "Decomposing and fragmented\nprecipitation particles",
    6: "Rounded Grains",
    7: "Faceted Crystals",
    8: "Depth Hoar",
    9: "Surface Hoar",
    10: "Rounding faceted particles",
    11: "Melt Forms",
    12: "Melt-Freeze Crust",
    13: "Ice Formations",
    14: "rare"
}

COLORS = {
    0: "dimgray",
    1: "chocolate",
    2: "darkslategrey",
    3: "#00FF00",
    4: "#808080",
    5: "#228B22",
    6: "#FFB6C1",
    7: "#ADD8E6",
    8: "#0000FF",
    9: "#FF00FF",
    10: "#D6C7D4", # mixture of fc and rg
    11: "#FF0000",
    12: "#890000", # mixture of mf (red) and stripes (C40000) should be changed in red with vertical stripes
    13: "#00FFFF",
    14: "#000000",
}

# arguments for Preprocessing
PARAMS = {
    "sum_mm": 1,
    "gradient": True,
    "window_size": [4, 12],
    "window_type": "gaussian",
    "window_type_std": 1,
    "rolling_cols": ["mean_force", "var_force", "min_force", "max_force"],
    "poisson_cols": ["median_force", "lambda", "delta", "L", "Density", "SSA", "Hardness"],
}

# Colors for the different models
# TODO: Pick good colors
MODEL_COLORS02 = {
    "Majority Vote": "grey",
    "K-means": "fuchsia",
    "Gaussian Mixture Model": "orchid",
    "Bayesian Gaussian Mixture Model": "palevioletred",
    "Self Trainer": "plum",
    "Label Propagation": "blueviolet",
    "Random Forest": "indigo",
    "Balanced Random Forest": "darkblue",
    "Support Vector Machine": "blue",
    "K-nearest Neighbors": "royalblue",
    "Easy Ensemble": "dodgerblue",
    "LSTM": "green",
    "BLSTM": "springgreen",
    "Encoder Decoder": "lime",
}

MODEL_LABELS = {
    "baseline": "Majority Vote",
    "bmm": "Bayesian Gaussian Mixture Model",
    "gmm": "Gaussian Mixture Model",
    "kmeans": "K-means",
    "self_trainer": "Self Trainer",
    "label_spreading": "Label Propagation",
    "easy_ensemble": "Easy Ensemble",
    "knn": "K-nearest Neighbors",
    "rf": "Random Forest",
    "rf_bal": "Balanced Random Forest",
    "svm": "Support Vector Machine",
    "lstm": "LSTM",
    "blstm": "BLSTM",
    "enc_dec": "Encoder Decoder"
}

# experimenting with the colors
# group 01: GRAY - baseline
# group 02: ORANGE/RED semi-supervised
#   kmeans, gmm, bgm, self-trainer, label propagation
# group 03: BLUE supervised
#   random forest, rf balanced, svm, knn, easy ensemble
# group 04: GREEN ANNs
#   lstm, blstm, enc dec
MODEL_COLORS = {
    "Majority Vote": "grey",
    "K-means": "xkcd:dandelion",
    "Gaussian Mixture Model": "xkcd:golden rod",
    "Bayesian Gaussian Mixture Model": "xkcd:orange",
    "Self Trainer": "xkcd:red",
    "Label Propagation": "xkcd:crimson",
    "Random Forest": "indigo",
    "Balanced Random Forest": "navy",
    "Support Vector Machine": "blue",
    "K-nearest Neighbors": "xkcd:sky blue",
    "Easy Ensemble": "xkcd:light teal",
    "LSTM": "xkcd:green apple", #"xkcd:slime green" #apple
    "BLSTM": "xkcd:emerald",
    "Encoder Decoder": "xkcd:forest green",
}

# Selection of snow layer types that we use
# TODO: Turn into dictionary with colors?
# Sorted by amount of examples?
SNOW_TYPES_SELECTION = ["pp","ppgp","df","rg","fc","dh","sh","fcxr","mf","mfcr","if"]
