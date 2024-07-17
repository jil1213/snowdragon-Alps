# get the necessary imports from folders above
import os
import git
import joblib
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tensorflow import keras
from itertools import groupby
from keras_self_attention import SeqSelfAttention

from models.baseline import predict_baseline
from models.cv_handler import assign_clusters_single_profile
from models.anns import predict_single_profile_ann
from models.run_models import remove_nans_mosaic, normalize_mosaic
from models.helper_funcs import int_to_idx
from visualization.plot_profile import smp_labelled
from tuning.tuning_parameters import BEST_PARAMS
from data_handling.data_parameters import ANTI_LABELS, PARAMS, EXP_LOC
from data_handling.data_preprocessing import export_pnt, npz_to_pd, search_markers

import xml.etree.ElementTree as ET

# make an argparser for knowing which model should be used
# and which files should be processed
# and where the results should be saved (and if visualizations should be stored as well)

# make predictions for all files in this folder
parentdir = Path(__file__).parent.as_posix()
IN_DIR = parentdir + "/data/raw_smp_prediction/"
MARKER_PATH = "data/markers_pred.csv"

_ns_caaml = 'caaml' # XML namespaces
_ns_gml = 'gml'
_ns = {_ns_caaml: 'http://caaml.org/Schemas/SnowProfileIACS/v6.0.3', _ns_gml: 'http://www.opengis.net/gml'}

#creating the settings for the export
settings = {
    'location_name': 'SMP observation point',
    'altitude': 0,
    'slope_exposition': 0,
    'slope_angle': 0
}

# save both the pics and the results as .ini files
def predict_profile(smp, model, data, model_type):
    """ Predict classification of single profile
    """
    train_data = data["x_train"]
    train_targets = data["y_train"]

    # prepare smp data for which want to create prediction
    x_unknown_profile = smp.drop(["label", "smp_idx"], axis=1)

    # predict on that model
    if model_type == "scikit": # XXX WORKS
        y_pred = model.predict(x_unknown_profile)
        # TODO does not work for svm --> maybe we stored the wrong model? (always the same prediction)

    elif model_type == "keras": # XXX WORKS
        # TODO won't work right now
        y_pred = predict_single_profile_ann(model, x_unknown_profile, train_targets)

    elif model_type == "baseline": # XXX WORKS
        majority_vote = model
        y_pred = predict_baseline(majority_vote, x_unknown_profile)

    elif model_type == "semi_manual": # XXX WORKS
        # determine the number of clusters/components
        if hasattr(model, "cluster_centers_"):
            num_components = model.cluster_centers_.shape[0]
        elif hasattr(model, "weights_"):
            num_components = model.weights_.shape[0]

        # prediction -> data points are assigned to clusters from training!
        pred_clusters = model.predict(x_unknown_profile)
        train_clusters = model.predict(train_data)
        # find out which cluster is which label!
        y_pred = assign_clusters_single_profile(pred_clusters, train_targets, train_clusters, num_components)

    else:
        raise ValueError("""This Model implementation types does not exist.
        Choose one of the following: \"scikit\" (for rf, rf_bal, svm, knn, easy_ensemble, self_trainer),
        \"semi_manual\" (for kmean, gmm, bmm), \"keras\" (for lstm, blsm, enc_dec),
        or \"baseline\" (for the majority vote baseline)""")

    return y_pred

def load_markers(marker_path):
    """ Loads and returns sfc and ground markers with profile name as key
    Parameters:
        marker_path (Path): where the markers are stored
    Returns:
        dict < smp_name: (sfc_marker, ground_marker) >: marker dictionary
    """
    # load markers
    with open(marker_path, 'r') as file:
        marker_dic = {}
        for line in file:
            line = line.strip('\n')
            (key, sfc_val, ground_val) = line.split(',')
            marker_dic[key] = (float(sfc_val), float(ground_val))
    return marker_dic

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

def predict_all(unlabelled_dir=IN_DIR, marker_path=MARKER_PATH, mm_window=1, overwrite=True):
    """  Main function to predict the given set of profiles
    Parameters:
        unlabelled_dir (Path): where the unlabelled data is stored
        marker_path (Path): csv file with sfc and ground markers
        mm_window (int or float): default: one unit represents 1 mm. Choose the value
            you have used during data-preprocessing. Default value is 1mm there as well.
        overwrite (bool): default = False means if the file exists it is not overwritten but skipped
    """
    # TODO make this a function parameter
    location = "output/predictions/"
    Path(location).mkdir(parents=True, exist_ok=True)

    # we need some of the information from our training data
    with open("data/preprocess_data.txt", "rb") as handle:
        data = pickle.load(handle)

    # get current git commit
    repo = git.Repo(search_parent_directories=True)
    git_id = repo.head.object.hexsha

    models = ["baseline","rf", "rf_bal", 
                "svm","lstm", "blstm", "enc_dec", "easy_ensemble"]
    models = ["baseline"]
    # TODO svm ()
    smp_profiles = load_profiles(unlabelled_dir)

    #markers = load_markers(marker_path)
    #print("Finished load markers")

    # for all desired models create predictions
    for model_name in models:
        print("Starting to create predictions for model {}:".format(model_name))
        sub_location_ini = location + "/" + model_name + "/ini/"
        # make dir if it doesnt exist yet
        if not os.path.exists(sub_location_ini):
            os.makedirs(sub_location_ini)

        sub_location_plot = location + "/" + model_name + "/plot/"
        # make dir if it doesnt exist yet
        if not os.path.exists(sub_location_plot):
            os.makedirs(sub_location_plot)

        # load model
        model, model_type = load_stored_model(model_name)

        # for all desired data create model predictions
        for unlabelled_smp in tqdm(smp_profiles):
            # reset index for unlabelled_smp
            unlabelled_smp.reset_index(inplace=True, drop=True)
            # get smp idx
            smp_idx_str = int_to_idx(unlabelled_smp["smp_idx"][0])
            save_file = sub_location_ini + "/" + smp_idx_str + ".ini"

            # load markers 
            markers = load_markers_ini(unlabelled_dir, smp_idx_str)

            # predict profile
            if (not Path(save_file).is_file()) or overwrite:
                # fill nans
                unlabelled_smp.fillna(method="ffill", inplace=True)
                # if only nans, ffill won't work, but in this case: skip profile
                if sum(unlabelled_smp.isnull().any(axis=1)) == 0:

                    labelled_smp = predict_profile(unlabelled_smp, model, data, model_type)
                    
                    labelled_smp_df = pd.DataFrame(labelled_smp, columns=["label"])
                    #delete old labels of unlabelled_smp
                    unlabelled_smp.drop(columns=["label"], inplace=True)
                    #merge unlabelled_smp with labelled_smp to combine all informations for visualization
                    smp = pd.concat([unlabelled_smp, labelled_smp_df], axis=1)

                    try: # get markers
                        sfc, ground = markers["surface"], markers["ground"]
                        # save ini 
                        save_as_ini(labelled_smp, sfc, ground, save_file, model_name, git_id)
                        #save_as_ini(labelled_smp, save_file, model_name, git_id)
                    except KeyError:
                        print("Skipping Profile " + smp_idx_str + " since it is not contained in the marker file.")
                    # save figs
                    #save_as_pic() #does not work yet
                    #smp_idx_list = [float(smp_idx_str)]
                    smp_idx_float = float(smp_idx_str)
                    smp_labelled(smp, smp_idx_float, file_name=sub_location_plot+smp_idx_str)
                    #export as caaml
                    export(settings, derivatives, grain_shapes, smp_idx_float, timestamp, longitude, latitude, altitude, outfile)
                else:
                    print("Skipping Profile "+ smp_idx_str + " since it contains to many NaNs.")
        #smp_idx_list = [float(smp_idx_str)]
        #plot_testing(y_pred=labelled_smp, y_pred_prob=[], metrics_per_label=[], x_test=[], y_test=[],
        #                 smp_idx_test=smp_idx_str, labels_order=None, annot="test", name=model_name, only_preds=True, plot_list= smp_idx_list, save_dir=location, **PARAMS)


def save_as_ini(labelled_smp, sfc, ground, location, model, git_commit_id, mm_window=1):  #(labelled_smp, location, model, git_commit_id, mm_window=1)
    """ Save the predictions of an smp profile as ini file.
    Parameters:
        labelled_smp (list): predictions of a single unknown smp profile
        sfc (float): marker where the profiles surface originally was
        ground (float): marker where the profiles ground originally was
        location (str): where to save the ini file. Could be output/predictions/
            or within a file directory
        model (str): which model was used to generate the prediction
        mm_window (int or float): default: one unit represents 1 mm. Choose the value
            you have used during data-preprocessing. Default value is 1mm there as well.
    """
    # find out how ini files look like
    # .ini files are simple text files
    # distance values of the labelled smp file
    dist_list = [dist * mm_window for dist in range(len(labelled_smp))]
    # move window over labelled smp

    labels_occs = [(key, sum(1 for i in group)) for key, group in groupby(labelled_smp)]
    str_label_dist_pairs = [] # (label_str, last dist point where label occurs (inklusive!))
    complete_dist = 0

    # label_occ: (label, number of consecutive occurences)
    for label_occ in labels_occs:
        str_label = ANTI_LABELS[label_occ[0]]
        #if complete_dist == 0:
        #    dist = (label_occ[1] - 1 + sfc) * mm_window
        #else:
        #    dist = (label_occ[1]) * mm_window#
        dist = (label_occ[1]) * mm_window
        complete_dist += dist
        str_label_dist_pairs.append((str_label, complete_dist))

    with open(location, 'w') as file:
        file.write("[markers]\n")
        # file.write("# [model] = " + model) # model must be included in function
        # add surface marker
        file.write("surface = " + str(sfc) + "\n")
        # add snowgrain markers
        for (label, dist) in str_label_dist_pairs:
            file.write(label + " = " + str(dist) + "\n")
        # ground level is the same like last label
        file.write("ground = " + str(ground) + "\n")
        file.write("[model]\n")
        file.write(model + " = " + git_commit_id)

def load_profiles(data_dir, overwrite=False):
    """
    Returns:
        list <pd.Dataframe>: normalized smp data
    """
    #export_dir = Path("data/smp_profiles_unlabelled/") export_dir = Path("data/smp_profiles_updated/")
    #export_dir = Path("data/smp_profiles/") #=in data_loader wird die auch verwendet, hier ist export dir= exp_loc
    export_dir = Path("data/smp_npz_profiles_pred/")
    data_dir = Path(data_dir)
    marker_path = Path("data/sfc_ground_markers.csv")
    export = False
    markers = False
    filter = False #True
    # export data from pnt to csv or npz
    if export:
        export_pnt(pnt_dir=data_dir, target_dir=export_dir, export_as="npz", overwrite=False, **PARAMS)
    if markers:
        search_markers(pnt_dir=data_dir, store_dir=marker_path)

    # load_data(npz_name, test_print=False, **kwargs)
    # load pd.DataFrame from all npz files and save this pd as united DataFrame in npz
    all_smp = npz_to_pd(export_dir, is_dir=True)

    # Filter all profiles out that are already labelled
    if filter:
        # unlabelled data
        all_smp = all_smp[(all_smp["label"] == 0)]

    # normalize the data to get correct predictions
    #all_smp = normalize_mosaic(all_smp) #not for other data than mosaic

    # create list structure of smp profiles
    num_profiles = all_smp["smp_idx"].nunique()
    num_points = all_smp["smp_idx"].count()
    idx_list = all_smp["smp_idx"].unique()
    smp_list = [all_smp[all_smp["smp_idx"] == idx] for idx in idx_list]

    print("Finished Loading " + str(len(smp_list)) + " Profiles")

    return smp_list

def load_stored_model(model_name):
    """
    """
    # find out which model_type we have
    if model_name == "baseline":
        model_type = "baseline"
    elif model_name == "bmm" or model_name == "gmm" or model_name == "kmeans":
        model_type = "semi_manual"
    elif model_name in ["easy_ensemble", "knn", "label_spreading", "self_trainer", "rf", "rf_bal", "svm"]:
        model_type = "scikit"
    elif model_name == "lstm" or model_name == "blstm" or model_name == "enc_dec":
        model_type = "keras"
    else:
        raise ValueError("The model you have chosen does not exist in the list of stored models. Consider running and storing it.")

    # get stored model (models/stored_models)
    if model_type == "keras":
        if model_name != "enc_dec":
            model_filename = "models/stored_models/" + model_name + ".hdf5"
            loaded_model = keras.models.load_model(model_filename)
        else:
            model_filename = "models/stored_models/" + model_name + ".keras"
            loaded_model = keras.models.load_model(model_filename, custom_objects={"SeqSelfAttention": SeqSelfAttention})
    else:
        model_filename = "models/stored_models/" + model_name + ".model"
        with open(model_filename, "rb") as handle:
                if model_name == "rf":
                    loaded_model = joblib.load(handle)
                else:
                    loaded_model = pickle.load(handle)

    return loaded_model, model_type

def make_dirs():
    """
    """
    pass

def save_as_pic():
    """
    """
    # TODO make function for afterwards instead!!!
    # generate picture only for specific profiles
    pass

def export(settings, derivatives, grain_shapes, prof_id, timestamp,
    longitude, latitude, altitude, outfile):
    """Source: https://github.com/slf-dot-ch/snowmicropyn/blob/master/snowmicropyn/serialize/caaml.py
    Adapted to work in snowdragon 
    CAAML export of an SMP snow profile with forces and derived values. This routing writes
    a CAAML XML file containing:
      - A stratigraphy profile with layers as would be contained in a manual snow profile.
        The observables in here are parameterized by means of regressions and machine learning.
      - A density profile. The values are parameterized from a shot noise model for the SMP forces.
      - A specific surface are profile. The values are parameterized from a shot noise model for the
        SMP forces.
      - A hardness profile. These are the directly measured SMP forces (but resampled and pre-processed).

    param settings: Dictionary with export settings. Relevant to this routine are the settings/dictionary
    keys 'location_name', 'altitude', 'slope_exposition' and 'slope_angle'. In addition, please have a
    look at the subroutines that are called.
    param derivatives: Pandas dataframe with derived SMP quantities.
    param grain_shapes: List of grain shapes (one entry per SMP data row).
    param prof_id: Profile id which will be written in the 'id' attribute.
    param timestamp: Date and time of measurement.
    param smp_serial: Serial number of the SMP device.
    param longitude: Longitude of point of measurement.
    param latitude: Latitude of point of measurement.
    param outfile: Filename to save to.
    """
    mm2cm = lambda mm : mm / 10
    m2mm = lambda m : m * 1000
    cm2m = lambda cm : cm / 100
    #parameterization = _get_parameterization_name(derivatives)
    parameterization = None

    # We keep two sets of derivatives: one for the stratigraphy profile with merged layers and
    # one with only basic pre-processing for the embedded density, SSA and hardness profiles
    # (because we don't want only 1 data point per thick layer for the embedded profiles):
    #derivatives = preprocess_lowlevel(derivatives, settings)
    #layer_derivatives, grain_shapes, profile_bottom = preprocess_layers(derivatives,
    #    grain_shapes, settings)

    # Meta data:
    root = ET.Element(f'{_ns_caaml}:SnowProfile')
    root.set(f'xmlns:{_ns_caaml}', _ns[_ns_caaml])
    root.set(f'xmlns:{_ns_gml}', _ns[_ns_gml])
    root.set(f'{_ns_gml}:id', prof_id)

    meta_data = ET.SubElement(root, f'{_ns_caaml}:metaData')
    #_addGenericComments(meta_data, parameterization)

    time_ref = ET.SubElement(root, f'{_ns_caaml}:timeRef')
    rec_time = ET.SubElement(time_ref, f'{_ns_caaml}:recordTime')
    time_inst = ET.SubElement(rec_time, f'{_ns_caaml}:TimeInstant')
    time_pos = ET.SubElement(time_inst, f'{_ns_caaml}:timePosition')
    time_pos.text = timestamp.isoformat()

    src_ref = ET.SubElement(root, f'{_ns_caaml}:srcRef')
    src_oper = ET.SubElement(src_ref, f'{_ns_caaml}:Operation')
    #src_oper.set(f'{_ns_gml}:id', 'SMP_serial')
    #src_name = ET.SubElement(src_oper, f'{_ns_caaml}:name')
    #src_name.text = smp_serial

    loc_ref = ET.SubElement(root, f'{_ns_caaml}:locRef')
    loc_ref.set(f'{_ns_gml}:id', 'LOC_ID')
    loc_name = ET.SubElement(loc_ref, f'{_ns_caaml}:name')
    loc_name.text = settings.get('location_name', 'SMP observation point')
    obs_sub = ET.SubElement(loc_ref, f'{_ns_caaml}:obsPointSubType')
    obs_sub.text = 'SMP profile location'
    if altitude: # SMP altitude is in cm
        altitude = cm2m(altitude)
    else: # if no altitude is recorded by the SMP we insert the user chosen one
        altitude = settings.get('altitude')
    if altitude:
        valid_elevation = ET.SubElement(loc_ref, f'{_ns_caaml}:validElevation')
        val_el_pos = ET.SubElement(valid_elevation, f'{_ns_caaml}:ElevationPosition')
        val_el_pos.set('uom', 'm')
        caaml_position = ET.SubElement(val_el_pos, f'{_ns_caaml}:position')
        caaml_position.text = str(altitude)
    valid_aspect = ET.SubElement(loc_ref, f'{_ns_caaml}:validAspect')
    aspect_pos = ET.SubElement(valid_aspect, f'{_ns_caaml}:AspectPosition')
    caaml_aspect = ET.SubElement(aspect_pos, f'{_ns_caaml}:position')
    caaml_aspect.text = str(settings.get('slope_exposition', 0))
    valid_slope_angle = ET.SubElement(loc_ref, f'{_ns_caaml}:validSlopeAngle')
    valid_slope_angle = ET.SubElement(valid_slope_angle, f'{_ns_caaml}:SlopeAnglePosition')
    valid_slope_angle.set('uom', 'deg')
    caaml_angle = ET.SubElement(valid_slope_angle, f'{_ns_caaml}:position')
    caaml_angle.text = str(settings.get('slope_angle', 0))

    point_loc = ET.SubElement(loc_ref, f'{_ns_caaml}:pointLocation')
    point_pt = ET.SubElement(point_loc, f'{_ns_gml}:Point')
    point_pt.set(f'{_ns_gml}:id', 'pointID')
    point_pt.set('srsName', 'urn:ogc:def:crs:OGC:1.3:CRS84')
    point_pt.set('srsDimension', '2')
    point_pos = ET.SubElement(point_pt, f'{_ns_gml}:pos')
    point_pos.text = f'{longitude} {latitude}'

    # Stratigraphy profile:
    snow_prof = ET.SubElement(root, f'{_ns_caaml}:snowProfileResultsOf')
    snow_prof_meas = ET.SubElement(snow_prof, f'{_ns_caaml}:SnowProfileMeasurements')
    snow_prof_meas.set('dir', 'top down')
    strat_prof = ET.SubElement(snow_prof_meas, f'{_ns_caaml}:stratProfile')
    strat_meta = ET.SubElement(strat_prof, f'{_ns_caaml}:stratMetaData')

    for idx, row in layer_derivatives.iterrows():
        layer = ET.SubElement(strat_prof, f'{_ns_caaml}:Layer')
        depth_top = ET.SubElement(layer, f'{_ns_caaml}:depthTop')
        depth_top.set('uom', 'cm')
        depth_top.text = str(mm2cm(row['distance']))
        thickness = ET.SubElement(layer, f'{_ns_caaml}:thickness')
        thickness.set('uom', 'cm')
        if idx == len(layer_derivatives) - 1:
            layer_thickness = profile_bottom - row.distance
        else:
            layer_thickness = layer_derivatives.distance[idx + 1] - row.distance
        thickness.text = str(mm2cm(layer_thickness))
        if len(grain_shapes) > 0: # we have grain classification available
            grain_primary = ET.SubElement(layer, f'{_ns_caaml}:grainFormPrimary')
            grain_primary.text = grain_shapes[idx]
        grain_size = ET.SubElement(layer, f'{_ns_caaml}:grainSize')
        grain_size.set('uom', 'mm')
        grain_components = ET.SubElement(grain_size, f'{_ns_caaml}:Components')
        grain_sz_avg = ET.SubElement(grain_components, f'{_ns_caaml}:avg')
        grain_sz_avg.text = str(m2mm(optical_thickness(row[f'{parameterization}_ssa'])))
        grain_hardness = ET.SubElement(layer, f'{_ns_caaml}:hardness')
        grain_hardness.set('uom', '')
        grain_hardness.text = hand_hardness_label(row['force_median'])

    # Density profile:
    dens_prof = ET.SubElement(snow_prof_meas, f'{_ns_caaml}:densityProfile')
    dens_meta = ET.SubElement(dens_prof, f'{_ns_caaml}:densityMetaData')
    dens_meth = ET.SubElement(dens_meta, f'{_ns_caaml}:methodOfMeas')
    dens_meth.text = "other"

    for _, row in derivatives.iterrows():
        layer = ET.SubElement(dens_prof, f'{_ns_caaml}:Layer')
        depth_top = ET.SubElement(layer, f'{_ns_caaml}:depthTop')
        depth_top.set('uom', 'cm')
        depth_top.text = str(mm2cm(row['distance']))
        density = ET.SubElement(layer, f'{_ns_caaml}:density')
        density.set('uom', 'kgm-3')
        density_val = row[f'{parameterization}_density']
        density.text = str(density_val)

    # Specific surface area profile:
    ssa_prof = ET.SubElement(snow_prof_meas, f'{_ns_caaml}:specSurfAreaProfile')
    ssa_meta = ET.SubElement(ssa_prof, f'{_ns_caaml}:specSurfAreaMetaData')
    ssa_meth = ET.SubElement(ssa_meta, f'{_ns_caaml}:methodOfMeas')
    ssa_meth.text = "other"
    ssa_comp = ET.SubElement(ssa_prof, f'{_ns_caaml}:MeasurementComponents')
    ssa_comp.set('uomDepth', 'cm')
    ssa_comp.set('uomSpecSurfArea', 'm2kg-1')
    ssa_depth = ET.SubElement(ssa_comp, f'{_ns_caaml}:depth')
    ssa_res = ET.SubElement(ssa_comp, f'{_ns_caaml}:specSurfArea')
    ssa_meas = ET.SubElement(ssa_prof, f'{_ns_caaml}:Measurements')
    ssa_tuple = ET.SubElement(ssa_meas, f'{_ns_caaml}:tupleList')

    tuple_list = ''
    for _, row in derivatives.iterrows():
        tuple_list = tuple_list + str(mm2cm(row['distance'])) + "," + str(row[f'{parameterization}_ssa']) + " "
    ssa_tuple.text = tuple_list

    # Hardness profile:
    hard_prof = ET.SubElement(snow_prof_meas, f'{_ns_caaml}:hardnessProfile')
    hard_meta = ET.SubElement(hard_prof, f'{_ns_caaml}:hardnessMetaData')
    hard_meth = ET.SubElement(hard_meta, f'{_ns_caaml}:methodOfMeas')
    hard_meth.text = "SnowMicroPen"
    hard_comp = ET.SubElement(hard_prof, f'{_ns_caaml}:MeasurementComponents')
    hard_comp.set('uomDepth', 'cm')
    hard_comp.set('uomHardness', 'N')
    hard_depth = ET.SubElement(hard_comp, f'{_ns_caaml}:depth')
    hard_res = ET.SubElement(hard_comp, f'{_ns_caaml}:penRes')
    hard_meas = ET.SubElement(hard_prof, f'{_ns_caaml}:Measurements')
    hard_tuple = ET.SubElement(hard_meas, f'{_ns_caaml}:tupleList')

    tuple_list = ''
    for _, row in derivatives.iterrows():
        tuple_list = tuple_list + str(mm2cm(row['distance'])) + "," + str(mm2cm(row['force_median'])) + " "
    hard_tuple.text = tuple_list

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0) # human-readable CAAML
    tree.write(outfile, encoding="UTF-8", xml_declaration=True)

if __name__ == "__main__":
    predict_all()
