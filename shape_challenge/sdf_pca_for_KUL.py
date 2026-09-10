# =============================================================================
#  The Shape Space Challenge, Mathematics for Industry 2026
#  Geometry Analysis and Design Space Exploration
#
#  Provided by: Noesis Solutions NV
#  Developed by: Research & Development Team
#  Version: 1.0
#  Date: September, 2026
#
#  PURPOSE
#  -------
#  This code is provided as a starting point for The Shape Space Challenge.
#  It is intended for educational and research purposes and may be modified,
#  extended, or replaced by the participants as part of the challenge.
#
#  DISCLAIMER
#  ----------
#  THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED. The authors and Noesis Solutions NV make no
#  representations or warranties regarding the accuracy, completeness,
#  reliability, suitability, or fitness for a particular purpose of the
#  software or of any results obtained from its use.
#
#  The user assumes all responsibility and risk arising from the use of this
#  software and from any conclusions or decisions based on its results.
#  Neither the authors nor Noesis Solutions shall be liable for any
#  direct, indirect, incidental, consequential, or other damages arising
#  from or in connection with the use of this software.
#
#  IMPORTANT
#  ---------
#  The code is provided as a reference implementation and is NOT intended
#  to represent a validated or production-ready engineering tool.
#  Results must not be used as the sole basis for engineering, safety,
#  manufacturing, or other real-world decisions.
#
#  =============================================================================


import numpy as np
import logging
import os
import sys
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances  # used in distance matrix

import generic_utils as gen_utils
import geometry_utils as geom_utils
import plot_utils as plot_utils
import sdf_pca_utils as sdf_pca
import sdf_plot_utils as sdf_plot

# ensure reproducibility
standard_seed = 42
np.random.seed(seed=standard_seed)

# removes warning from UMAP
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")

# ======================================================================================================================
#                                                                                                                Logging
# ======================================================================================================================
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# scalers to normalize
scaler_params = MinMaxScaler(feature_range=(-1, 1))  # StandardScaler()
scaler_design_variables = MinMaxScaler(feature_range=(-1, 1))  # StandardScaler()

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                   Paths and parameters
# ----------------------------------------------------------------------------------------------------------------------
case_id = 3  # index from following list
case_list = ["Tesla_valve", "Sokaris", "Blended", "Fan"]
case = case_list[case_id]  # "tesla_valve" "sokaris" "blended"

# number of grip points on the smallest of the geometric axes
if case_id == 0:
    min_grid_number = 5
elif case_id == 1:
    min_grid_number = 25
elif case_id == 2:
    min_grid_number = 16
elif case_id == 3:
    min_grid_number = 30
else:
    logging.error(f"Undefined case: {case_id}")
    sys.exit(1)

# reload boundaries from previous case and sdfs
load_project = False
# suppress all additional logs messages while loading geometries in load_project
suppress_all_logs = True
# 'first_PCA_component', 'hierarchical_clustering', 'optimal_leaf'
sorting_logic = 'optimal_leaf'

generate_images = True  # Use it once to generate the images for all the geometries (using load_project = False)
reduced_exp_number = 50  # limits the total number of geometries loaded, None to load all the available ones
n_closest_geometries = 1  # number of the closest geometries to be identifed for every candidate

# 2. Generate datasets, training and evaluation
debug_sdf = False  # Adds slice by slice SDFvisualization
sdf_visualization_slice_number = int(min_grid_number / 2)  # number of slices for the slide by slice plot of the sdf
clip = 0.2  # sdf clipping distance
sdf_alpha_weight = 5  # tune this according to the problem

# 3. Apply PCA on training set
plot_cumulative_pca = False  # plot components vs representativeness
# save latent space to file
save_latent_space = True

# 4. Report variance,
full_variance_report = False  # report variance and cumulative variance for all modes
variance_threshold = .99  # target variance to be achieved (number of modes to get this or better)

# 5. Visualize mode variations,
evaluate_modes = False  # adds visualization of the first min(5, n_modes)

# 6. Reconstruct a shape from the training set
# number of test reconstruction from latent space back to full goemetry, None to skip else number of tests
evaluate_reconstruction = True
n_reconstructions = 2

# 7. Build distance metrics on training set
make_plot_distance_matrix = False

# 8. Compare new geometries (random or specific experiments removed from training set)
# also used in UMAP to see wheere they land
evaluate_new_samples = False  # number of random geometries used for evaluation, False to skip
n_new_samples = 0  # number of tests
evaluate_specified_new_samples = False  # number of geometries used for evaluation, False to skip
plot_new_samples = False  # plot the new geometries compared with existing ones
plot_pca_foreach_new_experiment = True  # plot each new experiment in latent space
plot_comparison_foreach_new_experiment = False  # plot distance of proposed experiment from training dataset
specific_new_samples_ID = None  # if None, n_new_samples random are selected, otherwise specified ids are removed

# remove and later evaluate specific experiments, like most similar and most different
if evaluate_specified_new_samples or True:
    if case_id == 0:  # tesla
        specific_new_samples_ID = [0, 7, 10, 19, 21]
    elif case_id == 1: # sokaris
        specific_new_samples_ID = [4, 28, 29, 33]
    elif case_id == 2:  # bwb
        specific_new_samples_ID = [2, 12, 23, 41, 42, 45]
    elif case_id == 3:  # fan
        specific_new_samples_ID = [6, 9, 19, 23, 34, 35]
    else:
        logging.error(f"Undefined case: {case_id}")
        sys.exit(1)

# 9. Most isolated geometry
evaluate_most_isolated_geometry = True
make_plot_most_isolated_geometry = True

# 10. Least isolated
evaluate_least_isolated_geometry = True
make_plot_most_common_geometry = True

# 11. Visualize domain in PCA
make_plot_domain_scatter = True

# 12. Visualize SOM - Self Organizing Maps
evaluate_som = True  # uses SOM to represent experiments in the domain

# 13. Visualize UMAP - Uniform Manifold Approximation and Projection for Dimension Reduction
evaluate_umap = True  # uses UMPA to represent experiments in the domain

# 14. Propose new geometries
generate_new_samples = 0  # number of new geometries that will be created from maximin of latent space
make_plot_new_maximin = True
make_plot_new_gaussian = True
make_plot_new_multigaussian = True
make_plot_new_interpolate = True

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
base_source_path = os.path.join(f"test_files\\{case}\\Cases\\Case1")

# prepare output folders
images_folder = os.path.join(os.path.join(f"results\\images\\{case}"))
gen_utils.prepare_folder(images_folder, delete_if_exists=False)
gen_utils.prepare_folder(os.path.join(images_folder, 'mesh'),
                         delete_if_exists=generate_images)  # will delete folder if generation is requested
jsons_folder = os.path.join(os.path.join(f"results\\jsons\\{case}"))
gen_utils.prepare_folder(jsons_folder, delete_if_exists=False)
sdfs_folder = os.path.join(os.path.join(f"results\\sdfs\\{case}"))
gen_utils.prepare_folder(sdfs_folder, delete_if_exists=False)
pcas_folder = os.path.join(os.path.join(f"results\\pcas\\{case}"))
gen_utils.prepare_folder(pcas_folder, delete_if_exists=False)

case_to_exp_dict = None
if os.path.exists(os.path.join(base_source_path, "case_to_exp.json")):
    case_to_exp_dict = gen_utils.read_json(os.path.join(base_source_path, "case_to_exp.json"))

full_source_path = os.path.join(base_source_path, "TrainingBlock1\\ImportedData")
dataset_json_file = gen_utils.find_dataset_json(full_source_path)  # config.json and database_configuration.json
logging.info(f"Found dataset json: {dataset_json_file}")

# check config file exists
gen_utils.is_file_extension_XXX(dataset_json_file, 'json', n_digits=4)

# read json files
jsonobj_database = gen_utils.read_json(dataset_json_file)
attributes_json = os.path.join(full_source_path, 'attributes_on_trainingSet.json')

jsonobj_attributes = gen_utils.read_json(attributes_json)

# get settings to verify them
config_name, config_type, config_file = gen_utils.config_header_info(jsonobj_database)

# some consistency checks
folder_names = jsonobj_database['experiments']['folder_names']

logging.info("========================================================================================================")
logging.info(f"Loading from: {full_source_path}")
logging.info('\tconfig file name: {0}'.format(config_name))

input_variable_names = gen_utils.get_variable_names(jsonobj_database, 'inputs')

# remove "ShellMesh" from the input list if found
if "ShellMesh" in input_variable_names:
    input_variable_names.remove("ShellMesh")
    del jsonobj_database["variables"]["inputs"]["ShellMesh"]

# Fields attributes initialization
keys = ['inputs', 'outputs', 'coordinates']
variables_definitions = dict(zip(keys, ({} for _ in keys)))
variables_definitions['inputs'] = dict(zip(input_variable_names, ({'Ncomp': 1} for _ in input_variable_names)))
variables_attributes = gen_utils.initialize_variables_attributes(variables_definitions=variables_definitions)

logging.info(f"Design variables: {input_variable_names}")
design_variables_data = {var: [] for var in input_variable_names}
design_variables_filenames = {var: jsonobj_database["variables"]["inputs"][var]["basename"] for var in
                              input_variable_names}

base_dir = os.path.dirname(dataset_json_file)
exp_folder_basename = jsonobj_database['experiments']['folder_names'].replace('%d', '')
list_of_exp_ = [f for f in os.listdir(base_dir) if (os.path.isdir(os.path.join(base_dir, f))) and
                (exp_folder_basename in os.path.join(base_dir, f))]
list_of_exp_ids_ = [int(f.replace(exp_folder_basename, '')) for f in list_of_exp_]
list_of_exp_sorted_ = [x for _, x in sorted(zip(list_of_exp_ids_, list_of_exp_))]

# limit experiments with respect to full dataset
if reduced_exp_number:
    if len(list_of_exp_sorted_) < reduced_exp_number:
        logging.info(f"Number of available experiments ({len(list_of_exp_sorted_)}) is already smaller "
                     f"than reduced_exp_number {reduced_exp_number}")
    list_of_exp_sorted_ = list_of_exp_sorted_[:reduced_exp_number]
logging.info(f"Experiments: {list_of_exp_sorted_}")

# remove evaluation experiments if requested
if evaluate_new_samples:
    evaluation_experiments_id = np.random.randint(len(list_of_exp_sorted_), size=n_new_samples)
elif evaluate_specified_new_samples:
    evaluation_experiments_id = specific_new_samples_ID
else:
    evaluation_experiments_id = []


# ----------------------------------------------------------------------------------------------------------------------
#                                                                                       Investigate Geometries from Mesh
# ----------------------------------------------------------------------------------------------------------------------
logging.info(" Loop on Sources ---------------------------------------------------------------------------------------")
geometric_bb = {'x_min': sys.float_info.max, 'x_max': -1. * sys.float_info.max,
                'y_min': sys.float_info.max, 'y_max': -1. * sys.float_info.max,
                'z_min': sys.float_info.max, 'z_max': -1. * sys.float_info.max}

training_meshes = []
training_meshes_exp = []
evaluation_meshes = []
evaluation_meshes_exp = []

# list of geometry name, experiment folder, bounding box of the geometry
case_exp_bb = []

# loop on loaded experiments if possible and requested
filename_case_exp = f"{len(list_of_exp_sorted_)}.json"
if load_project and not os.path.exists(os.path.join(jsons_folder, filename_case_exp)):
    logging.critical(f'Unable to load file {os.path.join(jsons_folder, filename_case_exp)}')
    logging.critical(f'Switching to non-load configuration!')
    load_project = False

# check if mesh files are available, otherwise end execution
for source_mesh_id, folder_exp in enumerate(list_of_exp_sorted_):
    no_stl_found, missing_folder = geom_utils.all_folders_contain_stl([os.path.join(full_source_path, folder_exp)])
    if not no_stl_found:
        logging.critical('No stl file provided, ending execution for geometries')
        sys.exit(1)


# load geometries and evaluate BB, or load BB from file
if load_project and os.path.exists(os.path.join(jsons_folder, filename_case_exp)):
    logging.info(f"Loading bounding boxes from {os.path.join(jsons_folder, filename_case_exp)}")
    case_exp_bb = gen_utils.read_json(os.path.join(jsons_folder, f'{len(list_of_exp_sorted_)}.json'))

    for geom_entry in case_exp_bb:
        geometric_bb = sdf_pca.update_bb_min_max(geometric_bb,
                                                 geom_entry['bounding_box'][0], geom_entry['bounding_box'][1],
                                                 geom_entry['bounding_box'][2], geom_entry['bounding_box'][3],
                                                 geom_entry['bounding_box'][4], geom_entry['bounding_box'][5])

    # split training and evaluation
    for source_mesh_id, folder_exp in enumerate(list_of_exp_sorted_):
        if source_mesh_id not in evaluation_experiments_id:
            training_meshes_exp.append(folder_exp)
        else:
            evaluation_meshes_exp.append(folder_exp)

else:
    logging.info(f"Processing {len(list_of_exp_sorted_)} geometries and evaluating bounding boxes")
    for source_mesh_id, folder_exp in enumerate(list_of_exp_sorted_):

        logging.info(f"{folder_exp}")
        source_mesh, source_col, source_mesh_aabb = geom_utils.load_mesh_wrapper(logger, dataset_json_file,
                                                                                 full_source_path,
                                                                                 folder_exp,
                                                                                 suppress_all=False)

        geometric_bb = sdf_pca.update_bb_min_max(geometric_bb,
                                                 source_mesh_aabb.get_min_bound()[0],
                                                 source_mesh_aabb.get_max_bound()[0],
                                                 source_mesh_aabb.get_min_bound()[1],
                                                 source_mesh_aabb.get_max_bound()[1],
                                                 source_mesh_aabb.get_min_bound()[2],
                                                 source_mesh_aabb.get_max_bound()[2])

        logging.debug(f"\t{source_mesh}")
        logging.debug(
            f"\tVertices: {np.asarray(source_mesh.vertices).shape} / Triangles: {np.asarray(source_mesh.triangles).shape}")

        # update dictionary with case, folder and bb
        if case_to_exp_dict:
            geometry_name = next((k for k, v in case_to_exp_dict.items() if v == folder_exp), None)
        else:
            geometry_name = folder_exp

        case_exp_bb.append({"geometry": geometry_name,
                            "folder": folder_exp,
                            "bounding_box": list([source_mesh_aabb.get_min_bound()[0],
                                                  source_mesh_aabb.get_max_bound()[0],
                                                  source_mesh_aabb.get_min_bound()[1],
                                                  source_mesh_aabb.get_max_bound()[1],
                                                  source_mesh_aabb.get_min_bound()[2],
                                                  source_mesh_aabb.get_max_bound()[2]])})

        # split traing and eval
        if source_mesh_id not in evaluation_experiments_id:
            training_meshes.append(source_mesh)
            training_meshes_exp.append(folder_exp)
        else:
            evaluation_meshes.append(source_mesh)
            evaluation_meshes_exp.append(folder_exp)

        # generate images for all the loaded geometries
        if generate_images:
            for axis in ['x', 'y', 'z']:
                plot_utils.save_geometries_image([source_mesh],
                                                 os.path.join(images_folder, "mesh", f"{folder_exp}.png"),
                                                 window_title=folder_exp,
                                                 axis=axis)
    gen_utils.write_json(case_exp_bb, jsons_folder, f"{len(list_of_exp_sorted_)}.json")


logging.warning("Now were gonna do the correlation mesh depending on Hausdorff and the others :D - Yarne")

relationship = sdf_pca.mesh_comparator(logger, training_meshes)

# D_training = pairwise_distances(training_latent, metric='euclidean')
# D_training_sorted, labels_training_sorted = sdf_pca.distance_matrix_sorting(sorting_type=sorting_logic,
#                                                                             latent_space=training_latent,
#                                                                             D=D_training,
#                                                                             labels=training_meshes_exp)
#
# # plot distance matrix, as is (experiment by experiment) and sorted according to selected logic
# if make_plot_distance_matrix:
#     sdf_plot.plot_distance_matrix(D_training, title_="General distance Matrix (PCA space)", labels=training_meshes_exp,
#                                   save_folder=images_folder)
#     sdf_plot.plot_distance_matrix(D_training_sorted, title_=f"Sorted ({sorting_logic}) General distance Matrix (PCA space)",
#                                   labels=labels_training_sorted,
#                                   save_folder=images_folder)
#
# logging.info("")



logging.info(f"Global geometry BB:")
logging.info(f"\tX: {round(geometric_bb['x_min'], 2)} to {round(geometric_bb['x_max'], 2)}")
logging.info(f"\tY: {round(geometric_bb['y_min'], 2)} to {round(geometric_bb['y_max'], 2)}")
logging.info(f"\tZ: {round(geometric_bb['z_min'], 2)} to {round(geometric_bb['z_max'], 2)}")
logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                    1. Create global grid for later SDF
# ----------------------------------------------------------------------------------------------------------------------
logging.info(" 1. Create global grid ---------------------------------------------------------------------------------")
grid_points, minimum_axis, n_grids_x, n_grids_y, n_grids_z, grid_size = sdf_pca.generate_sdf_grid(logger,
                                                                                                  geometric_bb,
                                                                                                  min_grid_number)
logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                          2. Generate datasets, training and evaluation
# ----------------------------------------------------------------------------------------------------------------------
logging.info(" 2. Generate datasets, training and evaluation ---------------------------------------------------------")
training_sdfs, valid_training_exps = sdf_pca.compute_sdfs(logger, training_meshes, grid_points,
                                                          geometry_dict=case_exp_bb, type_='Training',
                                                          make_plot=debug_sdf,
                                                          axis=minimum_axis, n_slices=sdf_visualization_slice_number,
                                                          clip_=clip, save_path=os.path.join(sdfs_folder),
                                                          exp_ids=training_meshes_exp,
                                                          load_sdf=load_project)
# check validation experiments
if len(valid_training_exps) != len(training_meshes_exp):
    logging.warning(f"{len(training_meshes_exp)} experiments provided, but there are only {len(valid_training_exps)}"
                    f" valid geometries")
    training_meshes_exp = valid_training_exps

# scaled sdf to focus on surface
training_sdfs = sdf_pca.apply_weight(training_sdfs, sdf_alpha_weight)
logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                           3. Apply PCA on training set
# ----------------------------------------------------------------------------------------------------------------------
logging.info(" 3. Apply PCA on training set --------------------------------------------------------------------------")
logging.info("\tEvaluationg PCA...")

training_pca_tmp = PCA(svd_solver="full")
training_pca_tmp.fit_transform(training_sdfs)
# Decide how many modes to keep
n_components_geom = sdf_pca.n_components_for_accuracy(logger, training_pca_tmp, threshold=variance_threshold)
del training_pca_tmp

training_pca = PCA(n_components=n_components_geom, svd_solver='full')
training_latent = training_pca.fit_transform(training_sdfs)
logging.debug(f"\tOriginal dimension: {training_sdfs.shape[0]} x {training_sdfs.shape[1]}")
logging.info(f"\tLatent dimension: {training_latent.shape[0]} x {training_latent.shape[1]}")
logging.info("")

bounds_training_latent = sdf_pca.get_bounds(training_latent)
logging.info(f"\tCurrent latent space bounds")
logging.info(f"\t\t{'Component':<10} {'Lower bound':>15} {'Upper bound':>15} {'Range':>15}")
for i_bound, (lower, upper) in enumerate(bounds_training_latent):
    logging.info(f"\t\t{i_bound:<10d} {lower:>15.3e} {upper:>15.3e} {upper - lower:>15.3e}")


if save_latent_space:
    np.savetxt(os.path.join(pcas_folder, 'training_geometries.out'), np.array(training_latent))

# Assess impact of number of modes on fidelity of the representation
pca_error_results = sdf_pca.evaluate_error_from_pca_components(logger, training_sdfs, max_components=len(training_sdfs),
                                                               make_plot=plot_cumulative_pca)
_, cvars, _ = zip(*pca_error_results)
logging.info("")

# generate clusters
n_clusters, kmeans, cluster_labels, cluster_centroids = sdf_pca.evaluate_clusters(logger, training_latent)

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                     4. Report variance
# ----------------------------------------------------------------------------------------------------------------------
logging.info(" 4. Report variance ------------------------------------------------------------------------------------")

# pca is fitted PCA object
explained_variance = training_pca.explained_variance_ratio_  # fraction of variance per component
# Cumulative variance
cumulative_variance = np.cumsum(explained_variance)

# number of modes to achive threshold or more
n_representative_modes = len(cumulative_variance)
for i_variance, variance in enumerate(cumulative_variance):
    if variance >= variance_threshold:
        logging.info(f"\tVariance threshold of {variance_threshold} reached with {i_variance + 1} modes")
        n_representative_modes = i_variance + 1
        break

if full_variance_report:
    logging.info("\tPCA mode contributions and Cumulative variance:")
    for i, var in enumerate(explained_variance):
        logging.info(
            f"\t\tMode {i + 1}: {var * 100:.2f}% of total variance, (Cumulative: {cumulative_variance[i] * 100:.2f})")
else:
    logging.info(f"\tAll modes ({len(explained_variance)}); Total variance: {cumulative_variance[-1] * 100:.2f})")

logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                           5. Visualize mode variations
# ----------------------------------------------------------------------------------------------------------------------
if evaluate_modes:
    logging.info(" 5. Visualize mode variations ----------------------------------------------------------------------")
    # Parameters
    alpha = 0.5  # plus minus scaling applied to mode
    # max 5 modes
    for mode_index in range(0, min(n_representative_modes, 5)):
        # plot modes
        sdf_plot.plot_pca_mode_true_scale(training_pca, n_grids_x, n_grids_y, n_grids_z, mode_index, alpha)

    logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                           6. Reconstruct a shape from the training set
# ----------------------------------------------------------------------------------------------------------------------
if evaluate_reconstruction:
    logging.info(" 6. Reconstruct a shape from the training set ------------------------------------------------------")
    for test_index in specific_new_samples_ID: #np.random.randint(len(training_meshes_exp), size=n_reconstructions):

        # need the geometry, if load_project than it is not available, and must be loaded
        if len(training_meshes) == 0 or load_project:
            folder_exp = training_meshes_exp[test_index]
            source_mesh, source_col, source_mesh_aabb = geom_utils.load_mesh_wrapper(logger, dataset_json_file,
                                                                                     full_source_path,
                                                                                     folder_exp,
                                                                                     suppress_all=suppress_all_logs)
        else:
            # otherwise just pick it from list
            source_mesh = training_meshes[test_index]

        logging.info(f"Test index geoemtry: {test_index}")
        # uses backward projection to create the geometry again and compares it with original geometry
        sdf_pca.reconstruct_mesh(logger, training_pca, training_latent[test_index], test_index,
                                 n_grids_x, n_grids_y, n_grids_z, training_sdfs,
                                 source_mesh, grid_size, make_plot_=True)
    logging.info("")


# ----------------------------------------------------------------------------------------------------------------------
#                                                                              7. Build distance metrics on training set
# ----------------------------------------------------------------------------------------------------------------------
logging.info(" 7. Build distance metrics on training set -------------------------------------------------------------")
# distance matrix (n_mesh x n_mesh)
D_training = pairwise_distances(training_latent, metric='euclidean')
D_training_sorted, labels_training_sorted = sdf_pca.distance_matrix_sorting(sorting_type=sorting_logic,
                                                                            latent_space=training_latent,
                                                                            D=D_training,
                                                                            labels=training_meshes_exp)

# plot distance matrix, as is (experiment by experiment) and sorted according to selected logic
if make_plot_distance_matrix or True:
    sdf_plot.plot_distance_matrix(D_training, title_="General distance Matrix (PCA space)", labels=training_meshes_exp,
                                  save_folder=images_folder)
    sdf_plot.plot_distance_matrix(D_training_sorted, title_=f"Sorted ({sorting_logic}) General distance Matrix (PCA space)",
                                  labels=labels_training_sorted,
                                  save_folder=images_folder)

logging.info("")

sdf_plot.plot_pca_projection(training_latent, None,
                             training_labels=training_meshes_exp, new_labels=[],
                             cluster_labels=cluster_labels, new_cluster_labels=[],
                             annotate=True,
                             title=f"Latent space, all training experiments\n(PCA1+PCA2 = {cvars[1]:>4.2e})",
                             case=case, save_folder=None)

# ----------------------------------------------------------------------------------------------------------------------
#                                              8. Compare new geometries (evaluations, not included in training dataset)
#                                                                                with existing ones to find similarities
# ----------------------------------------------------------------------------------------------------------------------
latent_all = training_latent
labels_all = training_meshes_exp.copy()
if (evaluate_new_samples and n_new_samples > 0) or evaluate_specified_new_samples:
    logging.info(" 8. Compare new geometries -------------------------------------------------------------------------")
    logging.info("\tComparing new geometries with training dataset")

    similarity_metrics_results_dict = {"case": case,
                                       "training_experiments": int(len(training_meshes_exp)),
                                       "evaluation_experiments": int(len(evaluation_meshes_exp)),
                                       "pca_modes": int(training_pca.n_components_),
                                       "clusters": int(kmeans.n_clusters),
                                       "experiments": []}

    # store all new experiments in latent space form
    latent_new_all = []
    cluster_ids = []

    # loop on new meshes
    for i_new_mesh, new_mesh_exp in enumerate(evaluation_meshes_exp):

        logging.info(f"\tTesting new experiment: {new_mesh_exp}")

        if load_project:
            new_mesh, new_col, new_mesh_aabb = geom_utils.load_mesh_wrapper(logger,
                                                                            dataset_json_file,
                                                                            full_source_path,
                                                                            new_mesh_exp,
                                                                            suppress_all=suppress_all_logs)
        else:
            new_mesh = evaluation_meshes[i_new_mesh]

        # Compute SDF and PCA on new mesh
        latent_new = sdf_pca.project_new_mesh(logger, new_mesh, grid_points, sdf_alpha_weight, training_pca)
        latent_new_all.append(latent_new)

        # evaluate distance
        distances, exp_result = sdf_pca.evaluate_new_experiment_similarity(logger, case,
                                                                           latent_new, new_mesh_exp,
                                                                           training_pca, training_latent, training_meshes_exp,
                                                                           n_clusters, kmeans, cluster_labels, cluster_centroids, cluster_ids,
                                                                           sorting_logic,
                                                                           scaler=None,
                                                                           make_plot_distance_matrix=make_plot_distance_matrix,
                                                                           plot_pca_foreach_new_experiment=plot_pca_foreach_new_experiment,
                                                                           images_folder=images_folder)

        # (n_samples, n_components)
        latent_all = np.vstack((latent_all, latent_new))
        labels_all.append(new_mesh_exp)

        # get distances from existing latent space geometries
        closest_idxs = sdf_pca.get_n_closest_indices(distances, n_closest=n_closest_geometries, exclude_self_idx=None)

        window_title = f"New {evaluation_meshes_exp[i_new_mesh]} (Y)"

        closest_meshes = []
        distance_values = []
        for i in range(n_closest_geometries):
            if load_project:
                closest_mesh, _, _ = geom_utils.load_mesh_wrapper(logger,
                                                                  dataset_json_file,
                                                                  full_source_path,
                                                                  training_meshes_exp[closest_idxs[i]],
                                                                  suppress_all=suppress_all_logs)
            else:
                closest_mesh = training_meshes[closest_idxs[i]]
            closest_meshes.append(closest_mesh)

            logging.info(f"\tClosest shape: {i}, index: {closest_idxs[i]}")
            logging.info(
                f"\t\tEvaluation exp. / closest training exp.: {evaluation_meshes_exp[i_new_mesh]} / {training_meshes_exp[closest_idxs[i]]}")
            logging.info(f"\t\tDistance: {'{:.2f}'.format(distances[closest_idxs[i]])}")

            distance_values.append(distances[closest_idxs[i]])
            window_title += f", {training_meshes_exp[closest_idxs[i]]}: {'{:.2f}'.format(distances[closest_idxs[i]])}"

        # plot central refence geometry and most similar (counterclockwise, from +x, blue)
        if plot_new_samples:
            plot_utils.plot_geometries(logger, new_mesh, closest_meshes,
                                       distance_values=distance_values,
                                       window_title=window_title,
                                       offset_margin=1.1)

        similarity_metrics_results_dict["experiments"].append(exp_result)

    # all new exps
    latent_new_all = np.vstack(latent_new_all)

    if save_latent_space:
        np.savetxt(os.path.join(pcas_folder, 'evaluation_geometries.out'), np.array(latent_new_all))

    # plot of all new experiments in latent space
    sdf_plot.plot_pca_projection(training_latent, latent_new_all,
                                 training_labels=training_meshes_exp, new_labels=evaluation_meshes_exp,
                                 cluster_labels=cluster_labels, new_cluster_labels=cluster_ids,
                                 annotate=True,
                                 title=f"All evaluations in training latent space\n(PCA1+PCA2 = {cvars[1]:>4.2e})",
                                 case=case, save_folder=images_folder)

    # save all similarity metrics to json
    gen_utils.write_json(similarity_metrics_results_dict, jsons_folder, f"similarity_metrics_results.json")

    logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                              9. Most isolated geometry
# ----------------------------------------------------------------------------------------------------------------------
if evaluate_most_isolated_geometry:
    logging.info(" 9. Most isolated geometry -------------------------------------------------------------------------")
    idx_largest_average, avg_dist, idx_maximin, maximin_dist = sdf_pca.find_most_isolated(D_training)

    logging.info(f"\tMax average index: {idx_largest_average}, Experiment: {training_meshes_exp[idx_largest_average]}")
    logging.info(f"\tMaximin index: {idx_maximin}, Experiment: {training_meshes_exp[idx_maximin]}")

    # plot geometry which is the most different, and its closest sibling
    if make_plot_most_isolated_geometry:
        if idx_largest_average == idx_maximin:
            window_title = f"Most isolated max average and maximin geometry: {training_meshes_exp[idx_largest_average]}"
        else:
            window_title = f"Most isolated max average geometry: {training_meshes_exp[idx_largest_average]}"

        # load them if need be
        if load_project:
            most_unique_largest_average, _, _ = geom_utils.load_mesh_wrapper(logger, dataset_json_file,
                                                                             full_source_path,
                                                                             training_meshes_exp[idx_largest_average],
                                                                             suppress_all=suppress_all_logs)
        else:
            most_unique_largest_average = training_meshes[idx_largest_average]

        # plot largest average or largest average and maximin
        plot_utils.plot_geometries(logger, most_unique_largest_average,
                                   window_title=window_title,
                                   offset_margin=0.1)

        if idx_largest_average != idx_maximin:
            if load_project:
                most_shared_minimax, _, _ = geom_utils.load_mesh_wrapper(logger, dataset_json_file, full_source_path,
                                                                         training_meshes_exp[idx_maximin],
                                                                         suppress_all=suppress_all_logs)
            else:
                most_shared_minimax = training_meshes[idx_maximin]

            plot_utils.plot_geometries(logger, most_shared_minimax, None,
                                       window_title=f"Most isolated maximin geometry: {training_meshes_exp[idx_maximin]}",
                                       offset_margin=0.1)
    logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                            10. Least isolated geometry
# ----------------------------------------------------------------------------------------------------------------------
if evaluate_least_isolated_geometry:
    logging.info(" 10. Least isolated geometry -----------------------------------------------------------------------")

    # Most representative (minimum average) and most central in the minimax sense
    medoid_idx, metoid_distance, minimax_idx, minimax_distance = sdf_pca.find_least_isolated(D_training)

    logging.info(f"\tMedoid index: {medoid_idx}, Experiment: {training_meshes_exp[medoid_idx]}")
    logging.info(f"\tMinimax index: {minimax_idx}, Experiment: {training_meshes_exp[minimax_idx]}")

    if make_plot_most_common_geometry:
        if medoid_idx == minimax_idx:
            window_title = f"Least isolated metoid and minimax geometry: {training_meshes_exp[medoid_idx]}"
        else:
            window_title = f"Least isolated metoid geometry: {training_meshes_exp[medoid_idx]}"

        if load_project:
            most_shared_metoid, _, _ = geom_utils.load_mesh_wrapper(logger, dataset_json_file, full_source_path,
                                                                    training_meshes_exp[medoid_idx],
                                                                    suppress_all=suppress_all_logs)
        else:
            most_shared_metoid = training_meshes[medoid_idx]

        # plot medoid or metoid and minimax geometry
        plot_utils.plot_geometries(logger, most_shared_metoid, None, window_title=window_title, offset_margin=0.1)

        if medoid_idx != minimax_idx:
            if load_project:
                most_shared_minimax, _, _ = geom_utils.load_mesh_wrapper(logger, dataset_json_file, full_source_path,
                                                                         training_meshes_exp[minimax_idx],
                                                                         suppress_all=suppress_all_logs)
            else:
                most_shared_minimax = training_meshes[minimax_idx]

            plot_utils.plot_geometries(logger, most_shared_minimax, None,
                                       window_title=f"Least isolated minimax geometry: {training_meshes_exp[minimax_idx]}",
                                       offset_margin=0.1)
    logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                            11. Visualize domain in PCA
# ----------------------------------------------------------------------------------------------------------------------
if make_plot_domain_scatter:
    logging.info(" 11. Domain in PCA ---------------------------------------------------------------------------------")
    sdf_plot.plot_doe_distribution_pca(logger, training_sdfs, training_pca, threshold=variance_threshold,
                                       highlight_experiments=evaluation_experiments_id, title_='geometry',
                                       case=case, save_folder=images_folder)
    logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                         12. SOM - Self Organizing Maps
# ----------------------------------------------------------------------------------------------------------------------
if evaluate_som:
    logging.info(" 12. SOM - Self Organizing Maps --------------------------------------------------------------------")
    som_grid = int(np.sqrt(training_latent.shape[0]))
    logging.info(f"\tMap size: {som_grid} x {som_grid}")
    som_points = sdf_pca.evaluate_SOM(training_latent, som_size=som_grid)
    plot_utils.plot_som_simple(som_points, som_size=som_grid, labels=training_meshes_exp,
                               case=case, save_folder=images_folder)
    logging.info("")

# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                               13. UMAP
#                                                  Uniform Manifold Approximation and Projection for Dimension Reduction
# ----------------------------------------------------------------------------------------------------------------------
if evaluate_umap:
    logging.info(" 13. UMAP ------------------------------------------------------------------------------------------")

    all_embedding_geoms, all_embedding_labels_geom = sdf_pca.build_multiple_UMAPs(training_latent)
    plot_utils.plot_overlapped_umap(all_embedding_geoms, labels=list_of_exp_sorted_,
                                    legend_labels=all_embedding_labels_geom,
                                    show_quivers=True, title=f"Geometry UMAP",
                                    case=case, save_folder=images_folder)

    n_neighbors_pca = max(5, training_latent.shape[0] // 10)
    logging.info(f"\tn_neighbors_pca: {n_neighbors_pca}")
    umap_n_neightbours = [n_neighbors_pca - 2, n_neighbors_pca, n_neighbors_pca + 2]
    training_latent = training_pca.fit_transform(training_sdfs)
    for i_n in umap_n_neightbours:
        embedding_pca, reducer_pca, scaler_pca = sdf_pca.evaluate_UMAP(training_latent, n_neighbors=i_n,
                                                                       min_dist=0.1, n_components=2,
                                                                       seed=standard_seed)

        # loop on new meshes
        new_sdfs_flat = []
        for i_new_mesh, new_mesh_exp in enumerate(evaluation_meshes_exp):
            # new mesh
            if load_project:
                new_mesh, new_col, new_mesh_aabb = geom_utils.load_mesh_wrapper(logger, dataset_json_file,
                                                                                full_source_path,
                                                                                new_mesh_exp,
                                                                                suppress_all=suppress_all_logs)
            else:
                new_mesh = evaluation_meshes[i_new_mesh]

            # compute SDF of the new mesh
            new_sdf = sdf_pca.compute_sdf_from_mesh(logger, new_mesh, grid_points)  # shape (grid_size^3,)

            # apply weight to focus on geometry
            new_sdf = sdf_pca.apply_weight([new_sdf], sdf_alpha_weight)

            # flatten if needed
            new_sdfs_flat.append(new_sdf.flatten())

        embedding_pca_evaluation = None
        new_labels = None
        if evaluation_meshes_exp:
            # convert to array
            new_sdfs_flat = np.array(new_sdfs_flat)

            # project into PCA space and scale
            latent_new = training_pca.transform(new_sdfs_flat)  # shape (1, n_components)
            latent_new_scaled = scaler_pca.transform(latent_new)

            # pca space to umap
            embedding_pca_evaluation = reducer_pca.transform(latent_new_scaled)
            new_labels = evaluation_meshes_exp

        plot_utils.plot_umap(embedding_pca, new_embedding=embedding_pca_evaluation,
                             labels=training_meshes_exp, new_labels=new_labels,
                             title=f"Geometry UMAP from PCA - n_neighbors {i_n}",
                             case=case, save_folder=images_folder)

    logging.info("")


# ----------------------------------------------------------------------------------------------------------------------
#                                                                                             14. Propose new geometries
# ----------------------------------------------------------------------------------------------------------------------
if generate_new_samples and \
        (make_plot_new_maximin or make_plot_new_gaussian or make_plot_new_multigaussian or make_plot_new_interpolate):
    logging.info(" 15. Proposing new geometries ----------------------------------------------------------------------")

    updated_all_latent = training_latent.copy()

    # maximin sampling - works terribly
    if make_plot_new_maximin:
        logging.info("\tMaximin Sampling")
        for i in range(generate_new_samples):
            new_design, dist_from_existing = sdf_pca.simplified_maximim_sampling(updated_all_latent,
                                                                                 bounds_=bounds_training_latent)
            logging.info(f"\t\tDesign number {i}\nvalues {new_design}, distance from existing: {dist_from_existing}")
            sdf_pca.reconstruct_mesh(logger, training_pca, new_design, None, n_grids_x, n_grids_y, n_grids_z,
                                     None, None, grid_size, make_plot_=True,
                                     title_=f'Maximin sampling')
            updated_all_latent = np.vstack([updated_all_latent, new_design])

    # small gaussian perturbations - somehow better
    if make_plot_new_gaussian:
        sigma = 0.3
        logging.info(f"\tGaussian perturbation at sigma {sigma}:")
        for i in range(generate_new_samples):
            new_design = training_latent[i] + sigma * np.random.randn(training_latent.shape[1])
            logging.info(f"\t\tDesign number {i}\n\tvalues: {new_design}, \n\texisting: {training_latent[i]}")
            sdf_pca.reconstruct_mesh(logger, training_pca, new_design, None, n_grids_x, n_grids_y, n_grids_z,
                                     None, None, grid_size, make_plot_=True,
                                     title_=f'Gaussian perturbation from source geometry {training_meshes_exp[i]}')

    # multivariate gaussian sampling
    if make_plot_new_multigaussian:
        logging.info(f"\tMultivariate Gaussian sampling:")
        mean = training_latent.mean(axis=0)
        cov = np.cov(training_latent.T)
        for i in range(generate_new_samples):
            new_design = np.random.multivariate_normal(mean, cov)
            logging.info(f"\t\tDesign number {i}\n\tvalues: {new_design}")
            sdf_pca.reconstruct_mesh(logger, training_pca, new_design, None, n_grids_x, n_grids_y, n_grids_z,
                                     None, None, grid_size, make_plot_=True,
                                     title_=f"Multivariate normal, test {i}")

    # interpolate reasonably close designs
    if make_plot_new_interpolate:
        # distance between experiments in training set, D_eval.shape = (n_train, n_train)
        D_training = pairwise_distances(training_latent, metric='euclidean')
        D_training_sorted, labels_training_sorted = sdf_pca.distance_matrix_sorting(
            sorting_type=sorting_logic,
            latent_space=training_latent,
            D=D_training,
            labels=training_meshes_exp)

        logging.info(f"\tInterpolation sampling:")
        for i in range(generate_new_samples):
            result = sdf_pca.pick_random_and_neighbors(D_training_sorted, labels_training_sorted)
            logging.info(f"\tSelected random geometry: {result['selected_label']}")

            logging.info(f"\t\tClosest neighbors:")

            for n in result["neighbors"]:
                logging.info(f"\t\t\t{n['label']}, distance: {n['distance']}")

            neighbor_labels = [n["label"] for n in result["neighbors"]]
            original_indices = [labels_all.index(lbl) for lbl in neighbor_labels]

            if len(original_indices) > 1:
                new_design = 0.5 * training_latent[original_indices[0]] + 0.5 * training_latent[
                    original_indices[1]]
                sdf_pca.reconstruct_mesh(logger, training_pca, new_design, None, n_grids_x, n_grids_y, n_grids_z,
                                         None, None, grid_size, make_plot_=True,
                                         title_=f"Average of geometries {labels_all[original_indices[0]]} and "
                                                f"{labels_all[original_indices[1]]}")
