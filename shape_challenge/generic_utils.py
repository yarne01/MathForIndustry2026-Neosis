import json
import os
import shutil
import copy
import sys
import struct
import re

import numpy as np


# ---------------------------------------------------------------------------------------------------------------- FILES
def is_file_extension_INP(filename):
    """
    Check file extension, expected: .inp
    :param filename: str, name of the file, with extension
    :return: bool, True or False
    """
    return filename[len(filename) - 4:].lower() == ".inp"


def is_file_extension_STL(filename):
    """
    Check file extension, expected: .stl
    :param filename: str, name of the file, with extension
    :return: bool, True or False
    """
    return filename[len(filename) - 4:].lower() == ".stl"


def is_file_extension_OBJ(filename):
    """
    Check file extension, expected: .obj
    :param filename: str, name of the file, with extension
    :return: bool, True or False
    """
    return filename[len(filename) - 4:].lower() == ".obj"


def is_file_extension_CSV(filename):
    """
    Check file extension, expected: .csv
    :param filename: str, name of the file, with extension
    :return: bool, True or False
    """
    return filename[len(filename) - 4:].lower() == ".csv"


# generalised file extension check, number of digits does not include the dot
# i.e. extension='txt', n_digits=3
def is_file_extension_XXX(filename, extension, n_digits=3):
    """
    Check file extension, expected: .any
    :param filename: str, name of the file, with extension
    :param extension: str, type of file to be checked for
    :param n_digits: int, number of characters to be checked (. NOT included)
    :return: bool, True or False
    """
    return filename[len(filename) - (1+n_digits):].lower() == "."+extension


def read_noesis_binary(folder, filename):
    """
    Read a Noesis binary file containing double-precision floating-point data.

    The function reads the file header to determine the byte order of the
    stored data and returns the numerical values as a tuple of doubles.

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory containing the Noesis binary file.

    filename : str
        File name without the ``.noesis_bin`` extension.

    Returns
    -------
    tuple of float
        Flat tuple containing the double-precision values stored in the file.

    Notes
    -----
    - The function expects the file to begin with a 4-byte integer header
      (byte-order mark).
    - If the header value is ``67``, the data are interpreted as
      little-endian doubles.
    - Otherwise, the data are interpreted as big-endian doubles.
    - The returned data are not reshaped. If the file represents a matrix,
      reshaping must be performed by the caller using the appropriate
      dimensions.
    """

    if not filename.endswith(".noesis_bin"):
        filename += ".noesis_bin"

    with open(os.path.join(folder, filename), mode="rb") as zip_file:
        contents = zip_file.read()
        size_int = 4
        bom = struct.unpack("i", contents[:size_int])

        if bom[0] == 67:
            data = struct.unpack("d" * ((len(contents) - size_int) // 8), contents[size_int:])
        else:
            data = struct.unpack(">"+"d" * ((len(contents) - size_int) // 8), contents[size_int:])
        return data


def write_noesis_binary(data, folder, filename):
    """
    Write an array to a Noesis binary file.

    The function creates a Noesis binary file containing a 4-byte header
    followed by the data stored as double-precision floating-point values.

    Parameters
    ----------
    data : array_like
        Numerical data to be written. The data are converted to
        ``numpy.double`` before being written to disk.

    folder : str or pathlib.Path
        Directory where the output file will be created.

    filename : str
        Output file name. The ``.noesis_bin`` extension is appended
        automatically if not already present.

    Returns
    -------
    None

    Notes
    -----
    - The file begins with the unsigned 32-bit integer value ``67``,
      written in little-endian byte order, which serves as the Noesis
      binary file header.
    - The data are written immediately after the header in binary format
      using ``numpy.ndarray.tofile()``.
    - Existing files with the same name are overwritten.
    """

    if not filename.endswith(".noesis_bin"):
        filename += ".noesis_bin"

    path = os.path.join(folder, filename)

    with open(path, "wb") as file:
        file.write((67).to_bytes(4, byteorder="little", signed=False))

    with open(path, "ab") as file:
        np.asarray(data, dtype=np.double).tofile(file)


def read_txt(base_dir_, file_, separator_=',', type_=float):
    """
    Read a delimited text file into a nested Python list.

    The function parses the file line by line, splits each line using the
    specified separator, converts each non-empty field to the requested data
    type, and returns the resulting data as a list of lists.

    Parameters
    ----------
    base_dir_ : str or pathlib.Path
        Directory containing the text file.

    file_ : str
        File name. The ``.txt`` extension is appended automatically if not
        already present.

    separator_ : str, optional
        Field delimiter used in the text file. Default is ``,``, making the
        function suitable for comma-separated files.

    type_ : {float, int, str}, optional
        Data type used to convert each non-empty field. Default is ``float``.

    Returns
    -------
    list of list
        Nested list containing the contents of the file. Each inner list
        corresponds to one row of the input file.

    Notes
    -----
    - Empty fields are ignored and are not included in the returned rows.
    - Only ``float``, ``int``, and ``str`` conversions are currently
      supported.
    - The function performs no consistency checks on the number of columns
      per row.
    """

    if '.txt' not in file_:
        file_ += '.txt'

    with open(os.path.join(base_dir_, file_), "r") as f:
        lines = f.readlines()

    tmp = []
    for line in lines:
        vals = line.rstrip().split(separator_)
        assemble_line = []
        for val in vals:
            if val != '':
                if type_ == float:
                    assemble_line.append(float(val))
                elif type_ == int:
                    assemble_line.append(int(val))
                elif type_ == str:
                    assemble_line.append(str(val))
        tmp.append(assemble_line)
    return tmp


def read_file_as_extension(logger, path_, file_, extension_):
    """
    Wrapper to hadle reading .txt and .noesis_bin files
    :param logger: logger object
    :param path_: str, path to host folder
    :param file_: str, name of the target file
    :param extension_: str, extension of the file, dot included
    :return: numpy array with read data
    """
    if extension_ == ".noesis_bin":
        return read_noesis_binary(os.path.join(path_), file_)
    elif extension_ == ".txt":
        tmp_data = read_generic_file(os.path.join(path_, file_+extension_))
        data = []
        for entry in tmp_data:
            data.append(float(entry.rstrip().replace('\n', '')))
        return data
    else:
        logger.critical(f"{extension_} files still to be implemented")
        sys.exit(1)


def read_json(path):
    """
    Read json given path and name
    :param path: absolute path and filename
    :return: dict from parsed file
    """
    with open(path, 'r') as f:
        return json.load(f)


def write_json(data, folder, filename):
    """
    Write dictionary as json
    :param data: dictionary object to be saved
    :param folder: absolute path till target folder
    :param filename: name of the file to be created. extension not required.
    :return: None
    """
    if '.json' not in filename:
        filename += '.json'

    with open(os.path.join(folder, filename), 'w') as outfile:
        json.dump(data, outfile, indent=4)


# returns lines in file
def read_generic_file(full_path_name):
    """
    Simple helper to read from a file
    :param full_path_name: str, name and path to the file
    :return: array with read lines
    """
    f_in = open(full_path_name, "r")
    lines_in = f_in.readlines()
    f_in.close()
    return lines_in


def find_dataset_json(full_source_path_):
    """
    Look for a dataset configuration JSON file inside full_source_path_.

    Priority:
    1) config.json
    2) database_configuration.json

    Returns
    -------
    dataset_json : str
        Full path to the JSON file that exists.

    Raises
    ------
    FileNotFoundError
        If neither file is found.
    """

    candidates_ = ["config.json", "database_configuration.json"]

    for name in candidates_:
        path = os.path.join(full_source_path_, name)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(f"No dataset configuration JSON found in {full_source_path_}. Tried: {', '.join(candidates_)}")


# -------------------------------------------------------------------------------------------------------------- FOLDERS
def prepare_folder(path, delete_if_exists=True):
    """
    Remove the target folder if exists and creates it (anyway).

    Parameters
    ----------
    path: str
        Full path and name of the folder.
    delete_if_exists: bool
        Delete folder if already there. Default True
    Returns
    -------
    None
    """

    if not os.path.exists(path):
        # Folder does not exist, create it
        os.makedirs(path)
    else:
        if delete_if_exists:
            # Folder exists clear its content
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)   # remove file or symlink
                else:
                    shutil.rmtree(item_path)  # remove folder


# ---------------------------------------------------------------------------------------------------------------- JSONS
# return name, type and file extension
def config_header_info(jsonobj_):
    """
    Extract the main header information from a configuration dictionary.

    Parameters
    ----------
    jsonobj_ : dict
        Configuration dictionary parsed from a JSON file. The dictionary
        must contain a ``"header"`` section with the keys ``"name"``,
        ``"type"``, and ``"file_extension"``.

    Returns
    -------
    tuple of str
        Tuple containing:

        - ``name`` : dataset or model name.
        - ``type`` : data type or category.
        - ``file_extension`` : expected file extension associated with the data.

    Raises
    ------
    KeyError
        If the required ``"header"`` section or one of the expected keys
        is missing.
    """
    header = jsonobj_['header']
    return header['name'], header['type'], header['file_extension']


# vartype_ is 'inputs' or 'outputs'
def get_variable_names(jsonobj_, vartype_):
    """
    Return the names of the variables belonging to a given category.

    Parameters
    ----------
    jsonobj_ : dict
        Configuration dictionary parsed from a JSON file. The dictionary
        must contain a ``"variables"`` section with entries such as
        ``"inputs"`` and/or ``"outputs"``.

    vartype_ : str
        Variable category to retrieve. Typically, either ``"inputs"`` or
        ``"outputs"``.

    Returns
    -------
    list of str
        List containing the names of all variables belonging to the
        requested category, in the same order as they appear in the
        configuration.

    Raises
    ------
    KeyError
        If the ``"variables"`` section or the requested variable category
        is not present in the configuration.
    """
    vars_ = jsonobj_['variables'][vartype_]
    return [var for var in vars_]


def get_field(jsonobj_, vartype_, var_, field_):
    """
    Retrieve a specific field associated with a variable in the configuration.

    Parameters
    ----------
    jsonobj_ : dict
        Configuration dictionary parsed from a JSON file. The dictionary
        must contain a ``"variables"`` section organized by variable type.

    vartype_ : str
        Variable category containing the requested variable. Typically,
        either ``"inputs"`` or ``"outputs"``.

    var_ : str
        Name of the variable.

    field_ : str
        Name of the field to retrieve for the selected variable
        (e.g. ``"min"``, ``"max"``, ``"unit"``, ``"description"``,
        depending on the configuration format).

    Returns
    -------
    object
        Value associated with the requested field. The returned type
        depends on the field stored in the configuration.

    Raises
    ------
    KeyError
        If the variable category, variable name, or requested field
        does not exist in the configuration.
    """
    return jsonobj_['variables'][vartype_][var_][field_]


def get_max_and_min_from_attributes(attributes_json_obj_, vartype_, varname_):
    return attributes_json_obj_[vartype_][varname_]['min'][0], attributes_json_obj_[vartype_][varname_]['max'][0]


def initialize_variables_attributes(
        *,
        variables_definitions) -> dict:

    variables_attributes = copy.deepcopy(variables_definitions)

    input_variables_names = list(variables_definitions["inputs"].keys())
    variables_attributes["inputs"] = dict(
        zip(
            input_variables_names,
            ({"min": [sys.float_info.max] * variables_definitions["inputs"][var_name]["Ncomp"],
              "max": [-1.*sys.float_info.max] * variables_definitions["inputs"][var_name]["Ncomp"]} for var_name in input_variables_names)))

    output_variables_names = list(variables_definitions["outputs"].keys())
    variables_attributes["outputs"] = dict(
        zip(
            output_variables_names,
            ({"min": [sys.float_info.max] * variables_definitions["outputs"][var_name]["Ncomp"],
              "max": [-1.*sys.float_info.max] * variables_definitions["outputs"][var_name]["Ncomp"]} for var_name in output_variables_names)))

    coordinate_variables_names = list(variables_definitions["coordinates"].keys())
    variables_attributes["coordinates"] = dict(
        zip(
            coordinate_variables_names,
            ({"min": [sys.float_info.max] * variables_definitions["coordinates"][var_name]["Ncomp"],
              "max": [-1.*sys.float_info.max] * variables_definitions["coordinates"][var_name]["Ncomp"]} for var_name in coordinate_variables_names)))

    return variables_attributes


def generate_variables_attributes_file(*, attributes_dict: dict, attributes_filepath: str):
    json_file = open(attributes_filepath, "w")
    json.dump(attributes_dict, json_file, indent=4)
    json_file.close()


def update_variables_attributes(*, variables_attributes: dict, variable_type: str, variable_name: str, i_comp: int, current_exp_data: np.array):
    min_in_current_exp_data = np.amin(current_exp_data)
    max_in_current_exp_data = np.amax(current_exp_data)

    current_min = variables_attributes[variable_type][variable_name]["min"][i_comp]
    current_max = variables_attributes[variable_type][variable_name]["max"][i_comp]

    variables_attributes[variable_type][variable_name]["min"][i_comp] = min_in_current_exp_data if min_in_current_exp_data < current_min else current_min
    variables_attributes[variable_type][variable_name]["max"][i_comp] = max_in_current_exp_data if max_in_current_exp_data > current_max else current_max


# -------------------------------------------------------------------------------------------------------------- STRINGS
def clean_plot_title(title):
    """
    Convert an arbitrary plot title into a filename-safe string.

    Parameters
    ----------
    title : str
        Plot title.

    Returns
    -------
    str
        Filename-safe version of the title.
    """

    # Keep only the first line
    title = title.splitlines()[0]

    # Replace invalid filename characters with '_'
    title = re.sub(r'[^A-Za-z0-9_.-]+', '_', title)

    # Collapse multiple underscores
    title = re.sub(r'_+', '_', title)

    # Remove leading/trailing underscores and dots
    title = title.strip("._")

    return title
