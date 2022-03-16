""" Python Wrapper for CloudCompare CLI """
import subprocess
import sys
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import List, Any
import logging

__version__ = "0.2"
__author__ = "dwoiwode"
__license__ = "MIT"

_platform = sys.platform
if _platform.startswith("win32"):
    DEFAULT_EXECUTABLE = r"C:\Program Files\CloudCompare\CloudCompare.exe"
elif _platform.startswith("linux"):
    import warnings

    warnings.warn("Default executable for linux not set")
    DEFAULT_EXECUTABLE = "CloudCompare"  # TODO: Update default executable for linux
elif _platform.startswith("darwin"):
    import warnings

    warnings.warn("Default executable for macOS not set")
    DEFAULT_EXECUTABLE = "CloudCompare"  # TODO: Update default executable for macOS

_FLAG_SILENT = "-SILENT"
_logger = logging.getLogger("pyCloudCompare")


class FEATURES(Enum):
    SUM_OF_EIGENVALUES = "SUM_OF_EIGENVALUES"
    OMNIVARIANCE = "OMNIVARIANCE"
    EIGENTROPY = "EIGENTROPY"
    ANISOTROPY = "ANISOTROPY"
    PLANARITY = "PLANARITY"
    LINEARITY = "LINEARITY"
    PCA1 = "PCA1"
    PCA2 = "PCA2"
    SURFACE_VARIATION = "SURFACE_VARIATION"
    SPHERICITY = "SPHERICITY"
    VERTICALITY = "VERTICALITY"
    EIGENVALUE1 = "EIGENVALUE1"
    EIGENVALUE2 = "EIGENVALUE2"
    EIGENVALUE3 = "EIGENVALUE3"


class OCTREE_ORIENT(Enum):
    PLUS_ZERO = "PLUS_ZERO"
    MINUS_ZERO = "MINUS_ZERO"
    PLUS_BARYCENTER = "PLUS_BARYCENTER"
    MINUS_BARYCENTER = "MINUS_BARYCENTER"
    PLUS_X = "PLUS_X"
    MINUS_X = "MINUS_X"
    PLUS_Y = "PLUS_Y"
    MINUS_Y = "MINUS_Y"
    PLUS_Z = "PLUS_Z"
    MINUS_Z = "MINUS_Z"
    PREVIOUS = "PREVIOUS"


class OCTREE_MODEL(Enum):
    LS = "LS"
    TRI = "TRI"
    QUADRIC = "QUADRIC"


class SS_ALGORITHM(Enum):
    RANDOM = "RANDOM"
    SPATIAL = "SPATIAL"
    OCTREE = "OCTREE"


class SAMPLE_METHOD(Enum):
    POINTS = "POINTS"
    DENSITY = "DENSITY"


class DENSITY_TYPE(Enum):
    KNN = "KNN"
    SURFACE = "SURFACE"
    VOLUME = "VOLUME"


class CURV_TYPE(Enum):
    MEAN = "MEAN"
    GAUSS = "GAUSS"
    NORMAL_CHANGE = "NORMAL_CHANGE"


class ROT(Enum):
    XYZ = "XYZ"
    X = "X"
    Y = "Y"
    Z = "Z"
    NONE = "NONE"


class SF_ARITHMETICS(Enum):
    SQRT = "sqrt"
    POW2 = "pow2"
    POW3 = "pow3"
    EXP = "exp"
    LOG = "log"
    LOG10 = "log10"
    COS = "cos"
    SIN = "sin"
    TAN = "tan"
    ACOS = "acos"
    ASIN = "asin"
    ATAN = "atan"
    INT = "int"
    INVERSE = "inverse"


class SF_OPERATIONS(Enum):
    ADD = "add"
    SUBTRACT = "sub"
    MULTIPLY = "mult"
    DIVIDE = "div"


class SEPARATOR(Enum):
    SPACE = "SPACE"
    SEMICOLON = "SEMICOLON"
    COMMA = "COMMA"
    TAB = "TAB"

    @classmethod
    def from_string(cls, s):
        if s == " ":
            return SEPARATOR.SPACE
        if s == ";":
            return SEPARATOR.SEMICOLON
        if s == ",":
            return SEPARATOR.COMMA
        if s == "\t":
            return SEPARATOR.TAB
        _logger.critical(f"Converting string to separator: Invalid character (Has to be from ' ;,\\t'. Char is: '{s}'")
        raise ValueError(f"Invalid separator (Has to be ' ;,\\t'. Is: '{s}'")


class CLOUD_EXPORT_FORMAT(Enum):
    ASCII = "ASC"
    BIN = "BIN"
    PLY = "PLY"
    LAS = "LAS"
    E57 = "E57"
    VTK = "VTK"
    PCD = "PCD"
    SOI = "SOI"
    PN = "PN"
    PV = "PV"


class MESH_EXPORT_FORMAT(Enum):
    BIN = "BIN"
    OBJ = "OBJ"
    PLY = "PLY"
    STL = "STL"
    VTK = "VTK"
    MA = "MA"
    FBX = "FBX"


class FBX_EXPORT_FORMAT(Enum):
    BINARY = "FBX_binary"
    ASCII = "FBX_ascii"
    ENCRYPTED = "FBX_encrypted"
    BINARY_6_0 = "FBX_6.0_binary"
    ASCII_6_0 = "FBX_6.0_ascii"
    ENCRYPTED_6_0 = "FBX_6.0_encrypted"


class PLY_EXPORT_FORMAT(Enum):
    ASCII = "ASCII"
    BINARY_BIG_ENDIAN = "BINARY_BE"
    BINARY_LITTLE_ENDIAN = "BINARY_LE"


class RANSAC_PRIMITIVES(Enum):
    PLANE = "PLANE"
    SPHERE = "SPHERE"
    CYCLINDER = "CYLINDER"
    CONE = "CONE"
    TORUS = "TORUS"


class BOOL(Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"

    @classmethod
    def from_bool(cls, value):
        if value:
            return BOOL.TRUE
        return BOOL.FALSE


class ONOFF(Enum):
    ON = "ON"
    OFF = "OFF"

    @classmethod
    def from_bool(cls, value):
        if value:
            return ONOFF.ON
        return ONOFF.OFF


def cc(flag=None):
    def wrapper1(func):
        @wraps(func)
        def wrapper2(self: "CloudCompareCommand", *args, **kwargs):
            func_name = func.__name__
            _logger.debug(f"Add {func_name} to command")
            old_arguments = list(self.arguments)
            try:
                if flag is not None:
                    self.arguments.append(flag)
                func(self, *args, **kwargs)
                self._validate_command()
                n = len(old_arguments)
                self.commands.append(CCCommand(func_name, self.arguments[n:]))
                _logger.debug(f"New arguments: {self.arguments[n:]}")
            except Exception as e:
                _logger.error(f"Failed to add {func_name}! Cause: {str(e)}. Rolling back")
                self.arguments = old_arguments

        return wrapper2

    return wrapper1


class CCCommand:
    def __init__(self, function_name: str, arguments: List[Any]):
        self.functionName = function_name
        self.arguments = arguments

    def __repr__(self):
        return f"CCCommand({self.functionName}, {self.arguments})"


class CloudCompareCLI:
    def __init__(self, executable=DEFAULT_EXECUTABLE):
        self.exec = Path(executable).absolute()
        if str(self.exec) not in sys.path:
            sys.path.append(str(self.exec.parent))

    def __repr__(self):
        return f"CloudCompareCLI({self.exec})"

    def new_command(self):
        return CloudCompareCommand(self)


class CloudCompareCommand:
    def __init__(self, cc_cli: CloudCompareCLI, arguments: List[Any] = None):
        _logger.debug("Initializing new command")
        self.cc_cli = cc_cli
        self.arguments = arguments or []
        self.commands: List[CCCommand] = []

    def __repr__(self):
        return f"CloudCompareCMD({self.cc_cli}, {self.arguments})"

    def __str__(self):
        return self.to_cmd()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.execute()

    def to_cmd(self):
        arguments_copy = list(self.arguments)
        is_silent = False
        while _FLAG_SILENT in arguments_copy:
            is_silent = True
            arguments_copy.remove(_FLAG_SILENT)
        if is_silent:
            arguments_copy.insert(0, _FLAG_SILENT)
        args = " ".join([arg.value if isinstance(arg, Enum) else str(arg) for arg in arguments_copy])
        return f'"{str(self.cc_cli.exec)}" {args}'

    def execute(self):
        _logger.debug(f"Executing command: {str(self)}")
        proc = subprocess.run(self.to_cmd(), stdout=subprocess.PIPE)
        ret = proc.returncode
        if ret != 0:
            _logger.error(proc.stdout.decode("utf-8"))
            raise RuntimeWarning(f"Non-Zero Returncode ({ret})")
        return ret

    def _validate_command(self):
        self.to_cmd()

    # =================== Commands ===================
    @cc()
    def silent(self, is_silent=True):
        if is_silent:
            if _FLAG_SILENT not in self.arguments:
                self.arguments.append(_FLAG_SILENT)
        else:
            while _FLAG_SILENT in self.arguments:
                self.arguments.remove(_FLAG_SILENT)

    @cc("-O")
    def open(self,
             filepath: str or Path,
             skip=None,
             global_shift=None):
        self.arguments.append(f"\"{filepath}\"")
        if skip is not None:
            self.arguments.append("-SKIP")
            self.arguments.append(skip)
        if global_shift is not None:
            self.arguments.append("-GLOBAL_SHIFT")
            if isinstance(global_shift, str):
                assert global_shift == "AUTO" or global_shift == "FIRST"
            elif isinstance(global_shift, (tuple, list)):
                x, y, z = global_shift
                global_shift = f"{x} {y} {z}"
            self.arguments.append(global_shift)

    @cc("-MOMENT")
    def moment(self, radius: float):
        self.arguments.append(radius)

    @cc("-FEATURE")
    def feature(self,
                feature: FEATURES,
                radius: float):
        self.arguments += [feature, radius]

    @cc("-OCTREE_NORMALS")
    def octree_normals(self,
                       radius: float,
                       orient: OCTREE_ORIENT = None,
                       model: OCTREE_MODEL = None):
        self.arguments.append(radius)
        if orient is not None:
            self.arguments.append("-ORIENT")
            self.arguments.append(orient)
        if model is not None:
            self.arguments.append("-MODEL")
            self.arguments.append(model)

    @cc("-COMPUTE_NORMALS")
    def compute_normals(self):
        pass

    @cc("-NORMALS_TO_SFS")
    def normals_to_sfs(self):
        pass

    @cc("-NORMALS_TO_DIP")
    def normals_to_dip(self):
        pass

    @cc("-CLEAR_NORMALS")
    def clear_normals(self):
        pass

    @cc("-ORIENT_NORMS_MST")
    def orient_norms_mst(self, number_of_neighbors: int):
        self.arguments.append(number_of_neighbors)

    @cc("-MERGE_CLOUDS")
    def merge_clouds(self):
        pass

    @cc("-MERGE_MESHES")
    def merge_meshes(self):
        pass

    @cc("-SS")
    def subsampling(self,
                    algorithm: SS_ALGORITHM,
                    parameter):
        self.arguments.append(algorithm)
        self.arguments.append(parameter)

    @cc("-EXTRACT_CC")
    def extract_cc(self,
                   octree_level,
                   min_points):
        assert 1 <= octree_level <= 21
        self.arguments += [octree_level, min_points]

    @cc("-SAMPLE_MESH")
    def sample_mesh(self,
                    method: SAMPLE_METHOD,
                    parameter):
        self.arguments += [method, parameter]

    @cc("-EXTRACT_VERTICES")
    def extract_vertices(self):
        pass

    @cc("-C2C_DIST")
    def c2c_dist(self,
                 split_xyz=None,
                 max_dist=None,
                 octree_level=None,
                 model=None,
                 max_t_count=None):
        if split_xyz is not None:
            self.arguments += ["-SPLIT_XYZ", split_xyz]
        if max_dist is not None:
            self.arguments += ["-MAX_DIST", max_dist]
        if octree_level is not None:
            self.arguments += ["-OCTREE_LEVEL", octree_level]
        if model is not None:
            # TODO: Improve model input
            self.arguments += ["-MODEL", model]
        if max_t_count is not None:
            self.arguments += ["-MAX_TCOUNT", max_t_count]

    @cc("-C2M_DIST")
    def c2m_dist(self,
                 flip_norms: bool = False,
                 max_dist=None,
                 octree_level=None,
                 max_t_count=None):
        if flip_norms:
            self.arguments.append(flip_norms)
        if max_dist is not None:
            self.arguments += ["-MAX_DIST", max_dist]
        if octree_level is not None:
            self.arguments += ["-OCTREE_LEVEL", octree_level]
        if max_t_count is not None:
            self.arguments += ["-MAX_TCOUNT", max_t_count]

    @cc("-RASTERIZE")
    def rasterize(self,
                  grid_step,
                  vert_dir=None,
                  proj=None,
                  sf_proj=None,
                  empty_fill=None,
                  custom_height=None,
                  output_cloud=False,
                  output_mesh=False,
                  output_raster_z=False,
                  output_raster_rgb=False):
        self.arguments += ["-GRID_STEP", grid_step]
        if vert_dir is not None:
            self.arguments += ["-VERT_DIR", vert_dir]
        if proj is not None:
            self.arguments += ["-PROJ", proj]
        if sf_proj is not None:
            self.arguments += ["-SF_PROJ", sf_proj]
        if empty_fill is not None:
            self.arguments += ["-EMPTY_FILL", empty_fill]
        if custom_height is not None:
            self.arguments += ["-CUSTOM_HEIGHT", custom_height]
        if output_cloud:
            self.arguments.append("-OUTPUT_CLOUD")
        if output_mesh:
            self.arguments.append("-OUTPUT_MESH")
        if output_raster_z:
            self.arguments.append("-OUTPUT_RASTER_Z")
        if output_raster_rgb:
            self.arguments.append("-OUTPUT_RASTER_RGB")

    @cc("-VOLUME")
    def volume(self,
               grid_step,
               vert_dir=None,
               const_height=None,
               ground_is_first=False,
               output_mesh=False):
        self.arguments += ["-GRID_STEP", grid_step]
        if vert_dir is not None:
            self.arguments += ["-VERT_DIR", vert_dir]
        if const_height is not None:
            self.arguments += ["-CONST_HEIGHT", const_height]
        if ground_is_first:
            self.arguments.append("-GROUND_IS_FIRST")
        if output_mesh:
            self.arguments.append("-OUTPUT_MESH")

    @cc("-STAT_TEST")
    def stat_test(self,
                  distribution,
                  distribution_parameter,
                  p_value,
                  neighbor_count):
        self.arguments += [distribution, distribution_parameter, p_value, neighbor_count]

    @cc("-COORD_TO_SF")
    def coord_to_sf(self, dimension):
        assert dimension in ["X", "Y", "Z"]
        self.arguments.append(dimension)

    @cc("-FILTER_SF")
    def filter_sf(self,
                  min_val,
                  max_val):
        special_words = ["MIN", "DISP_MIN", "SAT_MIN", "MAX", "DISP_MAX", "SAT_MAX"]
        if isinstance(min_val, str):
            assert min_val in special_words
        if isinstance(max_val, str):
            assert max_val in special_words
        self.arguments += [min_val, max_val]

    @cc("-DENSITY")
    def density(self,
                sphere_radius,
                type_: DENSITY_TYPE = None):
        self.arguments.append(sphere_radius)
        if type_ is not None:
            self.arguments += ["-TYPE", type_]

    @cc("-APPROX_DENSITY")
    def approx_density(self, type_: DENSITY_TYPE = None):
        if type_ is not None:
            self.arguments += ["-TYPE", type_]

    @cc("-ROUGH")
    def rough(self, kernel_size):
        self.arguments.append(kernel_size)

    @cc("-CURV")
    def curvature(self,
                  type_: CURV_TYPE,
                  kernel_size):
        self.arguments += [type_, kernel_size]

    @cc("-SF_GRAD")
    def sf_grad(self, euclidian: bool):
        euclidian = BOOL.from_bool(euclidian)
        self.arguments.append(euclidian)

    @cc("-BEST_FIT_PLANE")
    def best_fit_plane(self,
                       make_horiz=False,
                       keep_loaded=False):
        if make_horiz:
            self.arguments.append("-MAKE_HORIZ")
        if keep_loaded:
            self.arguments.append("-KEEP_LOADED")

    @cc("-APPLY_TRANS")
    def apply_transformation(self, filename: str or Path):
        self.arguments.append(filename)

    @cc("-MATCH_CENTERS")
    def match_centers(self):
        pass

    @cc("-DELAUNAY")
    def delaunay(self,
                 xy_plane=False,
                 best_fit=False,
                 max_edge_length=None):
        if xy_plane:
            self.arguments.append("-AA")
        if best_fit:
            self.arguments.append("-BEST_FIT")
        if max_edge_length is not None:
            self.arguments += ["-MAX_EDGE_LENGTH", max_edge_length]

    @cc("-ICP")
    def icp(self,
            reference_is_first=False,
            min_error_diff: float = 1e-6,
            iter_: int = None,
            overlap: int = 100,
            adjust_scale=False,
            random_sampling_limit=20000,
            farthest_removal=False,
            rot: ROT = None):
        if reference_is_first:
            self.arguments.append("-REFERENCE_IS_FIRST")
        if min_error_diff != 1e-6:
            self.arguments += ["-MIN_ERROR_DIFF", min_error_diff]
        if iter_ is not None:
            self.arguments += ["-ITER", iter_]
        if overlap != 100:
            self.arguments += ["-OVERLAP", overlap]
        if adjust_scale:
            self.arguments.append("-ADJUST_SCALE")
        if random_sampling_limit != 20000:
            self.arguments += ["-RANDOM_SAMPLING_LIMIT", random_sampling_limit]
        if farthest_removal:
            self.arguments.append("-FARTHEST_REMOVAL")
        # TODO: DATA_SF_WEIGHTS
        # TODO: MODEL_SF_WEIGHTS
        if rot is not None:
            self.arguments.append("-ROT" + rot.value)

    @cc("-CROP")
    def crop(self,
             x_min, y_min, z_min,
             x_max, y_max, z_max,
             outside=False):
        box_string = f"{x_min}:{y_min}:{z_min}:{x_max}:{y_max}:{z_max}"
        self.arguments.append(box_string)
        if outside:
            self.arguments.append("-OUTSIDE")

    @cc("-CROP2D")
    def crop_2d(self,
                ortho_dim,
                *xy,
                outside=False):
        self.arguments += [ortho_dim, len(xy)]
        for x, y in xy:
            self.arguments += [x, y]
        if outside:
            self.arguments.append("-OUTSIDE")

    @cc("-CROSS_SECTION")
    def cross_section(self, xml_filename: str or Path):
        self.arguments.append(xml_filename)

    @cc("-SOR")
    def sor(self, number_of_neighbors, sigma):
        self.arguments += [number_of_neighbors, sigma]

    @cc("-SF_ARITHMETIC")
    def sf_arithmetic(self,
                      sf_index: int or str,
                      operation: SF_ARITHMETICS):
        if isinstance(sf_index, str):
            assert sf_index == "LAST"
        else:
            assert sf_index >= 0
        self.arguments += [sf_index, operation]

    @cc("-SF_OP")
    def sf_operation(self,
                     sf_index,
                     operation: SF_OPERATIONS,
                     value):
        if isinstance(sf_index, str):
            assert sf_index == "LAST"
        else:
            assert sf_index >= 0
        self.arguments += [sf_index, operation, value]

    @cc("-CBANDING")
    def c_banding(self,
                  dim: str,
                  freq: int):
        assert dim in ["X", "Y", "Z"]
        self.arguments += [dim, freq]

    @cc("-SF_COLOR_SCALE")
    def sf_color_scale(self, filename: str or Path):
        self.arguments.append(filename)

    @cc("-SF_CONVERT_TO_RGB")
    def sf_convert_to_rgb(self, replace_existing: bool):
        replace_existing = BOOL.from_bool(replace_existing)
        self.arguments.append(replace_existing)

    @cc("-M3C2")
    def m3c2(self, parameters_file: str or Path):
        self.arguments.append(parameters_file)

    @cc("-CANUPO_CLASSIFY")
    def canupo_classify(self,
                        parameters_file: str or Path,
                        use_confidence=None):
        self.arguments.append(parameters_file)
        if use_confidence is not None:
            assert 0 <= use_confidence <= 1
            self.arguments += ["USE_CONFIDENCE", use_confidence]

    @cc("-PCV")
    def pcv(self,
            n_rays=None,
            is_closed=False,
            northern_hemisphere=False,
            resolution=None):
        if n_rays is not None:
            self.arguments += ["-N_RAYS", n_rays]
        if is_closed:
            self.arguments.append("-IS_CLOSED")
        if northern_hemisphere:
            self.arguments.append("-180")
        if resolution is not None:
            self.arguments += ["-RESOLTION", resolution]

    @cc("-RANSAC")
    def ransac(self,
               epsilon_absolute: float = None,
               epsilon_percentage_of_scale: float = None,
               bitmap_epsilon_percentage_of_scale: float = None,
               bitmap_epsilon_absolute: float = None,
               support_points: int = None,
               max_normale_dev_degree: float = None,
               probability: float = None,
               out_cloud_dir: str = None,
               out_mesh_dir: str = None,
               out_pair_dir: str = None,
               out_group_dir: str = None,
               output_individual_subclouds: bool = False,
               output_individual_primitives: bool = False,
               output_individual_paired_cloud_primitive: bool = False,
               output_grouped: bool = False,
               enable_primitive: List[RANSAC_PRIMITIVES] = None):
        if epsilon_absolute is not None:
            self.arguments += ["EPSILON_ABSOLUTE", epsilon_absolute]
        if epsilon_percentage_of_scale is not None:
            assert 0 < epsilon_percentage_of_scale < 1
            self.arguments += ["EPSILON_PERCENTAGE_OF_SCALE", epsilon_percentage_of_scale]
        if bitmap_epsilon_percentage_of_scale is not None:
            assert 0 < bitmap_epsilon_percentage_of_scale < 1
            self.arguments += ["BITMAP_EPISLON_PERCENTAGE_OF_SCALE", bitmap_epsilon_percentage_of_scale]
        if bitmap_epsilon_absolute is not None:
            self.arguments += ["BITMAP_EPSILON_ABSOLUTE", bitmap_epsilon_absolute]
        if support_points is not None:
            self.arguments += ["SUPPORT_POINTS", support_points]
        if max_normale_dev_degree is not None:
            self.arguments += ["MAX_NORMAL_DEV", max_normale_dev_degree]
        if probability is not None:
            self.arguments += ["PROBABILITY", probability]
        if out_cloud_dir is not None:
            self.arguments += ["OUT_CLOUD_DIR", out_cloud_dir]
        if out_mesh_dir is not None:
            self.arguments += ["OUT_MESH_DIR", out_mesh_dir]
        if out_pair_dir is not None:
            self.arguments += ["OUT_PAIR_DIR", out_pair_dir]
        if out_group_dir is not None:
            self.arguments += ["OUT_GROUP_DIR", out_group_dir]
        if output_individual_subclouds:
            self.arguments.append("OUTPUT_INDIVIDUAL_SUBCLOUDS")
        if output_individual_primitives:
            self.arguments.append("OUTPUT_INDIVIDUAL_PRIMITIVES")
        if output_individual_paired_cloud_primitive:
            self.arguments.append("OUTPUT_INDIVIDUAL_PAIRED_CLOUD_PRIMITIVE")
        if output_grouped:
            self.arguments.append("OUTPUT_GROUPED")
        if enable_primitive is not None:
            self.arguments.append("ENABLE_PRIMITIVE")
            self.arguments += enable_primitive

    @cc("-C_EXPORT_FMT")
    def cloud_export_format(self,
                            format_: CLOUD_EXPORT_FORMAT,
                            precision=12,
                            separator=SEPARATOR.SPACE,
                            add_header=False,
                            add_point_count=False,
                            extension: str = None):
        self.arguments.append(format_)

        if precision != 12:
            self.arguments += ["-PREC", precision]
        if isinstance(separator, str):
            separator = SEPARATOR.from_string(separator)
        if separator != SEPARATOR.SPACE:
            self.arguments += ["-SEP", separator]
        if add_header:
            self.arguments.append("-ADD_HEADER")
        if add_point_count:
            self.arguments.append("-ADD_PTS_COUNT")
        if extension is not None:
            self.arguments += ["-EXT", extension]

    @cc("-M_EXPORT_FMT")
    def mesh_export_format(self,
                           format_: MESH_EXPORT_FORMAT,
                           extension: str = None):
        self.arguments.append(format_)

        if extension is not None:
            self.arguments += ["-EXT", extension]

    @cc("-H_EXPORT_FMT")
    def hierarchy_export_format(self, format_: str = "BIN"):
        """ Mostly the BIN format, but other formats that support a collection of objects might be eligible. """
        self.arguments.append(format_)

    @cc("-FBX")
    def fbx_export_format(self, format_: FBX_EXPORT_FORMAT):
        self.arguments += ["-EXPORT_FMT", format_]

    @cc("-PLY_EXPORT_FMT")
    def ply_export_format(self, format_: PLY_EXPORT_FORMAT):
        self.arguments.append(format_)

    @cc("-NO_TIMESTAMP")
    def no_timestamp(self):
        pass

    @cc("-BUNDLER_IMPORT")
    def bundler_import(self,
                       filename: str or Path,
                       alt_keypoints=None,
                       scale_factor=None,
                       color_dtm=None,
                       undistort=False):
        self.arguments.append(filename)
        if alt_keypoints is not None:
            self.arguments += ["-ALT_KEYPOINTS", alt_keypoints]
        if scale_factor is not None:
            self.arguments += ["-SCALE_FACTOR", scale_factor]
        if color_dtm is not None:
            self.arguments += ["-COLOR_DTM", color_dtm]
        if undistort:
            self.arguments.append("-UNDISTORT")

    @cc("-DROP_GLOBAL_SHIFT")
    def drop_global_shift(self):
        pass

    @cc("-SET_ACTIVE_SF")
    def set_active_sf(self, index: int):
        self.arguments.append(index)

    @cc("-REMOVE_ALL_SFS")
    def remove_all_sfs(self):
        pass

    @cc("-REMOVE_RGB")
    def remove_rgb(self):
        pass

    @cc("-REMOVE_NORMALS")
    def remove_normals(self):
        pass

    @cc("-REMOVE_SCAN_GRIDS")
    def remove_scan_grid(self):
        pass

    @cc("-AUTO_SAVE")
    def auto_save(self, on_off: bool):
        on_off = ONOFF.from_bool(on_off)
        self.arguments.append(on_off)

    @cc("-SAVE_CLOUDS")
    def save_clouds(self,
                    *files,
                    all_at_once=False):
        if all_at_once:
            self.arguments.append("ALL_AT_ONCE")
        if files:
            self.arguments.append("FILE")
            self.arguments.append('"' + ' '.join(map(str, files)) + '"')

    @cc("-SAVE_MESHES")
    def save_meshes(self,
                    *files,
                    all_at_once=False):
        if all_at_once:
            self.arguments.append("ALL_AT_ONCE")
        if files:
            self.arguments.append("FILE")
            self.arguments.append('"' + ' '.join(map(str, files)) + '"')

    @cc("-CLEAR")
    def clear(self):
        pass

    @cc("-CLEAR_CLOUDS")
    def clear_clouds(self):
        pass

    @cc("-CLEAR_MESHES")
    def clear_meshes(self):
        pass

    @cc("-POP_CLOUDS")
    def pop_clouds(self):
        pass

    @cc("-POP_MESHES")
    def pop_meshes(self):
        pass

    @cc("-LOG_FILE")
    def log_file(self, filename: Path or str):
        self.arguments.append(filename)

    # Alias
    def statistical_outlier_removal(self,
                                    number_of_neighbors,
                                    sigma):
        self.sor(number_of_neighbors, sigma)
