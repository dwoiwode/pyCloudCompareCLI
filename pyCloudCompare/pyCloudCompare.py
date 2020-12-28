""" Python Wrapper for CloudCompare CLI """
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import List, Any

__version__ = "0.0.1"
__author__ = "dwoiwode"
__license__ = "GNU GPLv3"

_platform = sys.platform
if _platform.startswith("win32"):
    DEFAULT_EXECUTABLE = r"C:\Program Files\CloudCompare\CloudCompare.exe"
elif _platform.startswith("linux"):
    DEFAULT_EXECUTABLE = "pyCloudCompare"  # TODO: Update default executable for linux
elif _platform.startswith("darwin"):
    DEFAULT_EXECUTABLE = "pyCloudCompare"  # TODO: Update default executable for macOS


class FLAGS(Enum):
    SILENT = "-SILENT"
    OPEN = "-O"
    MOMENT = "-MOMENT"
    FEATURE = "-FEATURE"
    OCTREE_NORMALS = "-OCTREE_NORMALS"
    COMPUTE_NORMALS = "-COMPUTE_NORMALS"
    NORMALS_TO_SFS = "-NORMALS_TO_SFS"
    NORMALS_TO_DIP = "-NORMALS_TO_DIP"
    CLEAR_NORMALS = "-CLEAR_NORMALS"
    ORIENT_NORMS_MST = "-ORIENT_NORMS_MST"
    MERGE_CLOUDS = "-MERGE_CLOUDS"
    MERGE_MESHES = "-MERGE_MESHES"
    SS = "-SS"
    EXTRACT_CC = "-EXTRACT_CC"
    SAMPLE_MESH = "-SAMPLE_MESH"
    EXTRACT_VERTICES = "-EXTRACT_VERTICES"
    C2C_DIST = "-C2C_DIST"
    C2M_DIST = "-C2M_DIST"
    RASTERIZE = "-RASTERIZE"
    VOLUME = "-VOLUME"
    STAT_TEST = "-STAT_TEST"
    COORD_TO_SF = "-COORD_TO_SF"
    FILTER_SF = "-FILTER_SF"
    DENSITY = "-DENSITY"
    APPROX_DENSITY = "-APPROX_DENSITY"
    ROUGH = "-ROUGH"
    CURV = "-CURV"
    SF_GRAD = "-SF_GRAD"
    BEST_FIT_PLANE = "-BEST_FIT_PLANE"
    APPLY_TRANS = "-APPLY_TRANS"
    MATCH_CENTERS = "-MATCH_CENTERS"
    DELAUNAY = "-DELAUNAY"
    ICP = "-ICP"
    CROP = "-CROP"
    CROP2D = "-CROP2D"
    CROSS_SECTION = "-CROSS_SECTION"
    SOR = "-SOR"
    SF_ARITHMETIC = "-SF_ARITHMETIC"
    SF_OP = "-SF_OP"
    CBANDING = "-CBANDING"
    SF_COLOR_SCALE = "-SF_COLOR_SCALE"
    SF_CONVERT_TO_RGB = "-SF_CONVERT_TO_RGB"
    M3C2 = "-M3C2"
    CANUPO_CLASSIFY = "-CANUPO_CLASSIFY"
    PCV = "-PCV"
    RANSAC = "-RANSAC"
    C_EXPORT_FMT = "-C_EXPORT_FMT"
    M_EXPORT_FMT = "-M_EXPORT_FMT"
    H_EXPORT_FMT = "-H_EXPORT_FMT"
    FBX = "-FBX"
    PLY_EXPORT_FMT = "-PLY_EXPORT_FMT"
    NO_TIMESTAMP = "-NO_TIMESTAMP"
    BUNDLER_IMPORT = "-BUNDLER_IMPORT"
    DROP_GLOBAL_SHIFT = "-DROP_GLOBAL_SHIFT"
    SET_ACTIVE_SF = "-SET_ACTIVE_SF"
    REMOVE_ALL_SFS = "-REMOVE_ALL_SFS"
    REMOVE_RGB = "-REMOVE_RGB"
    REMOVE_NORMALS = "-REMOVE_NORMALS"
    REMOVE_SCAN_GRIDS = "-REMOVE_SCAN_GRIDS"
    AUTO_SAVE = "-AUTO_SAVE"
    SAVE_CLOUDS = "-SAVE_CLOUDS"
    SAVE_MESHES = "-SAVE_MESHES"
    CLEAR = "-CLEAR"
    CLEAR_CLOUDS = "-CLEAR_CLOUDS"
    CLEAR_MESHES = "-CLEAR_MESHES"
    POP_CLOUDS = "-POP_CLOUDS"
    POP_MESHES = "-POP_MESHES"
    LOG_FILE = "-LOG_FILE"


class OPTIONS(Enum):
    SKIP = "-SKIP"  # Open
    GLOBAL_SHIFT = "-GLOBAL_SHIFT"  # Open
    ORIENT = "-ORIENT"  # Octree normals
    MODEL = "-MODEL"  # Octree normals
    GRID_STEP = "-GRID_STEP"  # Rasterize/Volume
    VERT_DIR = "-VERT_DIR"  # Rasterize/Volume
    PROJ = "-PROJ"  # Rasterize
    SF_PROJ = "-SF_PROJ"  # Rasterize
    EMPTY_FILL = "-EMPTY_FILL"  # Rasterize
    CUSTOM_HEIGHT = "-CUSTOM_HEIGHT"  # Rasterize
    OUTPUT_CLOUD = "-OUTPUT_CLOUD"  # Rasterize
    OUTPUT_MESH = "-OUTPUT_MESH"  # Rasterize/Volume
    OUTPUT_RASTER_Z = "-OUTPUT_RASTER_Z"  # Rasterize
    OUTPUT_RASTER_RGB = "-OUTPUT_RASTER_RGB"  # Rasterize
    GROUND_IS_FIRST = "-GROUND_IS_FIRST"  # Volume
    CONST_HEIGHT = "-CONST_HEIGHT"  # Volume
    SPLIT_XYZ = "-SPLIT_XYZ"
    MAX_DIST = "-MAX_DIST"
    OCTREE_LEVEL = "-OCTREE_LEVEL"
    MAX_TCOUNT = "-MAX_TCOUNT"
    FLIP_NORMS = "-FLIP_NORMS"
    TYPE = "-TYPE"

    # Delaunay
    AA = "-AA"
    BEST_FIT = "-BEST_FIT"
    MAX_EDGE_LENGTH = "-MAX_EDGE_LENGTH"

    # ICP
    REFERENCE_IS_FIRST = "-REFERENCE_IS_FIRST"
    MIN_ERROR_DIFF = "-MIN_ERROR_DIFF"
    ITER = "-ITER"
    OVERLAP = "-OVERLAP"
    ADJUST_SCALE = "-ADJUST_SCALE"
    RANDOM_SAMPLING_LIMIT = "-RANDOM_SAMPLING_LIMIT"
    FARTHEST_REMOVAL = "-FARTHEST_REMOVAL"
    DATA_SF_AS_WEIGHTS = "-DATA_SF_AS_WEIGHTS"
    MODEL_SF_AS_WEIGHTS = "-MODEL_SF_AS_WEIGHTS"
    ROT = "-ROT"

    # Crop, Crop2D
    OUTSIDE = "-OUTSIDE"

    # Canupo Classify
    USE_CONFIDENCE = "USE_CONFIDENCE"  # TODO: Test whether "-" is needed before Flag! (Manual says "no")

    # PCV
    N_RAYS = "-N_RAYS"
    IS_CLOSED = "-IS_CLOSED"
    NORTHERN_HEMISPHERE = "-180"
    RESOLTION = "-RESOLUTION"

    # RANSAC
    # SAVE CLOUDS/MESHES
    ALL_AT_ONCE = "ALL_AT_ONCE"  # TODO: Test whether "-" is needed before Flag!
    FILE = "FILE"  # TODO: Test whether "-" is needed before Flag!

    # BUNDLER IMPORT
    ALT_KEYPOINTS = "-ALT_KEYPOINTS"
    SCALE_FACTOR = "-SCALE_FACTOR"
    COLOR_DTM = "-COLOR_DTM"
    UNDISTORT = "-UNDISTORT"

    # EXPORT
    PRECISION = "-PREC"
    SEPARATOR = "-SEP"
    ADD_HEADER = "-ADD_HEADER"
    ADD_PTS_COUNT = "-ADD_PTS_COUNT"
    EXT = "-EXT"
    EXPORT_FMT = "-EXPORT_FMT"


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
    def fromString(cls, s):
        if s == " ":
            return SEPARATOR.SPACE
        if s == ";":
            return SEPARATOR.SEMICOLON
        if s == ",":
            return SEPARATOR.COMMA
        if s == "\t":
            return SEPARATOR.TAB
        raise ValueError(f"Invalid separator (Has to be ' ;,\\t'. Is: '{s}'")


class CLOUD_EXPORT_FORMAT(Enum):
    ASC = "ASC"
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


class Bool(Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"

    @classmethod
    def fromBool(cls, value):
        if value:
            return Bool.TRUE
        return Bool.FALSE


class ONOFF(Enum):
    ON = "ON"
    OFF = "OFF"

    @classmethod
    def fromBool(cls, value):
        if value:
            return Bool.ON
        return Bool.OFF


def cc(func):
    @wraps(func)
    def wrapper(self: "CloudCompareCMD", *args, **kwargs):
        oldArguments = list(self.arguments)
        funcName = func.__name__
        try:
            func(self, *args, **kwargs)
            self._validateCommand()
            n = len(oldArguments)
            self.commands.append(CCCommand(funcName, self.arguments[n:]))
        except Exception as e:
            print(f"Failed to add {funcName}! Cause: {str(e)}. Rolling back")
            self.arguments = oldArguments

    return wrapper


@dataclass
class CCCommand:
    funcName: str
    arguments: tuple


class CloudCompareCMD:
    def __init__(self, executable=DEFAULT_EXECUTABLE, arguments: List[Any] = None):
        self.exec = Path(executable).absolute()
        self.arguments = arguments or []
        self.commands: List[CCCommand] = []

        # Internal flags
        self._addedToPath = False

    def __repr__(self):
        return f"CloudCompareCMD({self.exec}, {self.arguments})"

    def __str__(self):
        return self.toCmd()

    def toCmd(self):
        argumentsCopy = list(self.arguments)
        isSilent = False
        while FLAGS.SILENT in argumentsCopy:
            isSilent = True
            argumentsCopy.remove(FLAGS.SILENT)
        if isSilent:
            argumentsCopy.insert(0, FLAGS.SILENT)
        args = " ".join([arg.value if isinstance(arg, Enum) else str(arg) for arg in argumentsCopy])
        return f'"{str(self.exec)}" {args}'

    def execute(self):
        if not self._addedToPath:
            self._addedToPath = True
            sys.path.append(str(self.exec.parent))

        proc = subprocess.run(self.toCmd())
        ret = proc.returncode
        if ret != 0:
            raise RuntimeWarning(f"Non-Zero Returncode ({ret})")
        return ret

    def _validateCommand(self):
        cmd = self.toCmd()

    # =================== Commands ===================
    @cc
    def silent(self, isSilent= True):
        if isSilent:
            if FLAGS.SILENT not in self.arguments:
                self.arguments.append(FLAGS.SILENT)
        else:
            while FLAGS.SILENT in self.arguments:
                self.arguments.remove(FLAGS.SILENT)

    @cc
    def open(self, filepath: str or Path, skip=None, global_shift=None):
        self.arguments.append(FLAGS.OPEN)
        self.arguments.append(f"\"{filepath}\"")
        if skip is not None:
            self.arguments.append(OPTIONS.SKIP)
            self.arguments.append(skip)
        if global_shift is not None:
            self.arguments.append(OPTIONS.GLOBAL_SHIFT)
            if isinstance(global_shift, str):
                assert global_shift == "AUTO" or global_shift == "FIRST"
            elif isinstance(global_shift, (tuple, list)):
                x, y, z = global_shift
                global_shift = f"{x} {y} {z}"
            self.arguments.append(global_shift)

    @cc
    def moment(self, radius: float):
        self.arguments += [FLAGS.MOMENT, radius]

    @cc
    def feature(self, feature: FEATURES, radius: float):
        self.arguments += [FLAGS.FEATURE, feature, radius]

    @cc
    def octreeNormals(self, radius: float, orient: OCTREE_ORIENT = None, model: OCTREE_MODEL = None):
        self.arguments += [FLAGS.OCTREE_NORMALS, radius]
        if orient is not None:
            self.arguments.append(OPTIONS.ORIENT)
            self.arguments.append(orient)
        if model is not None:
            self.arguments.append(OPTIONS.MODEL)
            self.arguments.append(model)

    @cc
    def computeNormals(self):
        self.arguments.append(FLAGS.COMPUTE_NORMALS)

    @cc
    def normalsToSfs(self):
        self.arguments.append(FLAGS.NORMALS_TO_SFS)

    @cc
    def normalsToDip(self):
        self.arguments.append(FLAGS.NORMALS_TO_DIP)

    @cc
    def clearNormals(self):
        self.arguments.append(FLAGS.CLEAR_NORMALS)

    @cc
    def orientNormsMst(self, numberOfNeighbors: int):
        self.arguments += [FLAGS.ORIENT_NORMS_MST, numberOfNeighbors]

    @cc
    def mergeClouds(self):
        self.arguments.append(FLAGS.MERGE_CLOUDS)

    @cc
    def mergeMeshes(self):
        self.arguments.append(FLAGS.MERGE_MESHES)

    @cc
    def subsampling(self, algorithm: SS_ALGORITHM, parameter):
        self.arguments += [FLAGS.SS, algorithm, parameter]

    @cc
    def extractCC(self, octreeLevel, minPoints):
        assert 1 <= octreeLevel <= 21
        self.arguments += [FLAGS.EXTRACT_CC, octreeLevel, minPoints]

    @cc
    def sampleMesh(self, method: SAMPLE_METHOD, parameter):
        self.arguments += [FLAGS.SAMPLE_MESH, method, parameter]

    @cc
    def extractVertices(self):
        self.arguments.append(FLAGS.EXTRACT_VERTICES)

    @cc
    def c2cDist(self, splitXYZ=None, maxDist=None, octreeLevel=None, model=None, maxTCount=None):
        self.arguments.append(FLAGS.C2C_DIST)
        if splitXYZ is not None:
            self.arguments += [OPTIONS.SPLIT_XYZ, splitXYZ]
        if maxDist is not None:
            self.arguments += [OPTIONS.MAX_DIST, maxDist]
        if octreeLevel is not None:
            self.arguments += [OPTIONS.OCTREE_LEVEL, octreeLevel]
        if model is not None:
            # TODO: Improve model input
            self.arguments += [OPTIONS.MODEL, model]
        if maxTCount is not None:
            self.arguments += [OPTIONS.MAX_TCOUNT, maxTCount]

    @cc
    def c2mDist(self, flipNorms: bool = False, maxDist=None, octreeLevel=None, maxTCount=None):
        self.arguments.append(FLAGS.C2C_DIST)
        if flipNorms:
            self.arguments.append(flipNorms)
        if maxDist is not None:
            self.arguments += [OPTIONS.MAX_DIST, maxDist]
        if octreeLevel is not None:
            self.arguments += [OPTIONS.OCTREE_LEVEL, octreeLevel]
        if maxTCount is not None:
            self.arguments += [OPTIONS.MAX_TCOUNT, maxTCount]

    @cc
    def rasterize(self, gridStep, vertDir=None, proj=None, sfProj=None, emptyFill=None, customHeight=None,
                  outputCloud=False, outputMesh=False, outputRasterZ=False, outputRasterRGB=False):
        self.arguments += [FLAGS.RASTERIZE, OPTIONS.GRID_STEP, gridStep]
        if vertDir is not None:
            self.arguments += [OPTIONS.VERT_DIR, vertDir]
        if proj is not None:
            self.arguments += [OPTIONS.PROJ, proj]
        if sfProj is not None:
            self.arguments += [OPTIONS.SF_PROJ, sfProj]
        if emptyFill is not None:
            self.arguments += [OPTIONS.EMPTY_FILL, emptyFill]
        if customHeight is not None:
            self.arguments += [OPTIONS.CUSTOM_HEIGHT, customHeight]
        if outputCloud:
            self.arguments.append(OPTIONS.OUTPUT_CLOUD)
        if outputMesh:
            self.arguments.append(OPTIONS.OUTPUT_MESH)
        if outputRasterZ:
            self.arguments.append(OPTIONS.OUTPUT_RASTER_Z)
        if outputRasterRGB:
            self.arguments.append(OPTIONS.OUTPUT_RASTER_RGB)

    @cc
    def volume(self, gridStep, vertDir=None, constHeight=None, groundIsFirst=False, outputMesh=False):
        self.arguments += [FLAGS.RASTERIZE, OPTIONS.GRID_STEP, gridStep]
        if vertDir is not None:
            self.arguments += [OPTIONS.VERT_DIR, vertDir]
        if constHeight is not None:
            self.arguments += [OPTIONS.CONST_HEIGHT, constHeight]
        if groundIsFirst:
            self.arguments.append(OPTIONS.GROUND_IS_FIRST)
        if outputMesh:
            self.arguments.append(OPTIONS.OUTPUT_MESH)

    @cc
    def statTest(self, distribution, distributionParameter, pValue, neighborCount):
        self.arguments += [FLAGS.STAT_TEST, distribution, distributionParameter, pValue, neighborCount]

    @cc
    def coordToSF(self, dimension):
        assert dimension in ["X", "Y", "Z"]
        self.arguments += [FLAGS.COORD_TO_SF, dimension]

    @cc
    def filterSF(self, minVal, maxVal):
        SPECIAL_WORDS = ["MIN", "DISP_MIN", "SAT_MIN", "MAX", "DISP_MAX", "SAT_MAX"]
        if isinstance(minVal, str):
            assert minVal in SPECIAL_WORDS
        if isinstance(maxVal, str):
            assert maxVal in SPECIAL_WORDS
        self.arguments += [FLAGS.FILTER_SF, minVal, maxVal]

    @cc
    def density(self, sphereRadius, type: DENSITY_TYPE = None):
        self.arguments += [FLAGS.DENSITY, sphereRadius]
        if type is not None:
            self.arguments += [OPTIONS.TYPE, type]

    @cc
    def approxDensity(self, type: DENSITY_TYPE = None):
        self.arguments.append(FLAGS.APPROX_DENSITY)
        if type is not None:
            self.arguments += [OPTIONS.TYPE, type]

    @cc
    def rough(self, kernelSize):
        self.arguments += [FLAGS.ROUGH, kernelSize]

    @cc
    def curvature(self, type: CURV_TYPE, kernelSize):
        self.arguments += [FLAGS.CURV, type, kernelSize]

    @cc
    def sfGrad(self, euclidian: bool):
        euclidian = Bool.fromBool(euclidian)
        self.arguments += [FLAGS.SF_GRAD, euclidian]

    @cc
    def bestFitPlane(self, makeHoriz=False, keepLoaded=False):
        self.arguments.append(FLAGS.BEST_FIT_PLANE)
        if makeHoriz:
            self.arguments.append(OPTIONS.MAKE_HORIZ)
        if keepLoaded:
            self.arguments.append(OPTIONS.KEEP_LOADED)

    @cc
    def applyTransformation(self, filename: str or Path):
        self.arguments += [FLAGS.APPLY_TRANS, filename]

    @cc
    def matchCenters(self):
        self.arguments.append(FLAGS.MATCH_CENTERS)

    @cc
    def delaunay(self, xyPlane=False, bestFit=False, maxEdgeLength=None):
        self.arguments.append(FLAGS.DELAUNAY)
        if xyPlane:
            self.arguments.append(OPTIONS.AA)
        if bestFit:
            self.arguments.append(OPTIONS.BEST_FIT)
        if maxEdgeLength is not None:
            self.arguments += [OPTIONS.MAX_EDGE_LENGTH, maxEdgeLength]

    @cc
    def icp(self, referenceIsFirst=False, minErrorDiff: float = 1e-6, iter: int = None, overlap: int = 100,
            adjustScale=False, randomSamplingLimit=20000, farthestRemoval=False, rot: ROT = None):
        self.arguments.append(FLAGS.ICP)
        if referenceIsFirst:
            self.arguments.append(OPTIONS.REFERENCE_IS_FIRST)
        if minErrorDiff != 1e-6:
            self.arguments += [OPTIONS.MIN_ERROR_DIFF, minErrorDiff]
        if iter is not None:
            self.arguments += [OPTIONS.ITER, iter]
        if overlap != 100:
            self.arguments += [OPTIONS.OVERLAP, overlap]
        if adjustScale:
            self.arguments.append(OPTIONS.ADJUST_SCALE)
        if randomSamplingLimit != 20000:
            self.arguments += [OPTIONS.RANDOM_SAMPLING_LIMIT, randomSamplingLimit]
        if farthestRemoval:
            self.arguments.append(OPTIONS.FARTHEST_REMOVAL)
        # TODO: DATA_SF_WEIGHTS
        # TODO: MODEL_SF_WEIGHTS
        if rot is not None:
            self.arguments.append(OPTIONS.ROT.value + rot.value)

    @cc
    def crop(self, xMin, yMin, zMin, xMax, yMax, zMax, outside=False):
        boxString = f"{xMin}:{yMin}:{zMin}:{xMax}:{yMax}:{zMax}"
        self.arguments += [FLAGS.CROP, boxString]
        if outside:
            self.arguments.append(OPTIONS.OUTSIDE)

    @cc
    def crop2D(self, orthoDim, *xy, outside=False):
        self.arguments += [FLAGS.CROP2D, orthoDim, len(xy)]
        for x, y in xy:
            self.arguments += [x, y]
        if outside:
            self.arguments.append(OPTIONS.OUTSIDE)

    @cc
    def crossSection(self, xmlFilename: str or Path):
        self.arguments += [FLAGS.CROSS_SECTION, xmlFilename]

    @cc
    def sor(self, numberOfNeighbors, sigma):
        self.arguments += [FLAGS.SOR, numberOfNeighbors, sigma]

    @cc
    def sfArithmetic(self, sfIndex: int or str, operation: SF_ARITHMETICS):
        if isinstance(sfIndex, str):
            assert sfIndex == "LAST"
        else:
            assert sfIndex >= 0
        self.arguments += [FLAGS.SF_ARITHMETIC, sfIndex, operation]

    @cc
    def sfOperation(self, sfIndex, operation: SF_OPERATIONS, value):
        if isinstance(sfIndex, str):
            assert sfIndex == "LAST"
        else:
            assert sfIndex >= 0
        self.arguments += [FLAGS.SF_OP, sfIndex, operation, value]

    @cc
    def cBanding(self, dim: str, freq: int):
        assert dim in ["X", "Y", "Z"]
        self.arguments += [FLAGS.CBANDING, dim, freq]

    @cc
    def sfColorScale(self, filename: str or Path):
        self.arguments += [FLAGS.SF_COLOR_SCALE, filename]

    @cc
    def sfConvertToRGB(self, replaceExisting: bool):
        replaceExisting = Bool.fromBool(replaceExisting)
        self.arguments += [FLAGS.SF_CONVERT_TO_RGB, replaceExisting]

    @cc
    def m3c2(self, parametersFile: str or Path):
        self.arguments += [FLAGS.M3C2, parametersFile]

    @cc
    def canupoClassify(self, parametersFile: str or Path, useConfidence=None):
        self.arguments += [FLAGS.CANUPO_CLASSIFY, parametersFile]
        if useConfidence is not None:
            assert 0 <= useConfidence <= 1
            self.arguments += [OPTIONS.USE_CONFIDENCE, useConfidence]

    @cc
    def pcv(self, nRays=None, isClosed=False, northernHemisphere=False, resolution=None):
        self.arguments.append(FLAGS.PCV)
        if nRays is not None:
            self.arguments += [OPTIONS.N_RAYS, nRays]
        if isClosed:
            self.arguments.append(OPTIONS.IS_CLOSED)
        if northernHemisphere:
            self.arguments.append(OPTIONS.NORTHERN_HEMISPHERE)
        if resolution is not None:
            self.arguments += [OPTIONS.RESOLTION, resolution]

    @cc
    def ransac(self, epsilonAbsolute, epsilonPercentageOfScale, bitMap):
        pass  # TODO

    @cc
    def cloudExportFormat(self, format: CLOUD_EXPORT_FORMAT, precision=12, separator=SEPARATOR.SPACE, addHeader=False,
                          addPointCount=False, extension: str = None):
        self.arguments += [FLAGS.C_EXPORT_FMT, format]

        if precision != 12:
            self.arguments += [OPTIONS.PRECSION, precision]
        if isinstance(separator, str):
            separator = SEPARATOR.fromString(separator)
        if separator != SEPARATOR.SPACE:
            self.arguments += [OPTIONS.SEPARATOR, separator]
        if addHeader:
            self.arguments.append(OPTIONS.ADD_HEADER)
        if addPointCount:
            self.arguments.append(OPTIONS.ADD_PTS_COUNT)
        if extension is not None:
            self.arguments += [OPTIONS.EXT, extension]

    @cc
    def meshExportFormat(self, format: MESH_EXPORT_FORMAT, extension: str = None):
        self.arguments += [FLAGS.M_EXPORT_FMT, format]

        if extension is not None:
            self.arguments += [OPTIONS.EXT, extension]

    @cc
    def hierarchyExportFormat(self, format: str = "BIN"):
        """ Mostly the BIN format, but other formats that support a collection of objects might be elligible. """
        self.arguments += [FLAGS.H_EXPORT_FMT, format]

    @cc
    def fbxExportFormat(self, format: FBX_EXPORT_FORMAT):
        self.arguments += [FLAGS.FBX, OPTIONS.EXPORT_FMT, format]

    @cc
    def plyExportFormat(self, format: PLY_EXPORT_FORMAT):
        self.arguments += [FLAGS.PLY_EXPORT_FMT, format]

    @cc
    def noTimestamp(self):
        self.arguments.append(FLAGS.NO_TIMESTAMP)

    @cc
    def bundlerImport(self, filename: str or Path, altKeypoints=None, scaleFactor=None, colorDTM=None, undistort=False):
        self.arguments += [FLAGS.BUNDLER_IMPORT, filename]
        if altKeypoints is not None:
            self.arguments += [OPTIONS.ALT_KEYPOINTS, altKeypoints]
        if scaleFactor is not None:
            self.arguments += [OPTIONS.SCALE_FACTOR, scaleFactor]
        if colorDTM is not None:
            self.arguments += [OPTIONS.COLOR_DTM, colorDTM]
        if undistort:
            self.arguments.append(OPTIONS.UNDISTORT)

    @cc
    def dropGlobalShift(self):
        self.arguments.append(FLAGS.DROP_GLOBAL_SHIFT)

    @cc
    def setActiveSF(self, index: int):
        self.arguments += [FLAGS.SET_ACTIVE_SF, index]

    @cc
    def removeAllSFS(self):
        self.arguments.append(FLAGS.REMOVE_ALL_SFS)

    @cc
    def removeRGB(self):
        self.arguments.append(FLAGS.REMOVE_RGB)

    @cc
    def removeNormals(self):
        self.arguments.append(FLAGS.REMOVE_NORMALS)

    @cc
    def removeScanGrid(self):
        self.arguments.append(FLAGS.REMOVE_SCAN_GRIDS)

    @cc
    def autoSave(self, onOff: bool):
        onOff = ONOFF.fromBool(onOff)
        self.arguments += [FLAGS.AUTO_SAVE, onOff]

    @cc
    def saveClouds(self, *files, allAtOnce=False):
        self.arguments.append(FLAGS.SAVE_CLOUDS)
        if allAtOnce:
            self.arguments.append(OPTIONS.ALL_AT_ONCE)
        if files:
            self.arguments.append(OPTIONS.FILE)
            self.arguments.append('"' + ' '.join(map(str, files)) + '"')

    @cc
    def saveMeshes(self,*files, allAtOnce=False):
        self.arguments.append(FLAGS.SAVE_MESHES)
        if allAtOnce:
            self.arguments.append(OPTIONS.ALL_AT_ONCE)
        if files:
            self.arguments.append(OPTIONS.FILE)
            self.arguments.append('"' + ' '.join(map(str, files)) + '"')

    @cc
    def clear(self):
        self.arguments.append(FLAGS.CLEAR)

    @cc
    def clearClouds(self):
        self.arguments.append(FLAGS.CLEAR_CLOUDS)

    @cc
    def clearMeshes(self):
        self.arguments.append(FLAGS.CLEAR_MESHES)

    @cc
    def popClouds(self):
        self.arguments.append(FLAGS.POP_CLOUDS)

    @cc
    def popMeshes(self):
        self.arguments.append(FLAGS.POP_MESHES)

    @cc
    def logFile(self, filename: Path or str):
        self.arguments.append(FLAGS.LOG_FILE)
        self.arguments.append(filename)

    # Alias
    def statisticalOutlierRemoval(self, numberOfNeighbors, sigma):
        self.sor(numberOfNeighbors, sigma)
