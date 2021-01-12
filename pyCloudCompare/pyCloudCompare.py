""" Python Wrapper for CloudCompare CLI """
import subprocess
import sys
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import List, Any

__version__ = "0.0.3"
__author__ = "dwoiwode"
__license__ = "MIT"

_platform = sys.platform
if _platform.startswith("win32"):
    DEFAULT_EXECUTABLE = r"C:\Program Files\CloudCompare\CloudCompare.exe"
elif _platform.startswith("linux"):
    DEFAULT_EXECUTABLE = "CloudCompare"  # TODO: Update default executable for linux
elif _platform.startswith("darwin"):
    DEFAULT_EXECUTABLE = "CloudCompare"  # TODO: Update default executable for macOS

_FLAG_SILENT = "-SILENT"

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


class BOOL(Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"

    @classmethod
    def fromBool(cls, value):
        if value:
            return BOOL.TRUE
        return BOOL.FALSE


class ONOFF(Enum):
    ON = "ON"
    OFF = "OFF"

    @classmethod
    def fromBool(cls, value):
        if value:
            return ONOFF.ON
        return ONOFF.OFF


def cc(flag=None):
    def wrapper1(func):
        @wraps(func)
        def wrapper2(self: "CloudCompareCLI", *args, **kwargs):
            oldArguments = list(self.arguments)
            funcName = func.__name__
            try:
                if flag is not None:
                    self.arguments.append(flag)
                func(self, *args, **kwargs)
                self._validateCommand()
                n = len(oldArguments)
                self.commands.append(CCCommand(funcName, self.arguments[n:]))
            except Exception as e:
                print(f"Failed to add {funcName}! Cause: {str(e)}. Rolling back")
                self.arguments = oldArguments

        return wrapper2

    return wrapper1


class CCCommand:
    def __init__(self, functionName: str, arguments: List[Any]):
        self.functionName = functionName
        self.arguments = arguments

    def __repr__(self):
        return f"CCCommand({self.functionName}, {self.arguments})"


class CloudCompareCLI:
    def __init__(self, executable=DEFAULT_EXECUTABLE):
        self.exec = Path(executable).absolute()
        if str(self.exec) not in sys.path:
            sys.path.append(str(self.exec.parent))

    def newCommand(self):
        return CloudCompareCommand(self)


class CloudCompareCommand:
    def __init__(self, ccCLI: CloudCompareCLI, arguments: List[Any] = None):
        self.ccCLI = ccCLI
        self.arguments = arguments or []
        self.commands: List[CCCommand] = []

        # Internal flags
        self._addedToPath = False

    def __repr__(self):
        return f"CloudCompareCMD({self.ccCLI}, {self.arguments})"

    def __str__(self):
        return self.toCmd()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.execute()

    def toCmd(self):
        argumentsCopy = list(self.arguments)
        isSilent = False
        while _FLAG_SILENT in argumentsCopy:
            isSilent = True
            argumentsCopy.remove(_FLAG_SILENT)
        if isSilent:
            argumentsCopy.insert(0, _FLAG_SILENT)
        args = " ".join([arg.value if isinstance(arg, Enum) else str(arg) for arg in argumentsCopy])
        return f'"{str(self.ccCLI.exec)}" {args}'

    def execute(self):
        proc = subprocess.run(self.toCmd(), stdout=subprocess.PIPE)
        ret = proc.returncode
        if ret != 0:
            print(proc.stdout.decode("utf-8"), file=sys.stderr)
            raise RuntimeWarning(f"Non-Zero Returncode ({ret})")
        return ret

    def _validateCommand(self):
        cmd = self.toCmd()

    # =================== Commands ===================
    @cc()
    def silent(self, isSilent=True):
        if isSilent:
            if _FLAG_SILENT not in self.arguments:
                self.arguments.append(_FLAG_SILENT)
        else:
            while _FLAG_SILENT in self.arguments:
                self.arguments.remove(_FLAG_SILENT)

    @cc("-O")
    def open(self, filepath: str or Path, skip=None, global_shift=None):
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
    def feature(self, feature: FEATURES, radius: float):
        self.arguments += [feature, radius]

    @cc("-OCTREE_NORMALS")
    def octreeNormals(self, radius: float, orient: OCTREE_ORIENT = None, model: OCTREE_MODEL = None):
        self.arguments.append(radius)
        if orient is not None:
            self.arguments.append("-ORIENT")
            self.arguments.append(orient)
        if model is not None:
            self.arguments.append("-MODEL")
            self.arguments.append(model)

    @cc("-COMPUTE_NORMALS")
    def computeNormals(self):
        pass

    @cc("-NORMALS_TO_SFS")
    def normalsToSfs(self):
        pass

    @cc("-NORMALS_TO_DIP")
    def normalsToDip(self):
        pass

    @cc("-CLEAR_NORMALS")
    def clearNormals(self):
        pass

    @cc("-ORIENT_NORMS_MST")
    def orientNormsMst(self, numberOfNeighbors: int):
        self.arguments.append(numberOfNeighbors)

    @cc("-MERGE_CLOUDS")
    def mergeClouds(self):
        pass

    @cc("-MERGE_MESHES")
    def mergeMeshes(self):
        pass

    @cc("-SS")
    def subsampling(self, algorithm: SS_ALGORITHM, parameter):
        self.arguments.append(algorithm)
        self.arguments.append(parameter)

    @cc("-EXTRACT_CC")
    def extractCC(self, octreeLevel, minPoints):
        assert 1 <= octreeLevel <= 21
        self.arguments += [octreeLevel, minPoints]

    @cc("-SAMPLE_MESH")
    def sampleMesh(self, method: SAMPLE_METHOD, parameter):
        self.arguments += [method, parameter]

    @cc("-EXTRACT_VERTICES")
    def extractVertices(self):
        pass

    @cc("-C2C_DIST")
    def c2cDist(self, splitXYZ=None, maxDist=None, octreeLevel=None, model=None, maxTCount=None):
        if splitXYZ is not None:
            self.arguments += ["-SPLIT_XYZ", splitXYZ]
        if maxDist is not None:
            self.arguments += ["-MAX_DIST", maxDist]
        if octreeLevel is not None:
            self.arguments += ["-OCTREE_LEVEL", octreeLevel]
        if model is not None:
            # TODO: Improve model input
            self.arguments += ["-MODEL", model]
        if maxTCount is not None:
            self.arguments += ["-MAX_TCOUNT", maxTCount]

    @cc("-C2M_DIST")
    def c2mDist(self, flipNorms: bool = False, maxDist=None, octreeLevel=None, maxTCount=None):
        if flipNorms:
            self.arguments.append(flipNorms)
        if maxDist is not None:
            self.arguments += ["-MAX_DIST", maxDist]
        if octreeLevel is not None:
            self.arguments += ["-OCTREE_LEVEL", octreeLevel]
        if maxTCount is not None:
            self.arguments += ["-MAX_TCOUNT", maxTCount]

    @cc("-RASTERIZE")
    def rasterize(self, gridStep, vertDir=None, proj=None, sfProj=None, emptyFill=None, customHeight=None,
                  outputCloud=False, outputMesh=False, outputRasterZ=False, outputRasterRGB=False):
        self.arguments += ["-GRID_STEP", gridStep]
        if vertDir is not None:
            self.arguments += ["-VERT_DIR", vertDir]
        if proj is not None:
            self.arguments += ["-PROJ", proj]
        if sfProj is not None:
            self.arguments += ["-SF_PROJ", sfProj]
        if emptyFill is not None:
            self.arguments += ["-EMPTY_FILL", emptyFill]
        if customHeight is not None:
            self.arguments += ["-CUSTOM_HEIGHT", customHeight]
        if outputCloud:
            self.arguments.append("-OUTPUT_CLOUD")
        if outputMesh:
            self.arguments.append("-OUTPUT_MESH")
        if outputRasterZ:
            self.arguments.append("-OUTPUT_RASTER_Z")
        if outputRasterRGB:
            self.arguments.append("-OUTPUT_RASTER_RGB")

    @cc("-VOLUME")
    def volume(self, gridStep, vertDir=None, constHeight=None, groundIsFirst=False, outputMesh=False):
        self.arguments += ["-GRID_STEP", gridStep]
        if vertDir is not None:
            self.arguments += ["-VERT_DIR", vertDir]
        if constHeight is not None:
            self.arguments += ["-CONST_HEIGHT", constHeight]
        if groundIsFirst:
            self.arguments.append("-GROUND_IS_FIRST")
        if outputMesh:
            self.arguments.append("-OUTPUT_MESH")

    @cc("-STAT_TEST")
    def statTest(self, distribution, distributionParameter, pValue, neighborCount):
        self.arguments += [distribution, distributionParameter, pValue, neighborCount]

    @cc("-COORD_TO_SF")
    def coordToSF(self, dimension):
        assert dimension in ["X", "Y", "Z"]
        self.arguments.append(dimension)

    @cc("-FILTER_SF")
    def filterSF(self, minVal, maxVal):
        SPECIAL_WORDS = ["MIN", "DISP_MIN", "SAT_MIN", "MAX", "DISP_MAX", "SAT_MAX"]
        if isinstance(minVal, str):
            assert minVal in SPECIAL_WORDS
        if isinstance(maxVal, str):
            assert maxVal in SPECIAL_WORDS
        self.arguments += [minVal, maxVal]

    @cc("-DENSITY")
    def density(self, sphereRadius, type: DENSITY_TYPE = None):
        self.arguments.append(sphereRadius)
        if type is not None:
            self.arguments += ["-TYPE", type]

    @cc("-APPROX_DENSITY")
    def approxDensity(self, type: DENSITY_TYPE = None):
        if type is not None:
            self.arguments += ["-TYPE", type]

    @cc("-ROUGH")
    def rough(self, kernelSize):
        self.arguments.append(kernelSize)

    @cc("-CURV")
    def curvature(self, type: CURV_TYPE, kernelSize):
        self.arguments += [type, kernelSize]

    @cc("-SF_GRAD")
    def sfGrad(self, euclidian: bool):
        euclidian = BOOL.fromBool(euclidian)
        self.arguments.append(euclidian)

    @cc("-BEST_FIT_PLANE")
    def bestFitPlane(self, makeHoriz=False, keepLoaded=False):
        if makeHoriz:
            self.arguments.append("-MAKE_HORIZ")
        if keepLoaded:
            self.arguments.append("-KEEP_LOADED")

    @cc("-APPLY_TRANS")
    def applyTransformation(self, filename: str or Path):
        self.arguments.append(filename)

    @cc("-MATCH_CENTERS")
    def matchCenters(self):
        pass

    @cc("-DELAUNAY")
    def delaunay(self, xyPlane=False, bestFit=False, maxEdgeLength=None):
        if xyPlane:
            self.arguments.append("-AA")
        if bestFit:
            self.arguments.append("-BEST_FIT")
        if maxEdgeLength is not None:
            self.arguments += ["-MAX_EDGE_LENGTH", maxEdgeLength]

    @cc("-ICP")
    def icp(self, referenceIsFirst=False, minErrorDiff: float = 1e-6, iter: int = None, overlap: int = 100,
            adjustScale=False, randomSamplingLimit=20000, farthestRemoval=False, rot: ROT = None):
        if referenceIsFirst:
            self.arguments.append("-REFERENCE_IS_FIRST")
        if minErrorDiff != 1e-6:
            self.arguments += ["-MIN_ERROR_DIFF", minErrorDiff]
        if iter is not None:
            self.arguments += ["-ITER", iter]
        if overlap != 100:
            self.arguments += ["-OVERLAP", overlap]
        if adjustScale:
            self.arguments.append("-ADJUST_SCALE")
        if randomSamplingLimit != 20000:
            self.arguments += ["-RANDOM_SAMPLING_LIMIT", randomSamplingLimit]
        if farthestRemoval:
            self.arguments.append("-FARTHEST_REMOVAL")
        # TODO: DATA_SF_WEIGHTS
        # TODO: MODEL_SF_WEIGHTS
        if rot is not None:
            self.arguments.append("-ROT".value + rot.value)

    @cc("-CROP")
    def crop(self, xMin, yMin, zMin, xMax, yMax, zMax, outside=False):
        boxString = f"{xMin}:{yMin}:{zMin}:{xMax}:{yMax}:{zMax}"
        self.arguments.append(boxString)
        if outside:
            self.arguments.append("-OUTSIDE")

    @cc("-CROP2D")
    def crop2D(self, orthoDim, *xy, outside=False):
        self.arguments += [orthoDim, len(xy)]
        for x, y in xy:
            self.arguments += [x, y]
        if outside:
            self.arguments.append("-OUTSIDE")

    @cc("-CROSS_SECTION")
    def crossSection(self, xmlFilename: str or Path):
        self.arguments.append(xmlFilename)

    @cc("-SOR")
    def sor(self, numberOfNeighbors, sigma):
        self.arguments += [numberOfNeighbors, sigma]

    @cc("-SF_ARITHMETIC")
    def sfArithmetic(self, sfIndex: int or str, operation: SF_ARITHMETICS):
        if isinstance(sfIndex, str):
            assert sfIndex == "LAST"
        else:
            assert sfIndex >= 0
        self.arguments += [sfIndex, operation]

    @cc("-SF_OP")
    def sfOperation(self, sfIndex, operation: SF_OPERATIONS, value):
        if isinstance(sfIndex, str):
            assert sfIndex == "LAST"
        else:
            assert sfIndex >= 0
        self.arguments += [sfIndex, operation, value]

    @cc("-CBANDING")
    def cBanding(self, dim: str, freq: int):
        assert dim in ["X", "Y", "Z"]
        self.arguments += [dim, freq]

    @cc("-SF_COLOR_SCALE")
    def sfColorScale(self, filename: str or Path):
        self.arguments.append(filename)

    @cc("-SF_CONVERT_TO_RGB")
    def sfConvertToRGB(self, replaceExisting: bool):
        replaceExisting = BOOL.fromBool(replaceExisting)
        self.arguments.append(replaceExisting)

    @cc("-M3C2")
    def m3c2(self, parametersFile: str or Path):
        self.arguments.append(parametersFile)

    @cc("-CANUPO_CLASSIFY")
    def canupoClassify(self, parametersFile: str or Path, useConfidence=None):
        self.arguments.append(parametersFile)
        if useConfidence is not None:
            assert 0 <= useConfidence <= 1
            self.arguments += ["USE_CONFIDENCE", useConfidence]

    @cc("-PCV")
    def pcv(self, nRays=None, isClosed=False, northernHemisphere=False, resolution=None):
        if nRays is not None:
            self.arguments += ["-N_RAYS", nRays]
        if isClosed:
            self.arguments.append("-IS_CLOSED")
        if northernHemisphere:
            self.arguments.append("-NORTHERN_HEMISPHERE")
        if resolution is not None:
            self.arguments += ["-RESOLTION", resolution]

    @cc("-RANSAC")
    def ransac(self, epsilonAbsolute, epsilonPercentageOfScale, bitMap):
        pass  # TODO

    @cc("-C_EXPORT_FMT")
    def cloudExportFormat(self, format: CLOUD_EXPORT_FORMAT, precision=12, separator=SEPARATOR.SPACE, addHeader=False,
                          addPointCount=False, extension: str = None):
        self.arguments.append(format)

        if precision != 12:
            self.arguments += ["-PRECSION", precision]
        if isinstance(separator, str):
            separator = SEPARATOR.fromString(separator)
        if separator != SEPARATOR.SPACE:
            self.arguments += ["-SEPARATOR", separator]
        if addHeader:
            self.arguments.append("-ADD_HEADER")
        if addPointCount:
            self.arguments.append("-ADD_PTS_COUNT")
        if extension is not None:
            self.arguments += ["-EXT", extension]

    @cc("-M_EXPORT_FMT")
    def meshExportFormat(self, format: MESH_EXPORT_FORMAT, extension: str = None):
        self.arguments.append(format)

        if extension is not None:
            self.arguments += ["-EXT", extension]

    @cc("-H_EXPORT_FMT")
    def hierarchyExportFormat(self, format: str = "BIN"):
        """ Mostly the BIN format, but other formats that support a collection of objects might be elligible. """
        self.arguments.append(format)

    @cc("-FBX")
    def fbxExportFormat(self, format: FBX_EXPORT_FORMAT):
        self.arguments += ["-EXPORT_FMT", format]

    @cc("-PLY_EXPORT_FMT")
    def plyExportFormat(self, format: PLY_EXPORT_FORMAT):
        self.arguments.append(format)

    @cc("-NO_TIMESTAMP")
    def noTimestamp(self):
        pass

    @cc("-BUNDLER_IMPORT")
    def bundlerImport(self, filename: str or Path, altKeypoints=None, scaleFactor=None, colorDTM=None, undistort=False):
        self.arguments.append(filename)
        if altKeypoints is not None:
            self.arguments += ["-ALT_KEYPOINTS", altKeypoints]
        if scaleFactor is not None:
            self.arguments += ["-SCALE_FACTOR", scaleFactor]
        if colorDTM is not None:
            self.arguments += ["-COLOR_DTM", colorDTM]
        if undistort:
            self.arguments.append("-UNDISTORT")

    @cc("-DROP_GLOBAL_SHIFT")
    def dropGlobalShift(self):
        pass

    @cc("-SET_ACTIVE_SF")
    def setActiveSF(self, index: int):
        self.arguments.append(index)

    @cc("-REMOVE_ALL_SFS")
    def removeAllSFS(self):
        pass

    @cc("-REMOVE_RGB")
    def removeRGB(self):
        pass

    @cc("-REMOVE_NORMALS")
    def removeNormals(self):
        pass

    @cc("-REMOVE_SCAN_GRIDS")
    def removeScanGrid(self):
        pass

    @cc("-AUTO_SAVE")
    def autoSave(self, onOff: bool):
        onOff = ONOFF.fromBool(onOff)
        self.arguments.append(onOff)

    @cc("-SAVE_CLOUDS")
    def saveClouds(self, *files, allAtOnce=False):
        if allAtOnce:
            self.arguments.append("ALL_AT_ONCE")
        if files:
            self.arguments.append("FILE")
            self.arguments.append('"' + ' '.join(map(str, files)) + '"')

    @cc("-SAVE_MESHES")
    def saveMeshes(self, *files, allAtOnce=False):
        if allAtOnce:
            self.arguments.append("ALL_AT_ONCE")
        if files:
            self.arguments.append("FILE")
            self.arguments.append('"' + ' '.join(map(str, files)) + '"')

    @cc("-CLEAR")
    def clear(self):
        pass

    @cc("-CLEAR_CLOUDS")
    def clearClouds(self):
        pass

    @cc("-CLEAR_MESHES")
    def clearMeshes(self):
        pass

    @cc("-POP_CLOUDS")
    def popClouds(self):
        pass

    @cc("-POP_MESHES")
    def popMeshes(self):
        pass

    @cc("-LOG_FILE")
    def logFile(self, filename: Path or str):
        self.arguments.append(filename)

    # Alias
    def statisticalOutlierRemoval(self, numberOfNeighbors, sigma):
        self.sor(numberOfNeighbors, sigma)
