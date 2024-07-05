# CloudCompare CLI Python Wrapper
This is a python wrapper for CloudCompare CLI based on this [Wiki Page](https://www.cloudcompare.org/doc/wiki/index.php?title=Command_line_mode).

You can build and chain commands which can be executed. 

## Installation
This package is uploaded to [pypi](https://pypi.org/project/pyCloudCompareCLI/), so you can install it with pip:
```
pip install pyCloudCompareCLI
```

Otherwise you can just install it from source using the [Github-Repository](https://github.com/dwoiwode/pyCloudCompareCLI/).

## Requirements
* Python3.6+
* An installed version of [CloudCompare](https://cloudcompare.org/)

## Basic Usage
Read ply-file and save in ascii-format with extension ".xyz".

````python
import pyCloudCompare as cc

cli = cc.CloudCompareCLI()
cmd = cli.new_command()
cmd.silent()  # Disable console
cmd.open("pointcloud.ply")  # Read file
cmd.cloud_export_format(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
cmd.save_clouds("newPointcloud.xyz")
print(cmd)
cmd.execute()
````

Same example with Context-Manager:

````python
import pyCloudCompare as cc

cli = cc.CloudCompareCLI()
with cli.new_command() as cmd:
    cmd.silent()  # Disable console
    cmd.open("pointcloud.ply")  # Read file
    cmd.cloud_export_format(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
    cmd.save_clouds("newPointcloud.xyz")
    print(cmd)
````

## Acknowledgements
The work in the scope of the CloudCompare CLI Python Wrapper in this repository is supported by the [Institute of Geo-Engineering at Clausthal University of Technology](https://www.ige.tu-clausthal.de).
