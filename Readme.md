# CloudCompare CLI Python Wrapper
This is a python wrapper for CloudCompare CLI based on this [Wiki Page](https://www.cloudcompare.org/doc/wiki/index.php?title=Command_line_mode).

You can build and chain commands which can be executed. 

## Requirements
* Python3.6+
* An installed version of [CloudCompare](https://cloudcompare.org/)

## Basic Usage
Read ply-file and save in ascii-format with extension ".xyz".

````python
import pyCloudCompare as cc

cmd = cc.CloudCompareCLI()
cmd.silent()  # Disable console
cmd.open("pointcloud.ply")  # Read file
cmd.cloudExportFormat(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
cmd.saveClouds("newPointcloud.xyz")
print(cmd)
cmd.execute()
````

Same example with Context-Manager: 
````python
import pyCloudCompare as cc

with cc.CloudCompareCLI() as cmd:
    cmd.silent()  # Disable console
    cmd.open("pointcloud.ply")  # Read file
    cmd.cloudExportFormat(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
    cmd.saveClouds("newPointcloud.xyz")
    print(cmd)
````

## Acknowledgements
The work in the scope of the CloudCompare CLI Python Wrapper in this repository is supported by the [Institute of Geo-Engineering at Clausthal University of Technology](https://www.ige.tu-clausthal.de).
