# CloudCompare CLI Wrapper
This is a simple interface for CloudCompare based on [Command line mode Wiki Page](https://www.cloudcompare.org/doc/wiki/index.php?title=Command_line_mode).

This library works by opening a new subprocess with a command you build step by step.  
When building is complete, the command has to be executed. 

## Requirements
* Python3.6+
* Installed version of [CloudCompare](https://cloudcompare.org/)

## Basic Usage
Read ply-file and save in ascii-format with extension ".xyz". Disable console

````python
import pyCloudCompare as cc

cmd = cc.CloudCompareCMD()
cmd.silent()  # Disable console
cmd.open("pointcloud.ply")  # Read file
cmd.cloudExportFormat(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
cmd.saveClouds("newPointcloud.xyz")
print(cmd)
cmd.execute()
````