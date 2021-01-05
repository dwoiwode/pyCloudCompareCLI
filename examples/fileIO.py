import pyCloudCompare as cc

with cc.CloudCompareCLI() as cmd:
    cmd.silent()
    cmd.open("test_data/pointcloud.ply")
    cmd.cloudExportFormat(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
    cmd.noTimestamp()
    cmd.saveClouds("pointcloud.xyz")
    print(cmd)
