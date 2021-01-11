import pyCloudCompare as cc

cli = cc.CloudCompareCLI()
with cli.newCommand() as cmd:
    cmd.silent()
    cmd.open("test_data/pointcloud.ply")
    cmd.cloudExportFormat(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
    cmd.noTimestamp()
    cmd.saveClouds("pointcloud.xyz")
    print(cmd)
