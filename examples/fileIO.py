import pyCloudCompare as cc

cmd = cc.CloudCompareCLI()
cmd.silent()
cmd.open("pointcloud.ply")
cmd.cloudExportFormat(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
cmd.noTimestamp()
cmd.saveClouds("test.xyz")
print(cmd)
cmd.execute()
