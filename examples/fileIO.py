import pyCloudCompare as cc

cmd = cc.CloudCompareCMD()
cmd.silent()
cmd.open("pointcloud.ply")
cmd.cloudExportFormat(cc.CLOUD_EXPORT_FORMAT.ASC, extension="xyz")
cmd.noTimestamp()
cmd.saveClouds("test.xyz")
print(cmd)
cmd.execute()
