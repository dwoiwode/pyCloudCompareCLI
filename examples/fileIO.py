import pyCloudCompare as cc

cli = cc.CloudCompareCLI()
with cli.new_command() as cmd:
    cmd.silent()
    cmd.open("test_data/pointcloud.ply")
    cmd.cloud_export_format(cc.CLOUD_EXPORT_FORMAT.ASCII, extension="xyz")
    cmd.no_timestamp()
    cmd.save_clouds("pointcloud.xyz")
    print(cmd)
