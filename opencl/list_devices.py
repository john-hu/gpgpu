import pyopencl as cl
import inspect
import sys

def dump_device(device):
    print("Device name: " + device.name)
    print("Device type: " + cl.device_type.to_string(device.type))
    print("Device version: " + device.version)
    print("Device memory: " + str(device.global_mem_size // 1024 // 1024) + "MB")
    print("Device max clock speed: " + str(device.max_clock_frequency) + "MHz")
    print("Device compute units: " + str(device.max_compute_units))
    print("Device max work group size: " + str(device.max_work_group_size))
    print("Device max work item sizes: " + str(device.max_work_item_sizes))
    print("Device local memory size: " + str(device.local_mem_size // 1024) + "KB")
    print("Device availability: " + ("True" if device.available == 1 else "False"))
    print("Device execution capabilities: " + str(device.execution_capabilities))
    print("Device min data type align size: " + str(device.min_data_type_align_size))
    # print("Device preferred work group size multiple: ", device.preferred_work_group_size_multiple)

    '''
    for creating better workgroup size, we may need to know more information from device info, a
    good post can be found at:
    http://stackoverflow.com/questions/17961331/opencl-are-work-group-axes-exchangeable/17968149#17968149
    '''

def dump_platform(platform):
    print("platform NAME   : %s" % (platform.name))
    print("         VERSION: %s" % (platform.version))
    print("         PROFILE: %s" % (platform.profile))
    print("         VENDOR : %s" % (platform.vendor))

    for device in platform.get_devices():
        print("Device: " + "-" * 32)
        dump_device(device)

def main():
    for platform in cl.get_platforms():
        print("=" * 40)
        dump_platform(platform)


if __name__ == '__main__':
    main()
