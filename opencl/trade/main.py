import pyopencl as cl
import numpy
import csv

workGroupSize = 64
workItemSize = 4096 * workGroupSize

def loadData():
    f = open("data.txt", "r")
    csvData = csv.DictReader(f)
    closeData = []
    for idx, row in enumerate(csvData):
        closeData.append(int(float(row["Close"]) * 100))

    print ("Number of Row : %d " %(len(closeData)))
    f.close()
    for i in range(0, 5):
        print("#", i, ": ", closeData[i])
    return closeData;

def runAtDevice(device, data):
    # if "Intel" in device.name:
    #     return
    print(device.name)
    print("-" * 40)
    bars = numpy.int32(data)
    values = numpy.zeros(workItemSize, dtype=numpy.int32)
    # prepare context
    ctx = cl.Context([device])
    # prepare queue
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    # prepare memory
    mf = cl.mem_flags
    dev_bars = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bars)
    dev_values = cl.Buffer(ctx, mf.WRITE_ONLY, values.nbytes)
    # read kernel code
    f = open("ma-cross-int.c", "r")
    fstr = "".join(f.readlines())
    f.close()
    # compile kernel code
    prg = cl.Program(ctx, fstr).build();

    global_size=(workItemSize,)
    local_size=(workGroupSize,)
    preferred_multiple = cl.Kernel(prg, 'maCross').get_work_group_info( \
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, \
        device)

    print("Preferred work group size multiple:", preferred_multiple)

    exec_evt = prg.maCross(queue, global_size, local_size, dev_bars, numpy.int32(len(data)),
                           dev_values, numpy.int32(workItemSize))
    exec_evt.wait()
    cl.enqueue_read_buffer(queue, dev_values, values).wait()
    elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

    print("Execution time of test: %g s" % elapsed)

    for i in range(0, 5):
        print("#", i, ": ", values[i] / 100)
    print("...")
    for i in range(workItemSize - 5, workItemSize):
        print("#", i, ": ", values[i] / 100)

def runAtPlatform(platform, data):
    print(platform.name)
    print("=" * 40)
    for device in platform.get_devices():
        runAtDevice(device, data)

print("start to load data")
data = loadData()
print("start to run at OpenCL")
for platform in cl.get_platforms():
    runAtPlatform(platform, data)
