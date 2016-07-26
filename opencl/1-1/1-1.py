import pyopencl as cl
import numpy

def runAtDevice(device):
    print(device.name)
    print("-" * 40)
    matrix = numpy.float32(list(range(0, 32, 2)))
    vec = numpy.float32([0, 0, 0, 0])
    correct = numpy.float32([0, 0, 0, 0])
    for i in range(0, 4):
        vec[i] = i * 3
        correct[0] += matrix[i] * vec[i]
        correct[1] += matrix[i + 4] * vec[i]
        correct[2] += matrix[i + 8] * vec[i]
        correct[3] += matrix[i + 12] * vec[i]

    # prepare context
    ctx = cl.Context([device])
    # prepare queue
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    # prepare memory
    mf = cl.mem_flags
    dev_matrix = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix)
    dev_vec = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vec)
    dev_correct = cl.Buffer(ctx, mf.WRITE_ONLY, correct.nbytes)
    # read kernel code
    f = open("1-1.c", "r")
    fstr = "".join(f.readlines())
    f.close()
    # compile kernel code
    prg = cl.Program(ctx, fstr).build();

    global_size=(16,)
    local_size=(4,)
    preferred_multiple = cl.Kernel(prg, 'matvec_mult').get_work_group_info( \
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, \
        device)

    print("Preferred work group size multiple:", preferred_multiple)

    exec_evt = prg.matvec_mult(queue, global_size, local_size, dev_matrix, dev_vec, dev_correct)
    exec_evt.wait()
    elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

    print("Execution time of test: %g s" % elapsed)

    result = numpy.empty_like(correct)
    cl.enqueue_read_buffer(queue, dev_correct, result).wait()
    equal = numpy.all(correct == result)
    if not equal:
        print("Results doesn't match!!")
    else:
        print("Results OK")


def runAtPlatform(platform):
    print(platform.name)
    print("=" * 40)
    for device in platform.get_devices():
        runAtDevice(device)

for platform in cl.get_platforms():
    runAtPlatform(platform)
