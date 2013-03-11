import pyopencl as cl
import Image # Python Image Library (PIL)
import numpy
import cv

kernel_file = "camera3_kernel"
gpu_context = cl.Context(cl.get_platforms()[0].get_devices(cl.device_type.GPU))
gpu_program = cl.Program(gpu_context, open(kernel_file, 'r').read()).build()
stream = cv.CaptureFromCAM(-1)
frame = cv.QueryFrame(stream)
cv.NamedWindow("camera_window", 1)
final = numpy.zeros(frame.width*frame.height*3, dtype=numpy.int32)
memory_flags = cl.mem_flags
command_queue = cl.CommandQueue(gpu_context)
contour_buffer = cl.Buffer(gpu_context, memory_flags.WRITE_ONLY, final.nbytes)
while True:
  frame = cv.GetMat(cv.QueryFrame(stream))
	frame = numpy.array(numpy.asarray(frame),dtype=numpy.int32)
	frame = frame.flatten()
	image_buffer = cl.Buffer(gpu_context, memory_flags.READ_ONLY | memory_flags.COPY_HOST_PTR, hostbuf=frame)
	gpu_program.calculate_differences(command_queue, frame.shape, None,image_buffer,contour_buffer)
	cl.enqueue_read_buffer(command_queue, contour_buffer, final).wait()
	img = numpy.uint8(final.reshape(480,640,3))
	img = cv.fromarray(img)
	cv.ShowImage("camera_window", img)
	if cv.WaitKey(10) == 27:
		breakcv.DestroyWindow("camera_window")
