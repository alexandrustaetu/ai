#!/usr/bin/env python
from Functions_cl import *
import pyopencl as cl
import numpy
import cv
import OpenGL.GL as gl


stream = cv.CaptureFromCAM(-1)
frame = cv.QueryFrame(stream)
frame_size =  cv.GetSize(frame)
frame_x = frame_size[0]
frame_y = frame_size[1]
img = cv.CreateMat( frame_y,frame_x, cv.CV_8UC4 )

localWorkSize = ( 16, 16 )
globalWorkSize = (round_up(16,frame_x),round_up(16,frame_y))
origin = ( 0, 0, 0 )
region = ( frame_x,frame_y, 1 )
gpu_context = cl.Context(cl.get_platforms()[0].get_devices(cl.device_type.GPU))
command_queue = cl.CommandQueue(gpu_context)
memory_flags = cl.mem_flags
gpu_program = cl.Program(gpu_context, open("canny_edge_detector.cl", 'r').read()).build()
sampler = cl.Sampler(gpu_context, False, cl.addressing_mode.CLAMP_TO_EDGE, cl.filter_mode.NEAREST)

cl_single_chanel_image_format = cl.ImageFormat(cl.channel_order.R,cl.channel_type.UNSIGNED_INT8)
rgba_image_format = cl.ImageFormat(cl.channel_order.RGBA,cl.channel_type.UNSIGNED_INT8)

gray_array = numpy.zeros(frame_x * frame_y, numpy.uint8)
gaussian_array = numpy.zeros(frame_x * frame_y, numpy.uint8)
sobel_array = numpy.zeros(frame_x * frame_y, numpy.uint8)
angles_array = numpy.zeros(frame_x * frame_y, numpy.uint8)
edges_array = numpy.zeros(frame_x * frame_y, numpy.uint8)
thin_edges_array = numpy.zeros(frame_x * frame_y, numpy.uint8)

gray_image = cl.Image(gpu_context, memory_flags.READ_ONLY, cl_single_chanel_image_format, frame_size)
gaussian_image = cl.Image(gpu_context, cl.mem_flags.READ_ONLY, cl_single_chanel_image_format, frame_size)
sobel_image = cl.Image(gpu_context, cl.mem_flags.READ_ONLY, cl_single_chanel_image_format, frame_size)
angles_image = cl.Image(gpu_context, cl.mem_flags.READ_ONLY, cl_single_chanel_image_format, frame_size)
edges_image = cl.Image(gpu_context, cl.mem_flags.READ_ONLY, cl_single_chanel_image_format, frame_size)
thin_edges_image = cl.Image(gpu_context, cl.mem_flags.READ_ONLY, cl_single_chanel_image_format, frame_size)

sobel_mask_x = numpy.array([-1,0,1,-2,0,2,-1,0,1], dtype=numpy.int32)
sobel_mask_y = numpy.array([-1,-2,-1,0,0,0,1,2,1], dtype=numpy.int32)
gaussian_mask = numpy.array([2,4,5,4,2,4,9,12,9,4,5,12,15,12,5,4,9,12,9,4,2,4,5,4,2], dtype=numpy.int32)
sobel_x_buffer = cl.Buffer(gpu_context, memory_flags.READ_ONLY | memory_flags.COPY_HOST_PTR, hostbuf=sobel_mask_x)
sobel_y_buffer = cl.Buffer(gpu_context, memory_flags.READ_ONLY | memory_flags.COPY_HOST_PTR, hostbuf=sobel_mask_y)
gaussian_buffer = cl.Buffer(gpu_context, memory_flags.READ_ONLY | memory_flags.COPY_HOST_PTR, hostbuf=gaussian_mask)

while True:
  frame = cv.QueryFrame(stream)
	cv.ShowImage("camera_window1", frame)
	cv.CvtColor( frame, img, cv.CV_RGB2RGBA)
	frame_string = cv.GetMat(img).tostring()
	clImage = cl.Image(gpu_context,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,rgba_image_format,frame_size,None,frame_string)
	event = gpu_program.convert_to_gray(command_queue,globalWorkSize,localWorkSize,clImage, gray_image, sampler, numpy.int32(frame_x), numpy.int32(frame_y)).wait()
	event2 = gpu_program.apply_gaussian_mask(command_queue,globalWorkSize,localWorkSize,gray_image, gaussian_image,gaussian_buffer, sampler, numpy.int32(frame_x), numpy.int32(frame_y)).wait()
	event2 = gpu_program.apply_sobel_mask(command_queue,globalWorkSize,localWorkSize,gaussian_image, sobel_image,angles_image,sobel_x_buffer,sobel_y_buffer, sampler, numpy.int32(frame_x), numpy.int32(frame_y)).wait()
	event2 = gpu_program.find_edges(command_queue,globalWorkSize,localWorkSize,sobel_image, edges_image,angles_image,sampler, numpy.int32(frame_x), numpy.int32(frame_y)).wait()
	event2 = gpu_program.suppress_edges(command_queue,globalWorkSize,localWorkSize,edges_image,sobel_image,angles_image, thin_edges_image,sampler, numpy.int32(frame_x), numpy.int32(frame_y)).wait()
	event3 = cl.enqueue_read_image(command_queue, thin_edges_image,origin, region, thin_edges_array).wait()
	cv.ShowImage("camera_window3", cv.fromarray(thin_edges_array.reshape(frame_y,frame_x)))
	if cv.WaitKey(10) == 27:
		breakcv.DestroyWindow("camera_window")
