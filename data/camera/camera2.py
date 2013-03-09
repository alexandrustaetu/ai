import pyopencl as cl
import numpy
import cv
import Image
import inspect
from time import gmtime, strftime

#detect margins, with opencl

class VISION:
	def __init__(self):
		
		#create variables for videocard
		self.gpu_context = cl.create_some_context()
		self.command_queue = cl.CommandQueue(self.gpu_context)
		self.memory_flags = cl.mem_flags
		
		#create window for frames and camera
		cv.NamedWindow("camera_window", 1)

	def prepare_environment(self, filename,camera_index):
		
		#build kernel for videocard
		kernel_file = open(filename, 'r')
		kernel_string = "".join(kernel_file.readlines())
		self.program = cl.Program(self.gpu_context, kernel_string).build()
		#get frames from the webcam
		self.stream = cv.CaptureFromCAM(camera_index)
		self.line_cols = cv.GetMat(cv.QueryFrame(self.stream)).cols
		while True:
			self.frame = cv.QueryFrame(self.stream)
			self.frame = cv.GetMat(self.frame)
			self.image_data = numpy.asarray(self.frame)
			self.image_data = numpy.array(self.image_data, dtype=numpy.int32)
			final = numpy.zeros(shape=(self.image_data.shape))
			for position,line in enumerate(self.image_data):
				if position == 0:
					continue
				if position == self.image_data.shape[0]-1:
					continue
				line = line.ravel()
				self.line_buffer = cl.Buffer(self.gpu_context, self.memory_flags.READ_ONLY | self.memory_flags.COPY_HOST_PTR, hostbuf=line)
				self.top_line_buffer = cl.Buffer(self.gpu_context, self.memory_flags.READ_ONLY | self.memory_flags.COPY_HOST_PTR, hostbuf=self.image_data[position-1])
				self.bottom_line_buffer = cl.Buffer(self.gpu_context, self.memory_flags.READ_ONLY | self.memory_flags.COPY_HOST_PTR, hostbuf=self.image_data[position+1])
				self.contour_buffer = cl.Buffer(self.gpu_context, self.memory_flags.WRITE_ONLY, line.nbytes)
				self.program.calculate_differences(self.command_queue, line.shape, None,self.top_line_buffer,self.line_buffer,self.bottom_line_buffer, self.contour_buffer)
				contour = numpy.empty_like(line)
				cl.enqueue_read_buffer(self.command_queue, self.contour_buffer, contour).wait()
				line = contour.reshape(self.line_cols,3)
				final[position] = line
			img = numpy.uint8(final)
			img = cv.fromarray(img)
			cv.ShowImage("camera_window", img)
			if cv.WaitKey(10) == 27:
				breakcv.DestroyWindow("camera_window")
		
if __name__ == "__main__":
	vision = VISION()
	vision.prepare_environment("kernel_file",-1)


