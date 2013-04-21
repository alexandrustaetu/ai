
__kernel void convert_to_gray(__read_only image2d_t input, __write_only image2d_t output, sampler_t sampler, unsigned int width, unsigned int height){
unsigned int x = get_global_id(0);
unsigned int y = get_global_id(1);
int4 pixel = read_imagei (input,sampler,(int2)(x,y));
int gray = pixel.s1*0.3 + pixel.s2*0.39 + pixel.s3*0.11;
write_imagei(output,(int2)(x,y), (int4)(gray,0,0,0));
}

__kernel void apply_gaussian_mask(__read_only image2d_t gray_image, __write_only image2d_t gaussian_image,__global int* gaussian_mask, sampler_t sampler, unsigned int width, unsigned int height){
  unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	if (x < 2 || y < 2){
		write_imagei(gaussian_image,(int2)(x,y), (int4)(1,0,0,0));
	}
	else if(x>width-2 || y > height-2){
		write_imagei(gaussian_image,(int2)(x,y), (int4)(3,0,0,0));
	}else{
		int gaussian = 0;
		for (int i = -2 ; i <= 2 ;i++ ){
			for (int j = -2 ; j <= 2 ;j++ ){
				gaussian += read_imagei (gray_image,sampler,(int2)(x-i,y-j)).s0*gaussian_mask[5*(i+2)+(j+2)];
			}
		}
		gaussian = gaussian/159;
		write_imagei(gaussian_image,(int2)(x,y), (int4)((int)gaussian,0,0,0));
	}
}

__kernel void apply_sobel_mask(__read_only image2d_t gaussian_image, __write_only image2d_t sobel_image, __write_only image2d_t angles_image,__global int* sobel_mask_x,__global int* sobel_mask_y, sampler_t sampler, unsigned int width, unsigned int height){
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	if (x < 2 || y < 2){
		write_imagei(sobel_image,(int2)(x,y), (int4)(1,0,0,0));
	}else if(x>width-2 || y > height-2){
		write_imagei(sobel_image,(int2)(x,y), (int4)(3,0,0,0));
	}else{
		int sobel_x = 0;
		int sobel_y = 0;
		for (int i = -1 ; i <= 1 ;i++ ){
			for (int j = -1 ; j <= 1 ;j++ ){
				sobel_x += read_imagei (gaussian_image,sampler,(int2)(x-i,y-j)).s0*sobel_mask_x[3*(i+1)+(j+1)];
				sobel_y += read_imagei (gaussian_image,sampler,(int2)(x-i,y-j)).s0*sobel_mask_y[3*(i+1)+(j+1)];
			}
		}
		int sobel = sqrt((float)sobel_x*(float)sobel_x+(float)sobel_y*(float)sobel_y);
		float angle = atan((float)(sobel_y/sobel_x));
		angle = (angle*180)/M_PI;
		if (angle > -22.5 & angle < 22.5){
			write_imagei(angles_image,(int2)(x,y), (int4)(0,0,0,0));
		}else if (angle < -67.5 || angle > 67.5){
			write_imagei(angles_image,(int2)(x,y), (int4)(90,0,0,0));
		}else if (angle > -67.5 & angle < -22.5){
			write_imagei(angles_image,(int2)(x,y), (int4)(135,0,0,0));
		}else if( angle > 22.5 & angle < 67.5){
			write_imagei(angles_image,(int2)(x,y), (int4)(45,0,0,0));
		}
		write_imagei(gaussian_image,(int2)(x,y), (int4)((int)sobel,0,0,0));
	}
}

__kernel void find_edges(__read_only image2d_t sobel_image, __write_only image2d_t edges_image, __read_only image2d_t angles_image, sampler_t sampler, unsigned int width, unsigned int height){
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	if (x < 2 || y < 2){
		write_imagei(edges_image,(int2)(x,y), (int4)(1,0,0,0));
	}else if(x>width-2 || y > height-2){
		write_imagei(edges_image,(int2)(x,y), (int4)(3,0,0,0));
	}else{
		int current_sobel = read_imagei (sobel_image,sampler,(int2)(x,y)).s0;
		int current_angle = read_imagei (angles_image,sampler,(int2)(x,y)).s0;
		if (current_sobel > 30){
			if (current_angle == 45){
				int new_x = x;
				int new_y = y;
				while(read_imagei (angles_image,sampler,(int2)(new_x,new_y)).s0 == 45 & read_imagei (sobel_image,sampler,(int2)(new_x,new_y)).s0 > 10){
					new_x = new_x+1;
					new_y = new_y-1;
					write_imagei(edges_image,(int2)(new_x,new_y), (int4)(255,0,0,0));
				}
			}else if (current_angle == 135){
				int new_x = x;
				int new_y = y;
				while(read_imagei (angles_image,sampler,(int2)(new_x,new_y)).s0 == 135 & read_imagei (sobel_image,sampler,(int2)(new_x,new_y)).s0 > 10){
					new_x = new_x+1;
					new_y = new_y+1;
					write_imagei(edges_image,(int2)(new_x,new_y), (int4)(255,0,0,0));
				}
			}else if (current_angle == 90){
				int new_x = x;
				int new_y = y;
				while(read_imagei (angles_image,sampler,(int2)(new_x,new_y)).s0 == 90 & read_imagei (sobel_image,sampler,(int2)(new_x,new_y)).s0 > 10){
					new_x = new_x;
					new_y = new_y+1;
					write_imagei(edges_image,(int2)(new_x,new_y), (int4)(255,0,0,0));
				}
			}else if (current_angle == 0){
				int new_x = x;
				int new_y = y;
				while(read_imagei (angles_image,sampler,(int2)(new_x,new_y)).s0 == 0 & read_imagei (sobel_image,sampler,(int2)(new_x,new_y)).s0 > 10){
					new_x = new_x+1;
					new_y = new_y;
					write_imagei(edges_image,(int2)(new_x,new_y), (int4)(255,0,0,0));
				}
			}
		}else{
			write_imagei(edges_image,(int2)(x,y), (int4)(0,0,0,0));
		}
	}
}

__kernel void suppress_edges(__read_only image2d_t edges_image, __read_only image2d_t sobel_image, __read_only image2d_t angles_image, __write_only image2d_t thin_edges_image, sampler_t sampler, unsigned int width, unsigned int height){
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	write_imagei(thin_edges_image,(int2)(x,y), (int4)(0,0,0,0));
	if (x < 2 || y < 2){
		write_imagei(thin_edges_image,(int2)(x,y), (int4)(1,0,0,0));
	}else if(x>width-2 || y > height-2){
		write_imagei(thin_edges_image,(int2)(x,y), (int4)(3,0,0,0));
	}else{
		if(read_imagei (angles_image,sampler,(int2)(x,y)).s0 == 0){
			if(read_imagei (edges_image,sampler,(int2)(x,y)).s0 == 255 & read_imagei (sobel_image,sampler,(int2)(x,y+1)).s0 < read_imagei (sobel_image,sampler,(int2)(x,y)).s0 & read_imagei (sobel_image,sampler,(int2)(x,y-1)).s0 < read_imagei (sobel_image,sampler,(int2)(x,y)).s0){
				write_imagei(thin_edges_image,(int2)(x,y), (int4)(255,0,0,0));
			} 
		}else if(read_imagei (angles_image,sampler,(int2)(x,y)).s0 == 90){
			if(read_imagei (edges_image,sampler,(int2)(x,y)).s0 == 255 & read_imagei (sobel_image,sampler,(int2)(x+1,y)).s0 < read_imagei (sobel_image,sampler,(int2)(x,y)).s0 & read_imagei (sobel_image,sampler,(int2)(x-1,y)).s0 < read_imagei (sobel_image,sampler,(int2)(x,y)).s0 ){
				write_imagei(thin_edges_image,(int2)(x,y), (int4)(255,0,0,0));
			} 
		}else if(read_imagei (angles_image,sampler,(int2)(x,y)).s0 == 45){
			if(read_imagei (edges_image,sampler,(int2)(x,y)).s0 == 255 & read_imagei (sobel_image,sampler,(int2)(x+1,y+1)).s0 < read_imagei (sobel_image,sampler,(int2)(x,y)).s0 & read_imagei (sobel_image,sampler,(int2)(x-1,y-1)).s0 < read_imagei (sobel_image,sampler,(int2)(x,y)).s0 ){
				write_imagei(thin_edges_image,(int2)(x,y), (int4)(255,0,0,0));
			} 
		}else if(read_imagei (angles_image,sampler,(int2)(x,y)).s0 == 135){
			if(read_imagei (edges_image,sampler,(int2)(x,y)).s0 == 255 & read_imagei (sobel_image,sampler,(int2)(x+1,y-1)).s0 < read_imagei (sobel_image,sampler,(int2)(x,y)).s0 & read_imagei (sobel_image,sampler,(int2)(x-1,y+1)).s0 < read_imagei (sobel_image,sampler,(int2)(x,y)).s0 ){
				write_imagei(thin_edges_image,(int2)(x,y), (int4)(255,0,0,0));
			} 
		}else{
			write_imagei(thin_edges_image,(int2)(x,y), (int4)(0,0,0,0));
		}
	}
}
