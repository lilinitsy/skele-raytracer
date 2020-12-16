#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <time.h>

#include <curand_kernel.h>

#include "raytrace.h"
#include "vec3.h"


void generate_rays(Scene scene, Options option, char *output);


__global__ void ray_generation(vecmath::vec3 *image, CudaScene scene, Options option, curandState *random_state, int* q1array,
	int* q2array, int* q3array, int* q4array, int numq1, int numq2, int numq3, int numq4)
{
	__shared__ struct Sphere qspheres[30];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float inv_width	 = 1.0f / (float) scene.width;
	float inv_height = 1.0f / (float) scene.height;

	float aspect_ratio = (float) scene.width / (float) scene.height;
	float angle		   = tan(M_PI * 0.5f * option.fov / 180.0f);


	if(x >= scene.width || y >= scene.height)
	{
		return;
	}

	int num;
	if(x < scene.width/2 )
	{
		if(y < scene.height/2)
		{
			for(int i=0; i< numq1; i++)
			{
				qspheres[i] = scene.spheres[q1array[i]];
			}
			num = numq1;
			__syncthreads();
		}
		else
		{
			for(int i=0; i< numq3; i++)
			{
				qspheres[i] = scene.spheres[q3array[i]];
			}
			num = numq3;
			__syncthreads();
		}		
	}
	else
	{
		if(y < scene.height/2)
		{
			for(int i=0; i< numq2; i++)
			{
				qspheres[i] = scene.spheres[q2array[i]];
			}
			num = numq2;
			__syncthreads();
		}
		else
		{
			for(int i=0; i< numq4; i++)
			{
				qspheres[i] = scene.spheres[q4array[i]];
			}
			num = numq4;
			__syncthreads();
		}
	}

	// This was the cpu code:
	float u = (2 * ((x + 0.5) * inv_width) - 1) * angle * aspect_ratio;
	float v = (1 - 2 * ((y + 0.5) * inv_height)) * angle;

	vecmath::vec3 ray_dir;
	ray_dir = scene.camera.direction + u * scene.camera.right + v * scene.camera.up;
	//ray_dir.y -= 1.0f; // this makes the ray_direction correct
	ray_dir = vecmath::normalize(ray_dir);

	Ray ray;
	ray.position  = scene.camera.position;
	ray.direction = ray_dir;
	ray.direction = vecmath::normalize(ray.direction);


	// the pixel index is y * width since width is the max x value, and then this gets us to the y coordinate
	// and then shift over by x amount to hit pixel (x, y)
	int pixel = y * scene.width + x;

	// initialize the random state for this pixel
	curand_init(5351 * pixel, 0, 0, &random_state[pixel]);

	image[pixel] = shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces, random_state,
		 qspheres, num);
	__syncthreads(); // can't tell if this is necessary; it might be with shared mem access
}


void generate_rays(Scene scene, Options option, char *output)
{
	scene.width	 = 1920;
	scene.height = 1080;


	// Input Binning Starts

	// With camera.position and camera.up & right, we have two planes Ax+By+Cz+D=0, calculate D first
	// A, B, C are just from up and right vectors (they are normal vectors to the above planes)

	float Dright = -1.0 * vecmath::dot(scene.camera.right, scene.camera.position);
	float lengthright = vecmath::length(scene.camera.right);

	vecmath::vec3 OrthogonalUp = vecmath::cross(-1.0f * scene.camera.right, scene.camera.direction);
	float Dup = -1.0 * vecmath::dot(OrthogonalUp, scene.camera.position);
	float lengthup = vecmath::length(OrthogonalUp);

	int Q1Array[scene.spheres.size()];
	int Q2Array[scene.spheres.size()];
	int Q3Array[scene.spheres.size()];
	int Q4Array[scene.spheres.size()];
	
	int Q1Idx = 0;
	int Q2Idx = 0;
	int Q3Idx = 0;
	int Q4Idx = 0;

	// Then, for each sphere, check two things:
	// 1) its relationship with up and right planes;
	// 2) the distance of its center to these two planes vs. its radius;
	// Quadrant 1=top left; 2=top right; 3=bottom left; 4=bottom right;
	for (unsigned int i = 0; i < scene.spheres.size(); i++)
	{
		float up = vecmath::dot(OrthogonalUp, scene.spheres[i].collider.position) + Dup;
		bool isup = (up >= 0);
		float DistanceToUp = abs(up) / lengthup;
		bool AwayFromUp = (DistanceToUp > scene.spheres[i].collider.radius);

		float right = vecmath::dot(scene.camera.right, scene.spheres[i].collider.position) + Dright;
		bool isright = (right >= 0);
		float DistanceToRight = abs(right) / lengthright;
		bool AwayFromRight = (DistanceToRight > scene.spheres[i].collider.radius);

		if(!AwayFromUp)
		{
			if(!AwayFromRight)
			{
				// In quadrant 1,2,3,4;
				Q1Array[Q1Idx] = i;
				Q2Array[Q2Idx] = i;
				Q3Array[Q3Idx] = i;
				Q4Array[Q4Idx] = i;
				Q1Idx += 1;
				Q2Idx += 1;
				Q3Idx += 1;
				Q4Idx += 1;
			}
			else
			{
				if( isright )
				{
					// In quadrant 2,4;
					Q2Array[Q2Idx] = i;
					Q4Array[Q4Idx] = i;
					Q2Idx += 1;
					Q4Idx += 1;
				}
				else
				{
					// In quadrant 1,3;
					Q1Array[Q1Idx] = i;
					Q3Array[Q3Idx] = i;
					Q1Idx += 1;
					Q3Idx += 1;
				}
			}
		}
		else
		{
			if( isup )
			{
				if(!AwayFromRight)
				{
					// In quadrant 1,2;
					Q1Array[Q1Idx] = i;
					Q2Array[Q2Idx] = i;
					Q1Idx += 1;
					Q2Idx += 1;
				}
				else
				{
					if(isright)
					{
						// In quadrant 2;
						Q2Array[Q2Idx] = i;
						Q2Idx += 1;
					}
					else
					{
						// In quadrant 1;
						Q1Array[Q1Idx] = i;
						Q1Idx += 1;
					}
				}
			}
			else
			{
				if(!AwayFromRight)
				{
					// In quadrant 3,4;
					Q3Array[Q3Idx] = i;
					Q4Array[Q4Idx] = i;
					Q3Idx += 1;
					Q4Idx += 1;
				}
				else
				{
					if(isright)
					{
						// In quadrant 4;
						Q4Array[Q4Idx] = i;
						Q4Idx += 1;
					}
					else
					{
						// In quadrant 3;
						Q3Array[Q3Idx] = i;
						Q3Idx += 1;
					}
				}
			}
		}

	}

	int numq1 = Q1Idx;
	int numq2 = Q2Idx;
	int numq3 = Q3Idx;
	int numq4 = Q4Idx;
	//printf("%d %d %d %d \n", numq1, numq2, numq3, numq4);

	int* d_Q1Array;
	int* d_Q2Array;
	int* d_Q3Array;
	int* d_Q4Array;

	cudaMalloc((void **) &d_Q1Array, numq1 * sizeof(int));
	cudaMalloc((void **) &d_Q2Array, numq2 * sizeof(int));
	cudaMalloc((void **) &d_Q3Array, numq3 * sizeof(int));
	cudaMalloc((void **) &d_Q4Array, numq4 * sizeof(int));
	cudaMemcpy(d_Q1Array, Q1Array, numq1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Q2Array, Q2Array, numq2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Q3Array, Q3Array, numq3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Q4Array, Q4Array, numq4 * sizeof(int), cudaMemcpyHostToDevice);

	// Input Binning Ends

	// The output image that will be written to
	size_t image_size = scene.width * scene.height * sizeof(vecmath::vec3);

	// The image host; will be copied back to CPU to be displayed
	vecmath::vec3 *image_host = (vecmath::vec3 *) malloc(image_size * sizeof(vecmath::vec3));

	// The image device, rendered pixels on gpu
	vecmath::vec3 *image;
	cudaMalloc((void **) &image, image_size);

	// Random state to get CUDA RNG
	curandState *random_state;
	cudaMalloc((void **) &random_state, image_size * sizeof(curandState));

	// Copy the Scene over to a CudaScene that lives on the host
	CudaScene host_cuda_scene = CudaScene(scene);

	// The CudaScene that will be passed to the device.
	CudaScene cuda_scene_data = allocate_device_cudascene_struct(host_cuda_scene);

	// Can test different block sizes; this will give us 16 * 16 = 256 = 32 * 8 warps.
	int thread_x = 16;
	int thread_y = 12;

	dim3 blocks;
	blocks.x = scene.width / thread_x + 1;
	blocks.y = scene.height / thread_y + 1;
	blocks.z = 1;

	dim3 grid;
	grid.x = thread_x;
	grid.y = thread_y;
	grid.z = 1;

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start,0);

	// Copy host memory to device memory
	cudaMemcpy(image, image_host, image_size, cudaMemcpyHostToDevice);

	// Launch kernel
	ray_generation<<<blocks, grid>>>(image, cuda_scene_data, option, random_state, d_Q1Array, d_Q2Array, 
		d_Q3Array, d_Q4Array, numq1, numq2, numq3, numq4);

	// Copy the memory back to the host and then synchronize
	cudaMemcpy(image_host, image, image_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("\nGPU time used:%f ms\n",time);

	// Read back on the host
	std::ofstream ofs(output, std::ios::out | std::ios::binary);
	ofs << "P6\n"
		<< scene.width << " " << scene.height << "\n255\n";

	for(int i = 0; i < scene.height; i++)
	{
		for(int j = 0; j < scene.width; j++)
		{
			// index from the ray generation kernel
			int index = i * scene.width + j;
			ofs << (unsigned char) (std::min(float(1), image_host[index].x) * 255) << (unsigned char) (std::min(float(1), image_host[index].y) * 255) << (unsigned char) (std::min(float(1), image_host[index].z) * 255);
		}
	}
	ofs.close();
	printf("***\nWROTE TO PPM\n***\n");

	cudaFree(image);
	cudaFree(d_Q1Array);
	cudaFree(d_Q2Array);
	cudaFree(d_Q3Array);
	cudaFree(d_Q4Array);

}


int main(int argc, char *argv[])
{

	Options option;
	Scene scene;

	int width  = 1024;
	int height = 768;
	char *path;
	char *output;

	bool output_path_passed = false;
	bool path_passed		= false;
	bool use_shadows;

	for(int i = 0; i < argc; i++)
	{
		if(strcmp(argv[i], "--gillum") == 0)
		{
			if(i + 1 < argc)
			{
				option.monte_carlo	   = true;
				option.num_path_traces = atoi(argv[i + 1]);
			}

			else
			{
				std::cerr << "gillum takes an int after flag for the number of paths traced" << std::endl;
			}
		}

		if(strcmp(argv[i], "--fov") == 0)
		{
			if(i + 1 < argc)
			{
				option.fov = atof(argv[i + 1]);
			}

			else
			{
				std::cerr << "fov takes a float (degrees) after flag for the field of view" << std::endl;
				return 0;
			}
		}

		if(strcmp(argv[i], "--jsample") == 0)
		{
			if(i + 1 < argc)
			{
				option.grid_size = atoi(argv[i + 1]);
			}

			else
			{
				std::cerr << "jsample takes an int after flag for the supersampling grid size" << std::endl;
				return 0;
			}
		}

		if(strcmp(argv[i], "--width") == 0)
		{
			if(i + 1 < argc)
			{
				width = atoi(argv[i + 1]);
			}

			else
			{
				std::cerr << "width takes an int after flag for the width" << std::endl;
				return 0;
			}
		}

		if(strcmp(argv[i], "--height") == 0)
		{
			if(i + 1 < argc)
			{
				height = atoi(argv[i + 1]);
			}

			else
			{
				std::cerr << "height takes an int after flag for the width" << std::endl;
				return 0;
			}
		}

		if(strcmp(argv[i], "--depth") == 0)
		{
			if(i + 1 < argc && atoi(argv[i + 1]) > 0)
			{
				option.max_depth = atoi(argv[i + 1]);
			}

			else
			{
				std::cerr << "depth takes a positive int after flag for the max depth" << std::endl;
				return 0;
			}
		}

		if(strcmp(argv[i], "--parallel") == 0)
		{
			if(i + 1 < argc && strcmp(argv[i + 1], "true") == 0)
			{
				option.visual = false;
			}

			if(i + 1 < argc && strcmp(argv[i + 1], "false") == 0)
			{
				option.visual = true;
			}
		}

		if(strcmp(argv[i], "--path") == 0)
		{
			if(i + 1 < argc)
			{
				path		= argv[i + 1];
				path_passed = true;
			}

			else
			{
				std::cerr << "path must be passed after --path" << std::endl;
				return 0;
			}
		}

		if(strcmp(argv[i], "--output") == 0)
		{
			if(i + 1 < argc)
			{
				output			   = argv[i + 1];
				output_path_passed = true;
			}

			else
			{
				std::cerr << "output path must be passed after --output" << std::endl;
				return 0;
			}
		}

		if(strcmp(argv[i], "--shadow") == 0)
		{
			use_shadows = true;
		}
	}

	if(!path_passed)
	{
		std::cerr << "no scene file was passed. Pass with --path path_to_scn" << std::endl;
		return 0;
	}

	if(!output_path_passed)
	{
		std::cerr << "no output destination was passed. Pass with --output destination_path.ppm" << std::endl;
		return 0;
	}

	scene			  = parseScene(path);
	scene.width		  = width;
	scene.height	  = height;
	scene.use_shadows = use_shadows;

	printf("Above option\n");
	//option.to_string();
	printf("below option\n");
	srand((unsigned) time(0));


	generate_rays(scene, option, output);




	return 0;
}
