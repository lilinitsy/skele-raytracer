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


__global__ void ray_generation(vecmath::vec3 *image, CudaScene scene, Options option, curandState *random_state)
{
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

	image[pixel] = shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces, random_state);
	__syncthreads(); // can't tell if this is necessary; it might be with shared mem access
}


void generate_rays(Scene scene, Options option, char *output)
{
	scene.width	 = 1920;
	scene.height = 1080;

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



	// Input Binning Starts

	// With camera.position and camera.up & right, we have two planes Ax+By+Cz+D=0, calculate D first
	// A, B, C are just from up and right vectors (they are normal vectors to the above planes)
	float Dup = -1.0 * vecmath::dot(cuda_scene_data.camera.up, cuda_scene_data.camera.position);
	float Dright = -1.0 * vecmath::dot(cuda_scene_data.camera.right, cuda_scene_data.camera.position);
	float lengthup = vecmath::length(cuda_scene_data.camera.up);
	float lengthright = vecmath::length(cuda_scene_data.camera.right);

	// Then, for each sphere, check two things:
	// 1) its relationship with up and right planes;
	// 2) the distance of its center to these two planes vs. its radius;
	// Quadrant 1=top left; 2=top right; 3=bottom left; 4=bottom right;
	// Use prime numbers 2,3,5,7 to index them as a sphere may stay in multiple quadrants;
	for (unsigned int i = 0; i < cuda_scene_data.num_spheres; i++)
	{
		float up = vecmath::dot(cuda_scene_data.camera.up, cuda_scene_data.spheres[i].collider.position) + Dup;
		bool isup = (up >= 0);
		float DistanceToUp = abs(up) / lengthup;
		bool AwayFromUp = (DistanceToUp >= cuda_scene_data.spheres[i].collider.radius);

		float right = vecmath::dot(cuda_scene_data.camera.right, cuda_scene_data.spheres[i].collider.position) + Dright;
		bool isright = (right >= 0);
		float DistanceToRight = abs(right) / lengthright;
		bool AwayFromRight = (DistanceToRight >= cuda_scene_data.spheres[i].collider.radius);

		if(!AwayFromUp)
		{
			if(!AwayFromRight)
			{
				cuda_scene_data.spheres[i].collider.quadrant *= 210; // In quadrant 1,2,3,4;
			}
			else
			{
				if(isright)
				{
					cuda_scene_data.spheres[i].collider.quadrant *= 21; // In quadrant 2,4;
				}
				else
				{
					cuda_scene_data.spheres[i].collider.quadrant *= 10; // In quadrant 1,3;
				}
			}
		}
		else
		{
			if(isup)
			{
				if(!AwayFromRight)
				{
					cuda_scene_data.spheres[i].collider.quadrant *= 6; // In quadrant 1,2;
				}
				else
				{
					if(isright)
					{
						cuda_scene_data.spheres[i].collider.quadrant *= 3; // In quadrant 2;
					}
					else
					{
						cuda_scene_data.spheres[i].collider.quadrant *= 2; // In quadrant 1;
					}
				}
			}
			else
			{
				if(!AwayFromRight)
				{
					cuda_scene_data.spheres[i].collider.quadrant *= 35; // In quadrant 3,4;
				}
				else
				{
					if(isright)
					{
						cuda_scene_data.spheres[i].collider.quadrant *= 7; // In quadrant 4;
					}
					else
					{
						cuda_scene_data.spheres[i].collider.quadrant *= 5; // In quadrant 3;
					}
				}
			}
		}

	}

	// Create arrays for spheres in four quadrants and send them to shared memory later (?);

	// Input Binning Ends

	// Can test different block sizes; this will give us 16 * 16 = 256 = 32 * 8 warps.
	int thread_x = 16;
	int thread_y = 16;

	dim3 blocks;
	blocks.x = scene.width / thread_x + 1;
	blocks.y = scene.height / thread_y + 1;
	blocks.z = 1;

	dim3 grid;
	grid.x = thread_x;
	grid.y = thread_y;
	grid.z = 1;

	// Copy host memory to device memory
	cudaMemcpy(image, image_host, image_size, cudaMemcpyHostToDevice);

	// Launch kernel
	ray_generation<<<blocks, grid>>>(image, cuda_scene_data, option, random_state);

	// Copy the memory back to the host and then synchronize
	cudaMemcpy(image_host, image, image_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

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
