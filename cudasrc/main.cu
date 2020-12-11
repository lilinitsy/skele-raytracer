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
	//ray_dir = vecmath::normalize(ray_dir);

	Ray ray;
	ray.position  = scene.camera.position;
	ray.direction = ray_dir;


	// the pixel index is y * 1920 since 1920 is the max x value, and then this gets us to the y coordinate
	// and then shift over by x amount to hit pixel (x, y)
	int pixel = y * scene.width + x;

	// initialize the random state for this pixel
	curand_init(5351 * pixel, 0, 0, &random_state[pixel]);

	//image[pixel] = ray_dir;
	image[pixel] = shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces, random_state);
	//image[pixel] = vecmath::vec3(u, v, 1);
	__syncthreads();
}


void generate_rays(Scene scene, Options option, char *output)
{
	scene.width = 1920;
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
	printf("allocated cudascene struct\n");

	//printf("cuda_scene_data vertex[1]: %f %f %f\n", cuda_scene_data.vertices[1].x,cuda_scene_data.vertices[1].y,cuda_scene_data.vertices[1].z);

	// Options that will live on device
	int thread_x = 8;
	int thread_y = 8;

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

	ray_generation<<<blocks, grid>>>(image, cuda_scene_data, option, random_state);
	cudaMemcpy(image_host, image, image_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Got out of ray generation\n");


	// Read back on the host
	std::ofstream ofs(output, std::ios::out | std::ios::binary);
	ofs << "P6\n"
		<< scene.width << " " << scene.height << "\n255\n";

	for(int i = 0; i < scene.height; i++)
	{
		for(int j = 0; j < scene.width; j++)
		{
			int index = i * scene.width + j;


			//printf("pixel[%d]: %f %f %f\n", index, image_host[index].x, image_host[index].y, image_host[index].z);
		

			ofs << (unsigned char) (std::min(float(1), image_host[index].x) * 255) << (unsigned char) (std::min(float(1), image_host[index].y) * 255) << (unsigned char) (std::min(float(1), image_host[index].z) * 255);
		}
	}
	ofs.close();
	printf("***\nWROTE TO PPM\n***\n");

	cudaFree(image);
	printf("max depth: %d\n", option.max_depth);
	printf("OldSCeneStruct num vertices and num triangles: %d %d\n", scene.vertices.size(), scene.triangles.size());
	printf("Host scene struct num vertices and num triangles: %d %d\n", host_cuda_scene.num_vertices, host_cuda_scene.num_triangles);

	for(int i = 0; i < host_cuda_scene.num_spheres; i++)
	{
		host_cuda_scene.spheres[i].to_string();
		printf("\n");
	}

	printf("Shadows: %d\n", host_cuda_scene.use_shadows);
	// make sure to free up the scene







	// Normalize tests
	vecmath::vec3 v1 = vecmath::vec3(1.0f, 1.0f, 1.0f);
	vecmath::vec3 v2 = vecmath::vec3(3.0f, 3.0f, 3.0f);
	vecmath::vec3 v3 = vecmath::vec3(9.4f, 0.3f, -8.4f);
	vecmath::vec3 v4 = vecmath::vec3(0.0f, 0.0f, 1.0f);
	vecmath::vec3 v5 = vecmath::vec3(8.3f, 0.0f, 3.4f);

	float f1 = vecmath::length(v1 - v2);
	float f2 = vecmath::length(v2 - v3);
	float f3 = vecmath::length(v3 - v4);
	float f4 = vecmath::length(v4 - v5);
	float f5 = vecmath::length(v5 - v1);

	printf("f1: %f\n", f1);
	printf("f2: %f\n", f2);
	printf("f3: %f\n", f3);
	printf("f4: %f\n", f4);
	printf("f5: %f\n", f5);
	
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
