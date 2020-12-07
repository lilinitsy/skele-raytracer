#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <time.h>

#include "raytrace.h"
#include "vec3.h"

void generate_rays(Scene scene, Options option, char *output);


__global__ void ray_generation(vecmath::vec3 *image, Scene **scene)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x >= 1920 || y >= 1080)
	{
		return;
	}

	// the pixel index is y * 1920 since 1920 is the max x value, and then this gets us to the y coordinate
	// and then shift over by x amount to hit pixel (x, y)
	int pixel = y * 1920 + x;

	image[pixel] = vecmath::vec3((float) x / 1920.0f, (float) y / 1920.0f, 0.1f);
}


void generate_rays(Scene scene, Options option, char *output)
{
	// The output image that will be written to
	size_t image_size = scene.width * scene.height * sizeof(vecmath::vec3);

	vecmath::vec3 *image_host = (vecmath::vec3*) malloc(image_size * sizeof(vecmath::vec3));

	vecmath::vec3 *image;
	cudaMalloc((void**) &image, image_size);

	Scene **cuda_scene_data;
	cudaMalloc((void**) &cuda_scene_data, scene.size() * sizeof(Scene*));
	printf("sizeof scene: %lu\n", sizeof(scene));
	printf("Size of this scene: %lu\n", scene.size());

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

	cudaMemcpy(image, image_host, image_size, cudaMemcpyHostToDevice);

	ray_generation<<<blocks, grid>>>(image, cuda_scene_data);
	printf("Got out\n");
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
			int index = i * scene.width + j;
			printf("pixel[%d]: %f %f %f\n", index, image_host[index].x, image_host[index].y, image_host[index].z);
			ofs << (unsigned char) (std::min(float(1), image_host[index].x) * 255) << (unsigned char) (std::min(float(1), image_host[index].y) * 255) << (unsigned char) (std::min(float(1), image_host[index].z) * 255);

		}
	}
	ofs.close();
	printf("***\nWROTE TO PPM\n***\n");

}

/*

	///////////////// Code for ray generation, can call this kernel "main_renderer" or something
	// Code for without a grid_size operates similarily without r.
	if(option.grid_size > 0)
	{
		for(int i = 0; i < option.grid_size; i++)
		{
			for(int j = 0; j < option.grid_size; j++)
			{
				// r adds some jitter to the ray that we're going to cast
				// u and v are basically the x / y coordinates transformed by the angle (fov basically) and the screen's aspect ratio
				float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
				float u = (2 * ((x + r) * inv_width) - 1) * angle * aspect_ratio;
				float v = (1 - 2 * ((y + r) * inv_height)) * angle;

				// Create the ray's direction vector as a combination of where our camera is looking, and the u & v pixel offsets
				// This will cast out to all pixels in the screen as these loops iterate and construct new u's and v's
				vecmath::vec3 ray_dir(scene.camera.direction + u * scene.camera.right + v * scene.camera.up);
				vecmath::normalize(ray_dir);

				Ray ray;
				ray.position  = scene.camera.position;
				ray.direction = ray_dir;

				// Iteratively add the results to shade for each grid computation to [y][x] in image
				image[y][x] += shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces);
			}
		}
		image[y][x] /= (option.grid_size * option.grid_size);
	}

	else
	{
		float u = (2 * ((x + 0.5) * inv_width) - 1) * angle * aspect_ratio;
		float v = (1 - 2 * ((y + 0.5) * inv_height)) * angle;

		vecmath::vec3 ray_dir(scene.camera.direction + u * scene.camera.right + v * scene.camera.up);
		vecmath::normalize(ray_dir);

		Ray ray;
		ray.position  = scene.camera.position;
		ray.direction = ray_dir;

		// Output the results of shade to the image at index [y][x]
		image[y][x] = shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces);
	}

	// Definitions for what ray corresponds to what pixel
	float inv_width	   = 1 / float(scene.width);
	float inv_height   = 1 / float(scene.height);
	float aspect_ratio = scene.width / float(scene.height);
	float angle		   = tan(M_PI * 0.5 * option.fov / 180.0f);



	std::ofstream ofs(output, std::ios::out | std::ios::binary);
	ofs << "P6\n"
		<< scene.width << " " << scene.height << "\n255\n";

	for(int i = 0; i < scene.height; i++)
	{
		for(int j = 0; j < scene.width; j++)
		{
			ofs << (unsigned char) (std::min(float(1), image[i][j].x) * 255) << (unsigned char) (std::min(float(1), image[i][j].y) * 255) << (unsigned char) (std::min(float(1), image[i][j].z) * 255);
		}
	}

	ofs.close();

	printf("***\nWROTE TO PPM\n***\n");

	delete[] image;
}*/


int main(int argc, char *argv[])
{

	Options option;
	Scene scene;

	int width  = scene.width;
	int height = scene.height;

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

	option.to_string();

	srand((unsigned) time(0));


	generate_rays(scene, option, output);




	return 0;
}
