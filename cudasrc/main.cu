#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <time.h>

#include <curand_kernel.h>

#include "raytrace.h"
#include "vec3.h"
#include "camera.h"

//__constant__  CudaScene scene;
#define BLOCK_SIZEX 32
#define BLOCK_SIZEy 32

void generate_rays(Scene scene, Options option, char *output);
__constant__ int w;
__constant__ int h;
//__constant__ Camera camera;


__global__ void shade(vecmath::vec3 *image,vecmath::vec3 *raydir, vecmath::vec3 pos,int nums,CudaScene scene, int depth, bool monte_carlo, short num_path_traces, curandState *random_state)
{    
	__shared__ int min_distance[32][32];// need to be change
	__shared__ Sphere spheres[100];
	// there are num_of_sphere* number of pixels thread, (thread 0,1,2,3,4,num) compute collisiton for each different object.
	int x = (threadIdx.x + blockIdx.x * blockDim.x)/nums; 
	int i=(threadIdx.x + blockIdx.x * blockDim.x)%nums;//correspons to object i;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= w || y >= h)
	{
		return;
	}
	
	if(blockIdx.x<nums){	
	spheres[i]=scene.spheres[i];
	}
	int pixelx=((threadIdx.x + blockIdx.x * blockDim.x)/nums)%32;
	if(i==0){
		min_distance[pixelx][threadIdx.y]=INT_MAX;
	}
	__syncthreads();

	int pixel = y * w + x;
	//float min_distance = INFINITY;
	Sphere intersected_sphere;
	bool hit_a_sphere = false;
	Ray currentray;
	currentray.position=pos;
	currentray.direction=raydir[pixel];

//	printf("cureent ray% f %f %f",currentray.position.x,currentray.position.y,currentray.position.z);

	Triangle intersected_triangle;
	bool hit_a_triangle = false;
	float distance=INFINITY;
	int num_of_sphere=scene.num_spheres;
	// find minimun distance for each object
	if(i<num_of_sphere){
	//printf("distance%f",distance);
	if(intersection_occurs(currentray,spheres[i].collider)){
	hit_a_sphere   = true;
	distance = collision_distance(currentray, spheres[i].collider);
	distance*=1000000;
	atomicMin(&min_distance[pixelx][threadIdx.y],int(distance));
 	}
   }else{//traiangle
		float t;
		float u;
		float v;
		if(triangle_intersection_occurs(currentray, scene.triangles[i-num_of_sphere], t, u, v))
		{
			if(t < min_distance[pixelx][threadIdx.y])
			{
				min_distance[pixelx][threadIdx.y]		 = t;
				hit_a_sphere		 = false;
				hit_a_triangle		 = true;
				intersected_triangle = scene.triangles[i];
			}
		}
	}
	__syncthreads(); 
	// If no sphere was hit, then the background was hit, so return that colour
	if(min_distance[pixelx][threadIdx.y]==INFINITY)
	{
		image[pixel]=scene.background;
		return;
	}
	__syncthreads(); 
	if(hit_a_sphere&&int(distance)==min_distance[pixelx][threadIdx.y])
	{
		// ray.direction is the SAME
		// ray.position is the SAME
		intersected_sphere =spheres[i];
		vecmath::vec3 e_c = currentray.position - intersected_sphere.collider.position; // e_c matches to e_c in cpu
		float a			  = vecmath::dot(currentray.direction, currentray.direction);
		float b			  = 2 * vecmath::dot(currentray.direction, e_c);
		float c			  = vecmath::dot(e_c, e_c) - intersected_sphere.collider.radius * intersected_sphere.collider.radius;
		float t			  = smallest_root(a, b, c); // t's match

		// Calculating the intersection point and the normal; the normal doesn't really need to be calculated seperately
		vecmath::vec3 intersection_point				   = currentray.position + currentray.direction * t; // ray_direction * t is the same, verified by a diff, and so is ray.position
		vecmath::vec3 intersection_point_normal			   = intersection_point - intersected_sphere.collider.position;
		vecmath::vec3 normalized_intersection_point_normal = vecmath::normalize(intersection_point_normal);

		// Get the direct illumination value; this is light that shines from light sources rather than the global illumination implementation
		vecmath::vec3 direct_colour = direct_illumination(currentray, scene, intersected_sphere, intersection_point, intersection_point_normal, depth, monte_carlo, num_path_traces, random_state);
		image[pixel]=direct_colour;
		//printf("color%f",direct_colour);
		return;
	}

	if(hit_a_triangle)
	{
		image[pixel]=vecmath::vec3(0.0f, 0.0f, 0.0f);

		return;
	}

	return;
}

__global__ void ray_generation(vecmath::vec3 *outputray,Options option, curandState *random_state, Camera camera)
{
	
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float inv_width	 = 1.0f / (float) w;
	float inv_height = 1.0f / (float) h;

	float aspect_ratio = (float) w / (float) h;
	float angle		   = tan(M_PI * 0.5f * option.fov / 180.0f);


	if(x >= w || y >= h)
	{
		return;
	}

	// This was the cpu code:
	float u = (2 * ((x + 0.5) * inv_width) - 1) * angle * aspect_ratio;
	float v = (1 - 2 * ((y + 0.5) * inv_height)) * angle;

	vecmath::vec3 ray_dir;
	ray_dir = camera.direction + u * camera.right + v * camera.up;
	//ray_dir.y -= 1.0f; // this makes the ray_direction correct
	ray_dir = vecmath::normalize(ray_dir);

	Ray ray;
	ray.position  = camera.position;
	ray.direction = ray_dir;
	ray.direction = vecmath::normalize(ray.direction);


	// the pixel index is y * width since width is the max x value, and then this gets us to the y coordinate
	// and then shift over by x amount to hit pixel (x, y)
	int pixel = y * w + x;
	//printf("outputray[pixel]");

	// initialize the random state for this pixel
//	curand_init(5351 * pixel, 0, 0, &random_state[pixel]);
	 outputray[pixel]=ray_dir;
	 //printf("outputray[pixel]%f",outputray[pixel].x);
	//ray[pixel] = shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces, random_state);
	//__syncthreads(); // can't tell if this is necessary; it might be with shared mem access
}


void generate_rays(Scene scene, Options option, char *output)
{
	scene.width	 = 1920;
	scene.height = 1080;

	// The output image that will be written to
	size_t image_size = scene.width * scene.height * sizeof(vecmath::vec3);
	size_t ray_size = scene.width * scene.height * sizeof(vecmath::vec3);

	// The image host; will be copied back to CPU to be displayed
	vecmath::vec3 *image_host = (vecmath::vec3 *) malloc(image_size * sizeof(vecmath::vec3));
	vecmath::vec3 *ray_host = (vecmath::vec3 *) malloc(image_size * sizeof(vecmath::vec3));
	//vecmath::vec3 *image_host = (vecmath::vec3 *) malloc(image_size * sizeof(vecmath::vec3));

	// The image device, rendered pixels on gpu
	vecmath::vec3 *image;
	cudaMalloc((void **) &image, image_size);
	vecmath::vec3 *raydir;
	cudaMalloc((void **) &raydir, image_size);

	// Random state to get CUDA RNG
	curandState *random_state;
	cudaMalloc((void **) &random_state, image_size * sizeof(curandState));

	// Copy the Scene over to a CudaScene that lives on the host
	CudaScene host_cuda_scene = CudaScene(scene); //try constant scnece


	// The CudaScene that will be passed to the device.
	CudaScene cuda_scene_data = allocate_device_cudascene_struct(host_cuda_scene);

	// Can test different block sizes; this will give us 16 * 16 = 256 = 32 * 8 warps.
	int numofobject=cuda_scene_data.num_spheres;

 	int thread_x = 30;
	int thread_y = 30;

	dim3 blocks;  //this is actually number of grid
	blocks.x = scene.width / thread_x + 1;
	
	blocks.y = scene.height / thread_y + 1;
	blocks.z = 1;

	dim3 grid;         // blocksize 
	grid.x = thread_x;
	grid.y = thread_y;
	grid.z = 1;

	// Copy host memory to device memory
	float time;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord( start,0);
  //  cudaMemcpyToSymbol(Mc, M.elements, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
	cudaMemcpy(image, image_host, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(raydir, ray_host, image_size, cudaMemcpyHostToDevice);

   // constant memory
   cudaMemcpyToSymbol(w,&scene.width,sizeof(int));
   cudaMemcpyToSymbol(h,&scene.height,sizeof(int));
   //cudaMemcpyToSymbol(camera, &scene.camera, sizeof(scene.camera));


	//allocate_constant_cudascene(host_cuda_scene);

	// Launch kernel
	ray_generation<<<blocks, grid>>>(raydir, option, random_state,scene.camera);//return a raydir

    cudaDeviceSynchronize();
    blocks.x*=numofobject;
	shade<<<blocks,grid>>>(image,raydir,cuda_scene_data.camera.position,numofobject,cuda_scene_data, option.max_depth, option.monte_carlo, option.num_path_traces, random_state); //return color;
	// Copy the memory back to the host and then synchronize
	cudaMemcpy(image_host, image, image_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//time function
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("ConvolutionKernel\n");
	printf("GPU time used:%f ms\n",time);
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
