#ifndef RAYTRACE_H
#define RAYTRACE_H


#include <algorithm>


#include "blinn_phong.h"
#include "material.h"
#include "ray.h"
#include "scene.h"
#include "shapes.h"
#include "utils.h"
#include "vec3.h"

__device__ vecmath::vec3 uniform_sample_hemi(float r1, float r2);
__device__ vecmath::vec3 direct_illumination(Ray ray, Scene scene, Sphere intersected_sphere, int depth, curandState *random_state);
//___ vecmath::vec3 shade(Ray ray, CudaScene scene, int depth, bool monte_carlo, short num_path_traces, curandState *random_state);


__device__ vecmath::vec3 uniform_sample_hemi(float r1, float r2)
{
	float s_theta = sqrtf(1 - powf(r1, 2.0f));
	float phi	  = 2.0f * M_PI * r2;
	float x		  = s_theta * cosf(phi);
	float z		  = s_theta * sinf(phi);

	return vecmath::vec3(x, r1, z);
}





__device__ vecmath::vec3 direct_illumination(Ray ray, CudaScene scene, Sphere intersected_sphere, vecmath::vec3 intersection_point, vecmath::vec3 intersection_point_normal, int depth, bool monte_carlo, short num_path_traces, curandState *random_state)
{
	// Iteratively add the ambient, diffuse, and specular components to the total colour
	vecmath::vec3 total_colour = vecmath::vec3(0.0f, 0.0f, 0.0f);
	total_colour += bp::ambient_shading(scene, intersected_sphere);
	total_colour += bp::diffuse_shading(scene, intersected_sphere, intersection_point, intersection_point_normal);
	//total_colour += bp::specular_shading(scene, intersected_sphere, intersection_point, intersection_point_normal);

	return total_colour;
}


// __device__ vecmath::vec3 montecarlo_global_illumination(Ray ray, CudaScene scene, Sphere intersected_sphere, vecmath::vec3 intersection_point, vecmath::vec3 intersection_point_normal, int depth, int num_rays, curandState *random_state)
// {
// 	vecmath::vec3 total_colour = vecmath::vec3(0.0f, 0.0f, 0.0f);

// 	// Getting portions of tuples requires the --expt-relaxed-constexpr flag
// 	TupleOfVec3 tmp				 = transform_coordinate_space(intersection_point_normal);
// 	vecmath::vec3 perp_to_normal = tmp.first;
// 	vecmath::vec3 perp_to_both	 = tmp.second;

// 	float probability_dist = 1 / (M_PI);

// 	for(int i = 0; i < num_rays; i++)
// 	{
// 		float r1 = curand_uniform(random_state);
// 		float r2 = curand_uniform(random_state);

// 		vecmath::vec3 sample = uniform_sample_hemi(r1, r2);
// 		vecmath::vec3 sample_world_space(sample.x * perp_to_both.x + sample.y * intersection_point_normal.x + sample.z * perp_to_normal.x,
// 										 sample.x * perp_to_both.y + sample.y * intersection_point_normal.y + sample.z * perp_to_both.y,
// 										 sample.x * perp_to_both.z + sample.y * intersection_point_normal.z + sample.z * perp_to_both.z);

// 		Ray indirect_ray;
// 		indirect_ray.position  = intersection_point + 0.00001f;
// 		indirect_ray.direction = sample_world_space;
// 		total_colour += (r1 * shade(indirect_ray, scene, depth - 1, true, num_rays, random_state)) / probability_dist;
// 	}

// 	total_colour /= num_rays;

// 	return total_colour;
// }



#endif
