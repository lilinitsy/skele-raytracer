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
__device__ vecmath::vec3 shade(Ray ray, CudaScene scene, int depth, bool monte_carlo, short num_path_traces, curandState *random_state);


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


__device__ vecmath::vec3 montecarlo_global_illumination(Ray ray, CudaScene scene, Sphere intersected_sphere, vecmath::vec3 intersection_point, vecmath::vec3 intersection_point_normal, int depth, int num_rays, curandState *random_state)
{
	vecmath::vec3 total_colour = vecmath::vec3(0.0f, 0.0f, 0.0f);

	// Getting portions of tuples requires the --expt-relaxed-constexpr flag
	TupleOfVec3 tmp				 = transform_coordinate_space(intersection_point_normal);
	vecmath::vec3 perp_to_normal = tmp.first;
	vecmath::vec3 perp_to_both	 = tmp.second;

	float probability_dist = 1 / (M_PI);

	for(int i = 0; i < num_rays; i++)
	{
		float r1 = curand_uniform(random_state);
		float r2 = curand_uniform(random_state);

		vecmath::vec3 sample = uniform_sample_hemi(r1, r2);
		vecmath::vec3 sample_world_space(sample.x * perp_to_both.x + sample.y * intersection_point_normal.x + sample.z * perp_to_normal.x,
										 sample.x * perp_to_both.y + sample.y * intersection_point_normal.y + sample.z * perp_to_both.y,
										 sample.x * perp_to_both.z + sample.y * intersection_point_normal.z + sample.z * perp_to_both.z);

		Ray indirect_ray;
		indirect_ray.position  = intersection_point + 0.00001f;
		indirect_ray.direction = sample_world_space;
		total_colour += (r1 * shade(indirect_ray, scene, depth - 1, true, num_rays, random_state)) / probability_dist;
	}

	total_colour /= num_rays;

	return total_colour;
}


__device__ vecmath::vec3 shade(Ray ray, CudaScene scene, int depth, bool monte_carlo, short num_path_traces, curandState *random_state)
{

	float min_distance = INFINITY;
	Sphere intersected_sphere;
	bool hit_a_sphere = false;
	for(unsigned int i = 0; i < scene.num_spheres; i++)
	{
		if(intersection_occurs(ray, scene.spheres[i].collider))
		{
			hit_a_sphere   = true;
			float distance = collision_distance(ray, scene.spheres[i].collider);

			if(distance < min_distance)
			{
				min_distance	   = distance;
				intersected_sphere = scene.spheres[i];
			}
		}
	}

	Triangle intersected_triangle;
	bool hit_a_triangle = false;

	for(unsigned int i = 0; i < scene.num_triangles; i++)
	{
		float t;
		float u;
		float v;
		if(triangle_intersection_occurs(ray, scene.triangles[i], t, u, v))
		{
			if(t < min_distance)
			{
				min_distance		 = t;
				hit_a_sphere		 = false;
				hit_a_triangle		 = true;
				intersected_triangle = scene.triangles[i];
			}
		}
	}

	// If no sphere was hit, then the background was hit, so return that colour
	if(!hit_a_sphere && !hit_a_triangle)
	{
		return scene.background;
	}

	if(hit_a_sphere)
	{
		// ray.direction is the SAME
		// ray.position is the SAME
		vecmath::vec3 e_c = ray.position - intersected_sphere.collider.position; // e_c matches to e_c in cpu
		float a			  = vecmath::dot(ray.direction, ray.direction);
		float b			  = 2 * vecmath::dot(ray.direction, e_c);
		float c			  = vecmath::dot(e_c, e_c) - intersected_sphere.collider.radius * intersected_sphere.collider.radius;
		float t			  = smallest_root(a, b, c); // t's match

		// Calculating the intersection point and the normal; the normal doesn't really need to be calculated seperately
		vecmath::vec3 intersection_point				   = ray.position + ray.direction * t; // ray_direction * t is the same, verified by a diff, and so is ray.position
		vecmath::vec3 intersection_point_normal			   = intersection_point - intersected_sphere.collider.position;
		vecmath::vec3 normalized_intersection_point_normal = vecmath::normalize(intersection_point_normal);

		// Get the direct illumination value; this is light that shines from light sources rather than the global illumination implementation
		vecmath::vec3 direct_colour = direct_illumination(ray, scene, intersected_sphere, intersection_point, intersection_point_normal, depth, monte_carlo, num_path_traces, random_state);


		//return direct_colour;
		return direct_colour;
	}

	if(hit_a_triangle)
	{
		return vecmath::vec3(0.0f, 0.0f, 0.0f);
	}

	return vecmath::vec3(0.1f, 0.8f, 0.8f);
}

#endif
