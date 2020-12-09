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
	total_colour += bp::specular_shading(scene, intersected_sphere, intersection_point, intersection_point_normal);

	// Fresnel effect for refraction
	float fr = bp::fresnel(ray.direction, intersection_point_normal, intersected_sphere);

	vecmath::vec3 refraction_colour = vecmath::vec3(0.0f, 0.0f, 0.0f);
	vecmath::vec3 reflection_colour = vecmath::vec3(0.0f, 0.0f, 0.0f);

	// If the intersected_sphere's material has a specular value and there are more reflections to calculate
	if(intersected_sphere.material.specular != vecmath::vec3(0.0f, 0.0f, 0.0f) && depth > 0)
	{
		// This does it for every point light
		for(unsigned int i = 0; i < scene.num_pointlights; i++)
		{
			vecmath::vec3 light_direction = vecmath::normalize(scene.point_lights[i].position - intersection_point);

			// If the fresnel value is under 1, then there is refraction occurring as the light passes through
			if(fr < 1)
			{
				vecmath::vec3 refraction_ray_direction = bp::refraction(ray.direction, intersection_point_normal, intersected_sphere);

				Ray refracted_ray;
				refracted_ray.position	= intersection_point;
				refracted_ray.direction = refraction_ray_direction;

				// Recursively call the shade function using the refracted ray, with one fewer depth
				refraction_colour = fr * shade(refracted_ray, scene, depth - 1, monte_carlo, num_path_traces, random_state);
			}

			// Add in the reflection colour recursively
			vecmath::vec3 reflected_ray_direction = bp::reflect_direction(light_direction, intersection_point_normal);
			Ray reflected_ray;
			reflected_ray.position	= intersection_point;
			reflected_ray.direction = reflected_ray_direction;
			reflection_colour += (1 - fr) * intersected_sphere.material.specular * shade(reflected_ray, scene, depth - 1, monte_carlo, num_path_traces, random_state);
		}

		// This does the same code for every directional light; obviously this could've been vastly modularized
		for(unsigned int i = 0; i < scene.num_directionallights; i++)
		{
			vecmath::vec3 light_direction = vecmath::normalize(scene.directional_lights[i].direction);

			if(fr < 1)
			{
				vecmath::vec3 refraction_ray_direction = bp::refraction(ray.direction, intersection_point_normal, intersected_sphere);
				Ray refracted_ray;
				refracted_ray.position	= intersection_point;
				refracted_ray.direction = refraction_ray_direction;

				refraction_colour = fr * shade(refracted_ray, scene, depth - 1, monte_carlo, num_path_traces, random_state);
			}

			vecmath::vec3 reflected_ray_direction = bp::reflect_direction(light_direction, intersection_point_normal);
			Ray reflected_ray;
			reflected_ray.position	= intersection_point;
			reflected_ray.direction = reflected_ray_direction;
			reflection_colour += (1 - fr) * intersected_sphere.material.specular * shade(reflected_ray, scene, depth - 1, monte_carlo, num_path_traces, random_state);
		}
	}

	return total_colour + refraction_colour + reflection_colour;
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
	// Base case: No more reflections to calculate (for the depth provided)
	if(depth <= 0)
	{
		return vecmath::vec3(0.0f, 0.0f, 0.0f);
	}

	// Check if a sphere intersection occurs and if it does,
	// get the distance and parse that and the intersected object
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


	// Check if a triangle intersection occurs, and if it does,
	// get the distance and parse that and the intersected object
	Triangle intersected_triangle;
	bool hit_a_triangle = false;
	/*
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
	}*/

	// If no sphere was hit, then the background was hit, so return that colour
/*	if(!hit_a_sphere && !hit_a_triangle)
	{
		return scene.background;
	}*/

	if(hit_a_sphere)
	{
		// This is just writing out the standard raytracing equations and solving for the smallest roots
		vecmath::vec3 e_c = ray.position - intersected_sphere.collider.position;
		float a			  = vecmath::dot(ray.direction, ray.direction);
		float b			  = 2 * vecmath::dot(ray.direction, e_c);
		float c			  = vecmath::dot(e_c, e_c) - intersected_sphere.collider.radius * intersected_sphere.collider.radius;
		float t			  = smallest_root(a, b, c);

		// Calculating the intersection point and the normal; the normal doesn't really need to be calculated seperately
		vecmath::vec3 intersection_point		= ray.position + ray.direction * t;
		vecmath::vec3 intersection_point_normal = vecmath::normalize(intersection_point - intersected_sphere.collider.position);

		// Get the direct illumination value; this is light that shines from light sources rather than the global illumination implementation
		vecmath::vec3 direct_colour = direct_illumination(ray, scene, intersected_sphere, intersection_point, intersection_point_normal, depth, monte_carlo, num_path_traces, random_state);

		/*if(monte_carlo)
		{
			vecmath::vec3 indirect_colour = montecarlo_global_illumination(ray, scene, intersected_sphere, intersection_point, intersection_point_normal, depth, num_path_traces);
			vecmath::vec3 total_colour	  = (direct_colour / (float) M_PI + 2.0f * indirect_colour) * intersected_sphere.material.diffuse;

			return total_colour;
		}*/

		return direct_colour;
	}

	if(hit_a_triangle)
	{
	}

	return vecmath::vec3(1.0f, 0.0f, 0.0f);
}

#endif
