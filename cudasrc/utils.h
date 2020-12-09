#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>

#include <curand_kernel.h>


#include "SphereCollider.h"
#include "lights.h"
#include "ray.h"
#include "scene.h"
#include "shapes.h"
#include "vec3.h"

struct Options
{
	bool monte_carlo	  = false;
	bool visual			  = true;
	float fov			  = 60;
	short num_path_traces = 1;
	short grid_size		  = 0;
	int max_depth		  = 3;

	void to_string()
	{
		printf("\n\nMonte carlo: %d\nvisual display: %d\nfov: %f\nnum paths traced: %d\nsupersample grid size: %d\nmax depth: %d\n", monte_carlo, visual, fov, num_path_traces, grid_size, max_depth);
	}
};


struct TupleOfVec3
{
	vecmath::vec3 first;
	vecmath::vec3 second;
};


__device__ float euclidean_distance(int x, int y);
__device__ float smallest_root(float a, float b, float c);
__device__ float collision_distance(Ray ray, SphereCollider collider);
__device__ float intensity_of(PointLight light);
__device__ float clamp(float a, float b, float input);
__device__ TupleOfVec3 transform_coordinate_space(vecmath::vec3 normal);
__device__ bool intersection_occurs(Ray ray, SphereCollider collider);
__device__ bool shadow(CudaScene scene, vecmath::vec3 intersection_point, PointLight point_light);
__device__ bool shadow(CudaScene scene, vecmath::vec3 intersection_point, DirectionalLight directional_light);


__device__ bool shadow(CudaScene scene, vecmath::vec3 intersection_point, PointLight point_light)
{
	vecmath::vec3 direction = vecmath::normalize(point_light.position - intersection_point);
	Ray ray;
	ray.position  = intersection_point + 0.000001f;
	ray.direction = direction;

	for(unsigned int i = 0; i < scene.num_spheres; i++)
	{
		if(intersection_occurs(ray, scene.spheres[i].collider))
		{
			return true;
		}
	}

	return false;
}

__device__ bool shadow(CudaScene scene, vecmath::vec3 intersection_point, DirectionalLight directional_light)
{
	vecmath::vec3 direction = vecmath::normalize(directional_light.direction);
	Ray ray;
	ray.position  = intersection_point + 0.000001f;
	ray.direction = direction;

	for(unsigned int i = 0; i < scene.num_spheres; i++)
	{
		if(intersection_occurs(ray, scene.spheres[i].collider))
		{
			return true;
		}
	}

	return false;
}

__device__ float euclidean_distance(int x, int y)
{
	if(x == 0 && y == 0)
	{
		return 1.0f;
	}
	return sqrtf(x * x + y * y);
}

__device__ float smallest_root(float a, float b, float c)
{
	float discriminant = b * b - 4 * a * c;

	if(discriminant < 0)
	{
		return INFINITY;
	}

	float t1 = ((-b) + sqrt(discriminant)) / (2 * a);
	float t2 = ((-b) - sqrt(discriminant)) / (2 * a);

	if(t1 < t2 && t1 >= 0)
	{
		return t1;
	}

	else if(t2 >= 0)
	{
		return t2;
	}

	return INFINITY;
}


__device__ float collision_distance(Ray ray, SphereCollider collider)
{
	vecmath::vec3 e_c = ray.position - collider.position;
	float a			  = vecmath::dot(ray.direction, ray.direction);
	float b			  = 2 * vecmath::dot(ray.direction, e_c);
	float c			  = vecmath::dot(e_c, e_c) - collider.radius * collider.radius;

	return smallest_root(a, b, c);
}



__device__ float intensity_of(PointLight light)
{
	float intensity = 0.30f * light.colour.x + 0.59f * light.colour.y + 0.11f * light.colour.z;
	return intensity;
}


__device__ float clamp(float a, float b, float input)
{
	if(input < a)
	{
		return a;
	}

	else if(input > b)
	{
		return b;
	}

	return input;
}


__device__ TupleOfVec3 transform_coordinate_space(vecmath::vec3 normal)
{
	vecmath::vec3 perp_to_normal;
	vecmath::vec3 perp_to_both;
	if(std::fabs(normal.x) > std::fabs(normal.y))
	{
		perp_to_normal = vecmath::vec3(normal.z, 0, -normal.x) / sqrtf(normal.x * normal.x + normal.z * normal.z);
	}

	else
	{
		perp_to_normal = vecmath::vec3(0, -normal.z, normal.y) / sqrtf(normal.y * normal.y + normal.z * normal.z);
	}

	perp_to_both = vecmath::cross(normal, perp_to_normal);


	TupleOfVec3 transformed_coordinate_spaces;
	transformed_coordinate_spaces.first	 = perp_to_normal;
	transformed_coordinate_spaces.second = perp_to_both;

	return transformed_coordinate_spaces;
}



__device__ bool intersection_occurs(Ray ray, SphereCollider collider)
{
	float distance = collision_distance(ray, collider);

	if(distance <= 1.0f || distance == INFINITY)
	{
		return false;
	}

	return true;
}

__device__ bool triangle_intersection_occurs(Ray ray, Triangle triangle, float &t, float &u, float &v)
{
	vecmath::vec3 v0v1 = triangle.v1 - triangle.v0;
	vecmath::vec3 v0v2 = triangle.v2 - triangle.v0;
	vecmath::vec3 v1v2 = triangle.v2 - triangle.v1;
	vecmath::vec3 p	   = vecmath::cross(ray.direction, v0v2);
	float d			   = vecmath::dot(v0v1, p);

	// ray parallel to triangle
	if(fabs(d) < 0.00001f)
	{
		return false;
	}

	float inverse = 1.0f / d;

	vecmath::vec3 t_vector = ray.position - triangle.v0;
	u					   = inverse * vecmath::dot(-t_vector, p);
	if(u < 0 || u > 1)
	{
		return false;
	}

	vecmath::vec3 q = vecmath::cross(t_vector, v0v1);
	v				= vecmath::dot(ray.direction, q) * inverse;
	if(v < 0 || u + v > 1)
	{
		return false;
	}

	t = vecmath::dot(v0v2, q) * inverse;
	return true;
}

__device__ float maxf(float f1, float f2)
{
	return f1 > f2 ? f1 : f2;
}

__device__ void swapf(float &f1, float &f2)
{
	float tmp = f1;
	f1		  = f2;
	f2		  = tmp;
}



#endif
