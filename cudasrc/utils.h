#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
#include <tuple>

#include "SphereCollider.h"
#include "lights.h"
#include "ray.h"
#include "scene.h"
#include "shapes.h"
#include "vec3.h"

float euclidean_distance(int x, int y);
float smallest_root(float a, float b, float c);
float collision_distance(Ray ray, SphereCollider collider);
float intensity_of(PointLight light);
float clamp(float a, float b, float input);
std::tuple<vecmath::vec3, vecmath::vec3> transform_coordinate_space(vecmath::vec3 normal);
bool intersection_occurs(Ray ray, SphereCollider collider);
bool shadow(Scene scene, vecmath::vec3 intersection_point, PointLight point_light);
bool shadow(Scene scene, vecmath::vec3 intersection_point, DirectionalLight directional_light);


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


bool shadow(Scene scene, vecmath::vec3 intersection_point, PointLight point_light)
{
	vecmath::vec3 direction = vecmath::normalize(point_light.position - intersection_point);
	Ray ray;
	ray.position  = intersection_point + 0.000001f;
	ray.direction = direction;

	for(unsigned int i = 0; i < scene.spheres.size(); i++)
	{
		if(intersection_occurs(ray, scene.spheres[i].collider))
		{
			return true;
		}
	}

	return false;
}

bool shadow(Scene scene, vecmath::vec3 intersection_point, DirectionalLight directional_light)
{
	vecmath::vec3 direction = vecmath::normalize(directional_light.direction);
	Ray ray;
	ray.position  = intersection_point + 0.000001f;
	ray.direction = direction;

	for(unsigned int i = 0; i < scene.spheres.size(); i++)
	{
		if(intersection_occurs(ray, scene.spheres[i].collider))
		{
			return true;
		}
	}

	return false;
}

float euclidean_distance(int x, int y)
{
	if(x == 0 && y == 0)
	{
		return 1.0f;
	}
	return sqrtf(x * x + y * y);
}

float smallest_root(float a, float b, float c)
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


float collision_distance(Ray ray, SphereCollider collider)
{
	vecmath::vec3 e_c = ray.position - collider.position;
	float a			  = vecmath::dot(ray.direction, ray.direction);
	float b			  = 2 * vecmath::dot(ray.direction, e_c);
	float c			  = vecmath::dot(e_c, e_c) - collider.radius * collider.radius;

	return smallest_root(a, b, c);
}



float intensity_of(PointLight light)
{
	float intensity = 0.30f * light.colour.x + 0.59f * light.colour.y + 0.11f * light.colour.z;
	return intensity;
}


float clamp(float a, float b, float input)
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


std::tuple<vecmath::vec3, vecmath::vec3> transform_coordinate_space(vecmath::vec3 normal)
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


	std::tuple<vecmath::vec3, vecmath::vec3> transformed_coordinate_spaces = std::make_tuple(perp_to_normal, perp_to_both);

	return transformed_coordinate_spaces;
}



bool intersection_occurs(Ray ray, SphereCollider collider)
{
	float distance = collision_distance(ray, collider);

	if(distance <= 1.0f || distance == INFINITY)
	{
		return false;
	}

	return true;
}

bool triangle_intersection_occurs(Ray ray, Triangle triangle, float &t, float &u, float &v)
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


vecmath::vec3 scattering_phase_function(vecmath::vec3 direction, float scattering)
{
	// keep it scattering in same general direction
	float x = -1.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (1 + 1));
	float y = -1.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (1 + 1));
	float z = -1.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (1 + 1));

	return vecmath::vec3(direction.x + x * scattering, direction.y + y * scattering, direction.z + z * scattering);
}


#endif
