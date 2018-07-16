#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
#include <tuple>

#include "ray.h"
#include "shapes.h"
#include "lights.h"
#include "scene.h"


float euclidean_distance(int x, int y);
float smallest_root(float a, float b, float c);
float collision_distance(Ray ray, Sphere sphere);
float intensity_of(PointLight light);
float clamp(float a, float b, float input);
std::tuple<glm::vec3, glm::vec3> transform_coordinate_space(glm::vec3 normal);
bool intersection_occurs(Ray ray, Sphere sphere);
bool shadow(Scene scene, glm::vec3 intersection_point, PointLight point_light);
bool shadow(Scene scene, glm::vec3 intersection_point, DirectionalLight directional_light);


struct Options
{
    bool monte_carlo = false;
    bool visual = true;
    float fov = 60;
    short num_path_traces = 1;
    short grid_size = 0;
    int max_depth = 3;

    void to_string()
    {
        printf("\n\nMonte carlo: %d\nvisual display: %d\nfov: %f\nnum paths traced: %d\nsupersample grid size: %d\nmax depth: %d\n", monte_carlo, visual, fov, num_path_traces, grid_size, max_depth);
    }
};


bool shadow(Scene scene, glm::vec3 intersection_point, PointLight point_light)
{
	glm::vec3 direction = glm::normalize(point_light.position - intersection_point);
	Ray ray(intersection_point + 0.000001f, direction);

	for(unsigned int i = 0; i < scene.spheres.size(); i++)
	{
		if(intersection_occurs(ray, scene.spheres[i]))
		{
			return true;
		}
	}

	return false;
}

bool shadow(Scene scene, glm::vec3 intersection_point, DirectionalLight directional_light)
{
	glm::vec3 direction = glm::normalize(directional_light.direction);
	Ray ray(intersection_point + 0.000001f, direction);

	for(unsigned int i = 0; i < scene.spheres.size(); i++)
	{
		if(intersection_occurs(ray, scene.spheres[i]))
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


float collision_distance(Ray ray, Sphere sphere)
{
	glm::vec3 e_c = ray.position - sphere.position;
	float a = glm::dot(ray.direction, ray.direction);
	float b = 2 * glm::dot(ray.direction, e_c);
	float c = glm::dot(e_c, e_c) - sphere.radius * sphere.radius;

	return smallest_root(a, b, c);
}



float intensity_of(PointLight light)
{
	float intensity = 0.30f * light.colour.r + 0.59f * light.colour.g + 0.11f * light.colour.b;
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


std::tuple<glm::vec3, glm::vec3> transform_coordinate_space(glm::vec3 normal)
{
	glm::vec3 perp_to_normal;
	glm::vec3 perp_to_both;
	if(std::fabs(normal.x) > std::fabs(normal.y))
	{
		perp_to_normal = glm::vec3(normal.z, 0, -normal.x) / sqrtf(normal.x * normal.x + normal.z * normal.z);
	}

	else
	{
		perp_to_normal = glm::vec3(0, -normal.z, normal.y) / sqrtf(normal.y * normal.y + normal.z * normal.z);
	}

	perp_to_both = glm::cross(normal, perp_to_normal);

	return {perp_to_normal, perp_to_both};
}



bool intersection_occurs(Ray ray, Sphere sphere)
{
	float distance = collision_distance(ray, sphere);

	if(distance <= 1.0f || distance == INFINITY)
	{
		return false;
	}

	return true;
}


#endif
