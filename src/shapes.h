#ifndef SHAPES_H
#define SHAPES_H

#include <iostream>

#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"
#include "material.h"

struct Sphere
{
	glm::vec3 position;
	float radius;
	Material material;

	Sphere()
	{
		position = glm::vec3(0, 0, 0);
		radius = 1;
		material = Material();
	}

	Sphere(glm::vec3 pos, float rad, Material mat)
	{
		position = pos;
		radius = rad;
		material = mat;
	}

	glm::vec3 get_Center()
	{
		return position;
	}

	void to_string()
	{
		std::cout << "Position: " << glm::to_string(position) << std::endl
					<< "radius: " << radius << std::endl;
		material.to_string();
	}
};


struct Triangle
{
	glm::vec4 v0_to_v1;
	glm::vec4 v0_to_v2;

	glm::vec3 vertex_one;
	glm::vec3 vertex_two;
	glm::vec3 vertex_three;

	Material materail;


	Triangle()
	{

	}
};


#endif
