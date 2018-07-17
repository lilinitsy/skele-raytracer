#ifndef SHAPES_H
#define SHAPES_H

#include <iostream>

#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"

#include "SphereCollider.h"
#include "material.h"

struct Sphere
{
	SphereCollider collider;
	Material material;

	Sphere()
	{
		collider.position = glm::vec3(0, 0, 0);
		collider.radius = 1;
		material = Material();
	}

	Sphere(glm::vec3 pos, float rad, Material mat)
	{
		collider.position = pos;
		collider.radius = rad;
		material = mat;
	}

	void to_string()
	{
		std::cout << "Position: " << glm::to_string(collider.position) << std::endl
					<< "radius: " << collider.radius << std::endl;
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
