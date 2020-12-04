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

	void to_string()
	{
		std::cout << "Position: " << glm::to_string(collider.position) << std::endl
				  << "radius: " << collider.radius << std::endl;
		material.to_string();
	}
};


struct Triangle
{
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;

	Material material;
};


#endif
