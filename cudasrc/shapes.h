#ifndef SHAPES_H
#define SHAPES_H

#include <iostream>


#include "SphereCollider.h"
#include "material.h"
#include "vec3.h"

struct Sphere
{
	SphereCollider collider;
	Material material;

	void to_string()
	{
		std::cout << "Position: " << vecmath::to_string(collider.position) << std::endl
				  << "radius: " << collider.radius << std::endl;
		material.to_string();
	}
};


struct Triangle
{
	vecmath::vec3 v0;
	vecmath::vec3 v1;
	vecmath::vec3 v2;

	Material material;
};


#endif
