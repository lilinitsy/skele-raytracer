#ifndef MATERIAL_H
#define MATERIAL_H

#include <iostream>

#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"

struct Material
{
	glm::vec3 ambient;
	glm::vec3 diffuse;
	glm::vec3 specular;
	glm::vec3 transmissive;

	float power;
	float ior;

	Material()
	{
		ambient = glm::vec3(0, 0, 0);
		diffuse = glm::vec3(0, 0, 0);
		specular = glm::vec3(0, 0, 0);
		transmissive = glm::vec3(0, 0, 0);
		power = 1;
		ior = 1;
	}

	Material(glm::vec3 amb, glm::vec3 diff, glm::vec3 spec, glm::vec3 trans, float p, float io)
	{
		ambient = amb;
		diffuse = diff;
		specular = spec;
		transmissive = trans;
		power = p;
		ior = io;
	}

	void to_string()
	{
		std::cout << "Ambient: " << glm::to_string(ambient) << std::endl
					<< "Diffuse: " << glm::to_string(diffuse) << std::endl
					<< "specular: " << glm::to_string(specular) << std::endl
					<< "transmissive: " << glm::to_string(transmissive) << std:: endl;
	}
};

#endif
