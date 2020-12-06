#ifndef MATERIAL_H
#define MATERIAL_H

#include <iostream>

#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"

struct Material
{
	glm::vec3 ambient	   = glm::vec3(0, 0, 0);
	glm::vec3 diffuse	   = glm::vec3(0, 0, 0);
	glm::vec3 specular	   = glm::vec3(0, 0, 0);
	glm::vec3 transmissive = glm::vec3(0, 0, 0);

	float power = 1.0f;
	float ior	= 1.0f;

	void to_string()
	{
		std::cout << "Ambient: " << glm::to_string(ambient) << std::endl
				  << "Diffuse: " << glm::to_string(diffuse) << std::endl
				  << "specular: " << glm::to_string(specular) << std::endl
				  << "transmissive: " << glm::to_string(transmissive) << std::endl;
	}
};

#endif
