#ifndef MATERIAL_H
#define MATERIAL_H

#include <iostream>

#include "vec3.h"

struct Material
{
	vecmath::vec3 ambient	   = vecmath::vec3(0, 0, 0);
	vecmath::vec3 diffuse	   = vecmath::vec3(0, 0, 0);
	vecmath::vec3 specular	   = vecmath::vec3(0, 0, 0);
	vecmath::vec3 transmissive = vecmath::vec3(0, 0, 0);

	float power = 1.0f;
	float ior	= 1.0f;

	void to_string()
	{
		std::cout << "Ambient: " << vecmath::to_string(ambient) << std::endl
				  << "Diffuse: " << vecmath::to_string(diffuse) << std::endl
				  << "specular: " << vecmath::to_string(specular) << std::endl
				  << "transmissive: " << vecmath::to_string(transmissive) << std::endl;
	}
};

#endif
