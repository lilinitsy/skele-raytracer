#ifndef RAY_H
#define RAY_H


#include "glm/glm.hpp"


struct Ray
{
	glm::vec3 position	= glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 direction = glm::vec3(0.0f, 0.0f, 0.0f);
};

#endif
