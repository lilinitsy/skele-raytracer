#ifndef RAY_H
#define RAY_H


#include "glm/glm.hpp"


struct Ray
{
	glm::vec3 position;
	glm::vec3 direction;

	Ray()
	{
		position = glm::vec3(0, 0, 0);
		direction = glm::vec3(0, 0, 0);
	}

	Ray(glm::vec3 pos, glm::vec3 dir)
	{
		position = pos;
		direction = glm::normalize(dir);
	}
};

#endif
