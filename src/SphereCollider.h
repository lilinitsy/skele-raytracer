#ifndef SPHERECOLLIDER_H
#define SPHERECOLLIDER_H


#include "glm/glm.hpp"


struct SphereCollider
{
	glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
	float radius = 1.0f;
};



#endif