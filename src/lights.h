#ifndef LIGHTS_H
#define LIGHTS_H


#include "glm/glm.hpp"


struct AmbientLight
{
	glm::vec3 colour = glm::vec3(0.0f, 0.0f, 0.0f);
};

struct DirectionalLight
{
	glm::vec3 direction = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 colour	= glm::vec3(0.0f, 0.0f, 0.0f);
};

struct PointLight
{
	glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 colour   = glm::vec3(0.0f, 0.0f, 0.0f);
};



#endif
