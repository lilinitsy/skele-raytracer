#ifndef LIGHTS_H
#define LIGHTS_H


#include "glm/glm.hpp"


struct AmbientLight
{
	glm::vec3 colour;

	AmbientLight()
	{
		colour = glm::vec3(0, 0, 0);
	}

	AmbientLight(glm::vec3 col)
	{
		colour = col;
	}
};

struct DirectionalLight
{
	glm::vec3 position;
	glm::vec3 colour;

	DirectionalLight()
	{
		colour = glm::vec3(0, 0, 0);
		position = glm::vec3(0, 0, 0);
	}
};

struct PointLight
{
	glm::vec3 position;
	glm::vec3 colour;

	PointLight()
	{
		position = glm::vec3(0, 0, 0);
		colour = glm::vec3(0, 0, 0);
	}

	PointLight(glm::vec3 pos, glm::vec3 col)
	{
		position = pos;
		colour = col;
	}
};



#endif
