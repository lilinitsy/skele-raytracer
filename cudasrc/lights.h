#ifndef LIGHTS_H
#define LIGHTS_H

#include "vec3.h"

struct AmbientLight
{
	vecmath::vec3 colour = vecmath::vec3(0.0f, 0.0f, 0.0f);
};

struct DirectionalLight
{
	vecmath::vec3 direction = vecmath::vec3(0.0f, 0.0f, 0.0f);
	vecmath::vec3 colour	= vecmath::vec3(0.0f, 0.0f, 0.0f);
};

struct PointLight
{
	vecmath::vec3 position = vecmath::vec3(0.0f, 0.0f, 0.0f);
	vecmath::vec3 colour   = vecmath::vec3(0.0f, 0.0f, 0.0f);
};



#endif
