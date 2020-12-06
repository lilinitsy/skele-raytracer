#ifndef RAY_H
#define RAY_H

#include "vec3.h"

struct Ray
{
	vecmath::vec3 position	= vecmath::vec3(0.0f, 0.0f, 0.0f);
	vecmath::vec3 direction = vecmath::vec3(0.0f, 0.0f, 0.0f);
};

#endif
