#ifndef SPHERECOLLIDER_H
#define SPHERECOLLIDER_H

#include "vec3.h"

struct SphereCollider
{
	vecmath::vec3 position = vecmath::vec3(0.0f, 0.0f, 0.0f);
	float radius		   = 1.0f;
};



#endif