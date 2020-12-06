#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"

struct Camera
{
	vecmath::vec3 position;
	vecmath::vec3 direction;
	vecmath::vec3 up;
	vecmath::vec3 right;
	float half_height_angle;

	Camera()
	{
		position  = vecmath::vec3(0, 0, 0);
		direction = vecmath::vec3(0, 0, 0);
		up		  = vecmath::vec3(0, 0, 0);
		right	  = vecmath::cross(direction * -1.0f, up);
	}

	Camera(vecmath::vec3 pos, vecmath::vec3 dir, vecmath::vec3 u, float h)
	{
		position		  = pos;
		direction		  = dir;
		up				  = u;
		half_height_angle = h;
		right			  = vecmath::cross(direction * -1.0f, up);
	}
};

#endif
