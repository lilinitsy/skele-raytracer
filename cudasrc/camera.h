#ifndef CAMERA_H
#define CAMERA_H


#include "glm/glm.hpp"


struct Camera
{
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;
	glm::vec3 right;
	float half_height_angle;

	Camera()
	{
		position  = glm::vec3(0, 0, 0);
		direction = glm::vec3(0, 0, 0);
		up		  = glm::vec3(0, 0, 0);
		right	  = glm::cross(direction * -1.0f, up);
	}

	Camera(glm::vec3 pos, glm::vec3 dir, glm::vec3 u, float h)
	{
		position		  = pos;
		direction		  = dir;
		up				  = u;
		half_height_angle = h;
		right			  = glm::cross(direction * -1.0f, up);
	}
};

#endif
