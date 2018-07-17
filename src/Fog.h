#ifndef FOG_H
#define FOG_H


#include "glm/glm.hpp"

#include "SphereCollider.h"


struct SphericalFog
{
	float scattering;
	float absorption;
	glm::vec3 albedo;
	SphereCollider collider;

	SphericalFog()
	{
		scattering = 0;
		albedo = glm::vec3(0.0f, 0.0f, 0.0f);
		collider.position = glm::vec3(0.0f, 0.0f, 0.0f);
	}

	SphericalFog(float s, float abso, glm::vec3 a, float r, glm::vec3 p)
	{
		scattering = s;
		absorption = abso;
		albedo = a;
		collider.radius = r;
		collider.position = p;
	}
};


#endif