#ifndef FOG_H
#define FOG_H


#include "SphereCollider.h"
#include "vec3.h"

struct SphericalFog
{
	float scattering;
	float absorption;
	vecmath::vec3 albedo;
	SphereCollider collider;

	SphericalFog()
	{
		scattering		  = 0;
		albedo			  = vecmath::vec3(0.0f, 0.0f, 0.0f);
		collider.position = vecmath::vec3(0.0f, 0.0f, 0.0f);
	}

	SphericalFog(float s, float abso, vecmath::vec3 a, float r, vecmath::vec3 p)
	{
		scattering		  = s;
		absorption		  = abso;
		albedo			  = a;
		collider.radius	  = r;
		collider.position = p;
	}
};


#endif