#ifndef BLINN_PHONG_H
#define BLINN_PHONG_H

#include "scene.h"
#include "utils.h"
#include "vec3.h"



namespace bp
{
	// Add in a uniform colour across the scene for ambient lighting
	__device__ vecmath::vec3 ambient_shading(CudaScene scene, Sphere sphere)
	{
		vecmath::vec3 colour = scene.ambient_light.colour * sphere.material.ambient;
		return colour;
	}


	// Add in the diffuse shading based on material properties of the sphere
	__device__ vecmath::vec3 diffuse_shading(CudaScene scene, Sphere sphere, vecmath::vec3 intersection_point, vecmath::vec3 norm)
	{
		vecmath::vec3 colour = vecmath::vec3(0.0f, 0.0f, 0.0f);
		// handle fog in difffuse and specular, maybe?
		for(unsigned int i = 0; i < scene.num_pointlights; i++)
		{
			if(!scene.use_shadows || !shadow(scene, intersection_point, scene.point_lights[i]))
			{
				vecmath::vec3 light_direction = vecmath::normalize(scene.point_lights[i].position - intersection_point);


				// without fog, make the colour the same as the if() case in the spherical_fog equation
				float distance	= vecmath::length(scene.point_lights[i].position - intersection_point);
				float intensity = 1.0f / powf(vecmath::length(distance), 2.0f);

				colour += sphere.material.diffuse * scene.point_lights[i].colour * intensity * maxf(0.0f, vecmath::dot(norm, light_direction));
			}
		}

		for(unsigned int i = 0; i < scene.num_directionallights; i++)
		{
			if(!scene.use_shadows || !shadow(scene, intersection_point, scene.directional_lights[i]))
			{
				vecmath::vec3 light_direction = vecmath::normalize(scene.directional_lights[i].direction);
				colour += sphere.material.diffuse * scene.directional_lights[i].colour * maxf(0.0f, vecmath::dot(norm, light_direction));
			}
		}

		return colour;
	}


	__device__ vecmath::vec3 specular_shading(CudaScene scene, Sphere sphere, vecmath::vec3 intersection_point, vecmath::vec3 norm)
	{
		vecmath::vec3 colour		 = vecmath::vec3(0.0f, 0.0f, 0.0f);
		vecmath::vec3 view_direction = vecmath::normalize(scene.camera.position - intersection_point);

		for(unsigned int i = 0; i < scene.num_pointlights; i++)
		{
			if(!scene.use_shadows || !shadow(scene, intersection_point, scene.point_lights[i]))
			{
				// This is an interpretation of the specular highlight vectors
				vecmath::vec3 light_direction = vecmath::normalize(scene.point_lights[i].position - intersection_point);
				vecmath::vec3 half_vector	  = (view_direction + light_direction) / vecmath::length(view_direction + light_direction);

				float distance	= vecmath::length(scene.point_lights[i].position - intersection_point);
				float intensity = 1.0f / powf(vecmath::length(distance), 2.0f);

				// Specular lighting equation; iteratively add it for every point light
				colour += sphere.material.specular * scene.point_lights[i].colour * intensity * powf(maxf(0.0f, vecmath::dot(norm, half_vector)), sphere.material.power);
			}
		}

		for(unsigned int i = 0; i < scene.num_directionallights; i++)
		{
			printf("checking size isn't 0\n");
			if(!scene.use_shadows || !shadow(scene, intersection_point, scene.directional_lights[i]))
			{
				vecmath::vec3 light_direction = vecmath::normalize(scene.directional_lights[i].direction);
				vecmath::vec3 half_vector	  = (view_direction + light_direction) / vecmath::length(view_direction + light_direction);
				colour += sphere.material.specular * scene.directional_lights[i].colour * powf(maxf(0.0f, vecmath::dot(norm, half_vector)), sphere.material.power);
			}
		}

		return colour;
	}


	__device__ vecmath::vec3 reflect_direction(vecmath::vec3 light_direction, vecmath::vec3 normal)
	{
		return vecmath::normalize(light_direction - 2.0f * vecmath::dot(light_direction, normal) * normal);
	}


	__device__ vecmath::vec3 refraction(vecmath::vec3 dir, vecmath::vec3 normal, Sphere sphere)
	{
		float k = 1.0f - powf(sphere.material.ior, 2.0f) * (1.0f - powf(vecmath::dot(dir, normal), 2.0f));

		if(k < 0.0f)
		{
			return vecmath::vec3(0.0f, 0.0f, 0.0f);
		}

		return sphere.material.ior * dir - (sphere.material.ior * vecmath::dot(dir, normal) + sqrtf(k)) * normal;
	}


	__device__ float fresnel(vecmath::vec3 ray_direction, vecmath::vec3 normal, Sphere sphere)
	{
		float cos_internal = clamp(-1.0f, 1.0f, vecmath::dot(ray_direction, normal));
		float et		   = 1.0f;
		float ior		   = sphere.material.ior;


		if(cos_internal > 0)
		{
			swapf(et, ior);
		}

		float sint = et / ior * sqrt(maxf(0.0f, 1.0f - powf(cos_internal, 2.0f)));

		// total reflection)
		if(sint >= 1.0f)
		{
			return 1.0f;
		}

		float cos_theta = sqrt(maxf(0.0f, 1 - powf(sint, 2.0f)));
		cos_internal	= abs(cos_internal);

		// Schlick approximation
		float Rs = ((ior * cos_internal) - (et * cos_theta)) / ((ior * cos_internal) + (et * cos_theta));
		float Rp = ((et * cos_internal) - (ior * cos_theta)) / ((ior * cos_internal) + (et * cos_theta));

		return (powf(Rs, 2.0f) + powf(Rp, 2.0f)) / 2.0f;
	}
}; // namespace bp


#endif
