#ifndef BLINN_PHONG_H
#define BLINN_PHONG_H

#include "glm/glm.hpp"
#include "scene.h"
#include "utils.h"



namespace bp
{
	// Add in a uniform colour across the scene for ambient lighting
	glm::vec3 ambient_shading(Scene scene, Sphere sphere)
	{
		glm::vec3 colour = scene.ambient_light.colour * sphere.material.ambient;
		return colour;
	}

	glm::vec3 spherical_fog_shading(PointLight light, SphericalFog fog, Sphere sphere, glm::vec3 light_direction, glm::vec3 intersection_point, glm::vec3 norm)
	{
		// Clamp the distance; I don't remember why this was done
		float distance = glm::length(sphere.collider.position - light.position);
		if(distance > 2 * fog.collider.radius)
		{
			distance = 2 * fog.collider.radius;
		}

		// Fog relies on a probabalistic model, so define that here and decide whether to use fog or not here
		float probability_no_interaction = exp(-1.0f * distance * (fog.absorption + fog.scattering));
		float random_num				 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		if(random_num > probability_no_interaction)
		{
			distance		= glm::length(light.position - intersection_point);
			float intensity = 1.0f / powf(glm::length(distance), 2.0f);

			// If it's used, multiply the sphere material's diffuse property by the light colour and intensity, and then
			// max it with the dot product of the normal vector of the sphere and the light's direction
			return sphere.material.diffuse * light.colour * intensity * std::max(0.0f, glm::dot(norm, light_direction));
		}

		// Otherwise, just run the standard phase function to take the fog into account,
		glm::vec3 new_light_direction = scattering_phase_function(light_direction, fog.scattering);
		return fog.albedo * light.colour * std::max(0.0f, glm::dot(norm, new_light_direction));
	}

	// Add in the diffuse shading based on material properties of the sphere
	glm::vec3 diffuse_shading(Scene scene, Sphere sphere, glm::vec3 intersection_point, glm::vec3 norm)
	{
		glm::vec3 colour = glm::vec3(0.0f, 0.0f, 0.0f);
		// handle fog in difffuse and specular, maybe?
		for(unsigned int i = 0; i < scene.point_lights.size(); i++)
		{
			if(!scene.use_shadows || !shadow(scene, intersection_point, scene.point_lights[i]))
			{
				glm::vec3 light_direction = glm::normalize(scene.point_lights[i].position - intersection_point);

				// Iteratively get fog colour
				if(scene.spherical_fog.size() > 0)
				{
					for(int j = 0; j < scene.spherical_fog.size(); j++)
					{
						colour += spherical_fog_shading(scene.point_lights[i], scene.spherical_fog[j], sphere, light_direction, intersection_point, norm);
					}
				}

				else
				{
					// without fog, make the colour the same as the if() case in the spherical_fog equation
					float distance	= glm::length(scene.point_lights[i].position - intersection_point);
					float intensity = 1.0f / powf(glm::length(distance), 2.0f);

					colour += sphere.material.diffuse * scene.point_lights[i].colour * intensity * std::max(0.0f, glm::dot(norm, light_direction));
				}
			}
		}

		for(unsigned int i = 0; i < scene.directional_lights.size(); i++)
		{
			if(!scene.use_shadows || !shadow(scene, intersection_point, scene.directional_lights[i]))
			{
				glm::vec3 light_direction = glm::normalize(scene.directional_lights[i].direction);
				colour += sphere.material.diffuse * scene.directional_lights[i].colour * std::max(0.0f, glm::dot(norm, light_direction));
			}
		}

		return colour;
	}


	glm::vec3 specular_shading(Scene scene, Sphere sphere, glm::vec3 intersection_point, glm::vec3 norm)
	{
		glm::vec3 colour		 = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec3 view_direction = glm::normalize(scene.camera.position - intersection_point);

		for(unsigned int i = 0; i < scene.point_lights.size(); i++)
		{
			if(!scene.use_shadows || !shadow(scene, intersection_point, scene.point_lights[i]))
			{
				// This is an interpretation of the specular highlight vectors
				glm::vec3 light_direction = glm::normalize(scene.point_lights[i].position - intersection_point);
				glm::vec3 half_vector	  = (view_direction + light_direction) / glm::length(view_direction + light_direction);

				if(scene.spherical_fog.size() > 0)
				{
					for(int j = 0; j < scene.spherical_fog.size(); j++)
					{
						colour += spherical_fog_shading(scene.point_lights[i], scene.spherical_fog[j], sphere, light_direction, intersection_point, norm);
					}
				}

				else
				{
					float distance	= glm::length(scene.point_lights[i].position - intersection_point);
					float intensity = 1.0f / powf(glm::length(distance), 2.0f);

					// Specular lighting equation; iteratively add it for every point light
					colour += sphere.material.specular * scene.point_lights[i].colour * intensity * powf(std::max(0.0f, glm::dot(norm, half_vector)), sphere.material.power);
				}
			}
		}

		for(unsigned int i = 0; i < scene.directional_lights.size(); i++)
		{
			printf("checking size isn't 0\n");
			if(!scene.use_shadows || !shadow(scene, intersection_point, scene.directional_lights[i]))
			{
				glm::vec3 light_direction = glm::normalize(scene.directional_lights[i].direction);
				glm::vec3 half_vector	  = (view_direction + light_direction) / glm::length(view_direction + light_direction);
				colour += sphere.material.specular * scene.directional_lights[i].colour * powf(std::max(0.0f, glm::dot(norm, half_vector)), sphere.material.power);
			}
		}

		return colour;
	}


	glm::vec3 reflect_direction(glm::vec3 light_direction, glm::vec3 normal)
	{
		return glm::normalize(light_direction - 2.0f * glm::dot(light_direction, normal) * normal);
	}


	glm::vec3 refraction(glm::vec3 dir, glm::vec3 normal, Sphere sphere)
	{
		float k = 1.0f - powf(sphere.material.ior, 2.0f) * (1.0f - powf(glm::dot(dir, normal), 2.0f));

		if(k < 0.0f)
		{
			return glm::vec3(0.0f, 0.0f, 0.0f);
		}

		return sphere.material.ior * dir - (sphere.material.ior * glm::dot(dir, normal) + sqrtf(k)) * normal;
	}


	float fresnel(glm::vec3 ray_direction, glm::vec3 normal, Sphere sphere)
	{
		float cos_internal = clamp(-1.0f, 1.0f, glm::dot(ray_direction, normal));
		float et		   = 1.0f;
		float ior		   = sphere.material.ior;


		if(cos_internal > 0)
		{
			std::swap(et, ior);
		}

		float sint = et / ior * sqrt(std::max(0.0f, 1.0f - powf(cos_internal, 2.0f)));

		// total reflection)
		if(sint >= 1.0f)
		{
			return 1.0f;
		}

		float cos_theta = sqrt(std::max(0.0f, 1 - powf(sint, 2.0f)));
		cos_internal	= abs(cos_internal);

		// Schlick approximation
		float Rs = ((ior * cos_internal) - (et * cos_theta)) / ((ior * cos_internal) + (et * cos_theta));
		float Rp = ((et * cos_internal) - (ior * cos_theta)) / ((ior * cos_internal) + (et * cos_theta));

		return (powf(Rs, 2.0f) + powf(Rp, 2.0f)) / 2.0f;
	}
}; // namespace bp


#endif
