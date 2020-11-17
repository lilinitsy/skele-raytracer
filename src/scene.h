#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <vector>

#include "Fog.h"
#include "camera.h"
#include "lights.h"
#include "shapes.h"


struct Scene
{
	int width = 1920, height = 1080; // used for res
	std::vector<Sphere> spheres;
	std::vector<Triangle> triangles;
	std::vector<PointLight> point_lights;
	std::vector<DirectionalLight> directional_lights;
	AmbientLight ambient_light;

	Camera camera;
	std::vector<SphericalFog> spherical_fog;
	glm::vec3 background;
	int maxDepth	 = 1;
	bool use_shadows = false;
};

void toString(Scene &scene);
Scene parseScene(std::string fileName);

#endif