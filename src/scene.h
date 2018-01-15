#pragma once

#include <string>
#include <vector>
#include "shapes.h"
#include "lights.h"
#include "camera.h"


using std::string;
using std::vector;


struct Scene
{
	int width = 1920, height = 1080; // used for res
	vector<Sphere> spheres;
	vector<Triangle> triangles;
	vector<PointLight> point_lights;
	vector<DirectionalLight> directional_lights;
	Camera camera;
	AmbientLight ambient_light = AmbientLight();
	glm::vec3 background;
	int maxDepth = 1;
	bool use_shadows = false;
};

void toString(Scene &scene);
Scene parseScene(string fileName);
