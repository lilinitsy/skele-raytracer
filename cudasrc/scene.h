#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <vector>

#include "Fog.h"
#include "camera.h"
#include "lights.h"
#include "shapes.h"
#include "vec3.h"

struct Scene
{
	int width = 1920;
	int height = 1080;
	std::vector<Sphere> spheres;
	std::vector<vecmath::vec3> vertices;
	std::vector<Triangle> triangles;
	std::vector<PointLight> point_lights;
	std::vector<DirectionalLight> directional_lights;
	AmbientLight ambient_light;

	Camera camera;
	std::vector<SphericalFog> spherical_fog;
	vecmath::vec3 background;
	int maxDepth	 = 1;
	bool use_shadows = false;

	size_t size()
	{
		size_t size = 0;
		size += 2 * sizeof(int); // width, height
		size += spheres.size() * sizeof(Sphere);
		size += vertices.size() * sizeof(vecmath::vec3);
		size += triangles.size() * sizeof(Triangle);
		size += point_lights.size() * sizeof(PointLight);
		size += directional_lights.size() * sizeof(DirectionalLight);
		size += sizeof(AmbientLight);
		size += sizeof(Camera);
		size += spherical_fog.size() * sizeof(SphericalFog);
		size += sizeof(vecmath::vec3); // background
		size += sizeof(int); // maxdepth;
		size += sizeof(bool); // use_shadows

		return size;
	}
};

using std::string;


Scene parseScene(string fileName)
{
	// open the file containing the scene description
	Scene scene;
	Material mat;

	FILE *fp = fopen(fileName.c_str(), "r");
	char line[1024]; //Assumes no line is longer than 1024 characters!

	// check for errors in opening the file
	if(fp == NULL)
	{
		printf("Can't open file '%s'\n", fileName.c_str());
		exit(0);
	}

	//Loop through reading each line
	while(fgets(line, 1024, fp))
	{ //Assumes no line is longer than 1024 characters!
		if(line[0] == '#')
		{
			printf("Skipping comment: %s\n", line);
			continue;
		}

		char command[100];
		int fieldsRead = sscanf(line, "%s ", command); //Read first word in the line (i.e., the command type)

		if(fieldsRead < 1)
		{ //No command read
			//Blank line
			continue;
		}

		if(strcmp(command, "sphere") == 0)
		{ //If the command is a sphere command
			float x, y, z, r;
			sscanf(line, "sphere %f %f %f %f", &x, &y, &z, &r);
			printf("Sphere as position (%f, %f, %f) with radius %f\n", x, y, z, r);
			Sphere sphere;
			sphere.collider.position = vecmath::vec3(x, y, z);
			sphere.collider.radius	 = r;
			sphere.material			 = mat;

			scene.spheres.push_back(sphere);
		}

		else if(strcmp(command, "vertex") == 0)
		{
			float x, y, z;
			sscanf(line, "vertex %f %f %f", &x, &y, &z);
			printf("Vertex at position (%f, %f, %f)\n", x, y, z);
			scene.vertices.push_back(vecmath::vec3(x, y, z));
		}

		else if(strcmp(command, "triangle") == 0)
		{
			float v0, v1, v2;
			sscanf(line, "triangle %f %f %f", &v0, &v1, &v2);
			printf("Triangle with vertices (%f, %f, %f)\n", v0, v1, v2);
			vecmath::vec3 first_vert  = scene.vertices[v0];
			vecmath::vec3 second_vert = scene.vertices[v1];
			vecmath::vec3 third_vert  = scene.vertices[v2];

			Triangle triangle;
			triangle.v0		  = first_vert;
			triangle.v1		  = second_vert;
			triangle.v2		  = third_vert;
			triangle.material = mat;
			scene.triangles.push_back(triangle);
		}

		else if(strcmp(command, "camera") == 0)
		{
			float posX, posY, posZ,
				viewDirX, viewDirY, viewDirZ,
				ux, uy, uz, halfHeightAngle;
			sscanf(line, "camera %f %f %f %f %f %f %f %f %f %f\n", &posX, &posY, &posZ, &viewDirX, &viewDirY, &viewDirZ, &ux, &uy, &uz, &halfHeightAngle);
			printf("Camera with position (%f, %f, %f) with viewing direction (%f, %f, %f), up vecmath::vec3 (%f, %f, %f), and halfHeightAngle %f\n", posX, posY, posZ, viewDirX, viewDirY, viewDirZ, ux, uy, uz, halfHeightAngle);
			Camera cam = Camera(vecmath::vec3(posX, posY, posZ), vecmath::vec3(viewDirX, viewDirY, viewDirZ), vecmath::vec3(ux, uy, uz), halfHeightAngle);
			vecmath::normalize(cam.direction);
			vecmath::normalize(cam.up);
			scene.camera = cam;

			FILE *spheres1 = fopen("simplesphere.txt", "w");

			fprintf(spheres1, "Up: (%f, %f, %f)\t Position: (%f, %f, %f)\n",
					scene.camera.up.x, scene.camera.up.y, scene.camera.up.z,
					scene.camera.position.x, scene.camera.position.y, scene.camera.position.z);

			fclose(spheres1);
		}

		else if(strcmp(command, "film_resolution") == 0)
		{
			sscanf(line, "film_resolution %d %d", &scene.width, &scene.height);
			printf("Film resolution: %d x %d\n", scene.width, scene.height);
		}

		else if(strcmp(command, "background") == 0)
		{ //If the command is a background command
			float r, g, b;
			sscanf(line, "background %f %f %f", &r, &g, &b);
			printf("Background color of (%f,%f,%f)\n", r, g, b);
			scene.background = vecmath::vec3(r, g, b);
		}

		else if(strcmp(command, "material") == 0)
		{
			float ambR, ambG, ambB,
				diffR, diffG, diffB,
				specR, specG, specB, phongCos,
				tranR, tranG, tranB, ior;

			sscanf(line, "material %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
				   &ambR, &ambG, &ambB, &diffR, &diffG, &diffB, &specR, &specG, &specB, &phongCos, &tranR, &tranG, &tranB, &ior);
			printf("material properties with ambient colour (%f, %f, %f), diffuse colour (%f, %f, %f), specular colour (%f, %f, %f), phong Cosine power %f, transmissive colour (%f, %f, %f), index of refraction %f\n",
				   ambR, ambG, ambB, diffR, diffG, diffB, specR, specG, specB, phongCos, tranR, tranG, tranB, ior);

			mat.ambient		 = vecmath::vec3(ambR, ambG, ambB);
			mat.diffuse		 = vecmath::vec3(diffR, diffG, diffB);
			mat.specular	 = vecmath::vec3(specR, specG, specB);
			mat.transmissive = vecmath::vec3(tranR, tranG, tranB);
			mat.power		 = phongCos;
			mat.ior			 = ior;
		}

		else if(strcmp(command, "directional_light") == 0)
		{
			float r, g, b, x, y, z;
			sscanf(line, "directional_light %f %f %f %f %f %f", &r, &g, &b, &x, &y, &z);
			printf("directional light colour (%f, %f, %f), direction (%f, %f, %f)\n", r, g, b, x, y, z);
			if(r > 1)
			{
				r = 1;
			}
			if(g > 1)
			{
				g = 1;
			}
			if(b > 1)
			{
				b = 1;
			}

			DirectionalLight directional_light;
			directional_light.direction = vecmath::vec3(x, y, z);
			directional_light.colour	= vecmath::vec3(r, g, b);
			printf("Directional light colour (%f, %f %f) with direction (%f, %f, %f)\n",
				   directional_light.colour.x, directional_light.colour.y, directional_light.colour.z,
				   directional_light.direction.x, directional_light.direction.x, directional_light.direction.z);
		}

		else if(strcmp(command, "point_light") == 0)
		{
			float r, g, b, x, y, z;
			sscanf(line, "point_light %f %f %f %f %f %f", &r, &g, &b, &x, &y, &z);
			printf("point light colour (%f, %f, %f), located at (%f, %f, %f)\n", r, g, b, x, y, z);

			PointLight point_light;
			point_light.position = vecmath::vec3(x, y, z);
			point_light.colour	 = vecmath::vec3(r, g, b);

			printf("Point light colour (%f, %f, %f), located at (%f, %f, %f)\n",
				   point_light.colour.x, point_light.colour.y, point_light.colour.z,
				   point_light.position.x, point_light.position.y, point_light.position.z);
			scene.point_lights.push_back(point_light);
		}

		else if(strcmp(command, "ambient_light") == 0)
		{
			float r, g, b;
			sscanf(line, "ambient_light %f %f %f", &r, &g, &b);
			printf("Ambient light colour (%f, %f, %f)\n", r, g, b);

			scene.ambient_light.colour.x += r;
			scene.ambient_light.colour.y += g;
			scene.ambient_light.colour.z += b;
		}

		else if(strcmp(command, "max_depth") == 0)
		{
			float n;
			sscanf(line, "max_depth %f", &n);
			printf("max_depth %f\n", n);
			scene.maxDepth = n;
		}

		else if(strcmp(command, "output_image") == 0)
		{ //If the command is an output_image command
			char outFile[1024];
			sscanf(line, "output_image %s", outFile);
			printf("Render to file named: %s\n", outFile);
		}

		else if(strcmp(command, "spherical_fog") == 0)
		{
			float x, y, z, rad, r, g, b, s, abso;
			sscanf(line, "fog %f %f %f %f %f %f %f %f %f", &x, &y, &z, &rad, &r, &g, &b, &s, &abso);
			scene.spherical_fog.push_back(SphericalFog(s, abso, vecmath::vec3(r, g, b), rad, vecmath::vec3(x, y, z)));
		}

		else
		{
			printf("WARNING. Do not know command: %s\n", command);
		}
	}

	printf("\n\n\n\n\n\n");
	for(unsigned int i = 0; i < scene.spheres.size(); i++)
	{
		scene.spheres[i].to_string();
	}

	return scene;
}

void toString(Scene &scene)
{
	printf("Width: %d\n", scene.width);
	printf("Height: %d\n", scene.height);
	printf("Camera: Position: (%f, %f, %f)\t Direction: (%f, %f, %f)\tUp: (%f, %f, %f)\tRight: (%f, %f, %f)\n",
		   scene.camera.position.x, scene.camera.position.y, scene.camera.position.z,
		   scene.camera.direction.x, scene.camera.direction.y, scene.camera.direction.z,
		   scene.camera.up.x, scene.camera.up.y, scene.camera.up.z,
		   scene.camera.right.x, scene.camera.right.y, scene.camera.right.z);
}

#endif