#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "scene.h"


using std::string;


Scene parseScene(string fileName)
{
    // open the file containing the scene description
	Scene the_scene;
	Material mat;

    FILE* fp = fopen(fileName.c_str(), "r");
    char line[1024]; //Assumes no line is longer than 1024 characters!

    // check for errors in opening the file
    if(fp == NULL)
    {
        printf("Can't open file '%s'\n", fileName.c_str());
        exit(0);
    }

    //Loop through reading each line
    while(fgets(line,1024,fp))
    { //Assumes no line is longer than 1024 characters!
        if(line[0] == '#')
        {
            printf("Skipping comment: %s\n", line);
            continue;
        }

        char command[100];
        int fieldsRead = sscanf(line,"%s ",command); //Read first word in the line (i.e., the command type)

        if(fieldsRead < 1)
        { //No command read
            //Blank line
            continue;
        }

        if(strcmp(command, "sphere") == 0)
        { //If the command is a sphere command
            float x,y,z,r;
            sscanf(line,"sphere %f %f %f %f", &x, &y, &z, &r);
            printf("Sphere as position (%f,%f,%f) with radius %f\n",x,y,z,r);
			Sphere sphere = Sphere(glm::vec3(x, y, z), r, mat);

			the_scene.spheres.push_back(sphere);
        }

        else if(strcmp(command, "camera") == 0)
        {
            float   posX, posY, posZ,
                    viewDirX, viewDirY, viewDirZ,
                    ux, uy, uz, halfHeightAngle;
            sscanf(line, "camera %f %f %f %f %f %f %f %f %f %f\n", &posX, &posY, &posZ, &viewDirX, &viewDirY, &viewDirZ, &ux, &uy, &uz, &halfHeightAngle);
            printf("Camera with position (%f, %f, %f) with viewing direction (%f, %f, %f), up glm::vec3 (%f, %f, %f), and halfHeightAngle %f\n", posX, posY, posZ, viewDirX, viewDirY, viewDirZ, ux, uy, uz, halfHeightAngle);
			Camera cam = Camera(glm::vec3(posX, posY, posZ), glm::vec3(viewDirX, viewDirY, viewDirZ), glm::vec3(ux, uy, uz), halfHeightAngle);
			glm::normalize(cam.direction);
			glm::normalize(cam.up);
			the_scene.camera = cam;

			FILE *spheres1 = fopen("simplesphere.txt", "w");


					fprintf(spheres1, "Up: (%f, %f, %f)\t Position: (%f, %f, %f)\n",
											the_scene.camera.up.x, 	the_scene.camera.up.y, 	the_scene.camera.up.z,
											the_scene.camera.position.x, the_scene.camera.position.y, the_scene.camera.position.z);

			fclose(spheres1);

		}

        else if(strcmp(command, "film_resolution") == 0)
        {
            sscanf(line, "film_resolution %d %d", &the_scene.width, &the_scene.height);
            printf("Film resolution: %d x %d\n", the_scene.width, the_scene.height);
        }

        else if(strcmp(command, "background") == 0)
        { //If the command is a background command
            float r,g,b;
            sscanf(line,"background %f %f %f", &r, &g, &b);
            printf("Background color of (%f,%f,%f)\n",r,g,b);
			the_scene.background = glm::vec3(r, g, b);
        }

        else if(strcmp(command, "material") == 0)
        {
            float   ambR, ambG, ambB,
                    diffR, diffG, diffB,
                    specR, specG, specB, phongCos,
                    tranR, tranG, tranB, ior; // hwk 2 will ignore these 4
            /** spec's should be used to determine the colour
                of a highlight and the contribution of the
                reflected ray.
            **/

            sscanf(line, "material %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
                            &ambR, &ambG, &ambB, &diffR, &diffG, &diffB, &specR, &specG, &specB, &phongCos, &tranR, &tranG, &tranB, &ior);
            printf("material properties with ambient colour (%f, %f, %f), diffuse colour (%f, %f, %f), specular colour (%f, %f, %f), phong Cosine power %f, transmissive colour (%f, %f, %f), index of refraction %f\n",
                                                            ambR, ambG, ambB,           diffR, diffG, diffB,             specR, specG, specB,            phongCos,               tranR, tranG, tranB,            ior);
			mat = Material(glm::vec3(ambR, ambG, ambB),
                    glm::vec3(diffR, diffG, diffB),
                    glm::vec3(specR, specG, specB),
                    glm::vec3(tranR, tranG, tranB), phongCos, ior);
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
	/*		DirectionalLight light();
			light.colour = Colour(r, g, b);
			light.x = x;
			light.y = y;
			light.z = z;
			the_scene.dirLight.push_back(light); */
		}

        else if(strcmp(command, "point_light") == 0)
        {
            float r, g, b, x, y, z;
            sscanf(line, "point_light %f %f %f %f %f %f", &r, &g, &b, &x, &y, &z);
            printf("point light colour (%f, %f, %f), located at (%f, %f, %f)\n", r, g, b, x, y, z);

			PointLight point_light = PointLight(glm::vec3(x, y, z), glm::vec3(r, g, b));
			printf("Point light colour (%f, %f, %f), located at (%f, %f, %f)\n",
				point_light.colour.r, point_light.colour.g, point_light.colour.b,
				point_light.position.x, point_light.position.y, point_light.position.z);
			the_scene.point_lights.push_back(point_light);
		}

        else if(strcmp(command, "ambient_light") == 0)
        {
            float r, g, b;
            sscanf(line, "ambient_light %f %f %f", &r, &g, &b);
            printf("Ambient light colour (%f, %f, %f)\n", r, g, b);

			the_scene.ambient_light.colour.r += r;
			the_scene.ambient_light.colour.g += g;
			the_scene.ambient_light.colour.b += b;
        }

        else if(strcmp(command, "max_depth") == 0)
        {
            float n;
            sscanf(line, "max_depth %f", &n);
            printf("max_depth %f\n", n);
			the_scene.maxDepth = n;
        }

        else if(strcmp(command, "output_image") == 0)
        { //If the command is an output_image command
            char outFile[1024];
            sscanf(line,"output_image %s", outFile);
            printf("Render to file named: %s\n", outFile);
        }

        else
        {
            printf("WARNING. Do not know command: %s\n",command);
        }
    }

	printf("\n\n\n\n\n\n");
	for(unsigned int i = 0; i < the_scene.spheres.size(); i++)
	{
		the_scene.spheres[i].to_string();
	}

	return the_scene;
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