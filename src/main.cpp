#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <cstring>

#include <SDL.h>
#include <SDL_opengl.h>

#include "raytrace.h"


struct Options
{
    bool monte_carlo = false;
    bool visual = true;
    float fov = 60;
    short num_path_traces = 1;
    short grid_size = 0;
    int max_depth = 3;

    void to_string()
    {
        printf("\n\nMonte carlo: %d\nvisual display: %d\nfov: %f\nnum paths traced: %d\nsupersample grid size: %d\nmax depth: %d\n", monte_carlo, visual, fov, num_path_traces, grid_size, max_depth);
    }
};



void generate_rays_parallel(Scene scene, Options option)
{
    glm::vec3 **image = new glm::vec3*[scene.height];

    for(int i = 0; i < scene.height; i++)
    {
        image[i] = new glm::vec3[scene.width];
    }

    #pragma omp parallel for
    for (int y = 0; y < scene.height; y++)
	{
        for (int x = 0; x < scene.width; x++)
		{
            float inv_width = 1 / float(scene.width);
            float inv_height = 1 / float(scene.height);
            float aspect_ratio = scene.width / float(scene.height);
            float angle = tan(M_PI * 0.5 * option.fov / 180.);

            if(option.grid_size > 0)
            {
                for(int i = 0; i < option.grid_size; i++)
                {
                    for(int j = 0; j < option.grid_size; j++)
                    {
                        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        float u = (2 * ((x + r) * inv_width) - 1) * angle * aspect_ratio;
                        float v = (1 - 2 * ((y + r) * inv_height)) * angle;

            			glm::vec3 ray_dir(scene.camera.direction + u * scene.camera.right + v * scene.camera.up);
            			glm::normalize(ray_dir);
            			Ray ray = Ray(scene.camera.position, ray_dir);

                        image[y][x] += shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces);
                    }
                }

                image[y][x] /= (option.grid_size * option.grid_size);
            }

            else
            {
                float u = (2 * ((x + 0.5) * inv_width) - 1) * angle * aspect_ratio;
                float v = (1 - 2 * ((y + 0.5) * inv_height)) * angle;

                glm::vec3 ray_dir(scene.camera.direction + u * scene.camera.right + v * scene.camera.up);
                glm::normalize(ray_dir);
                Ray ray = Ray(scene.camera.position, ray_dir);

                image[y][x] = shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces);
            }
        }
    }

    std::ofstream ofs("raytrace.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << scene.width << " " << scene.height << "\n255\n";

	for (int i = 0; i < scene.height; i++)
	{
        for(int j = 0; j < scene.width; j++)
        {
            ofs << (unsigned char)(std::min(float(1), image[i][j].x) * 255) <<
                   (unsigned char)(std::min(float(1), image[i][j].y) * 255) <<
                   (unsigned char)(std::min(float(1), image[i][j].z) * 255);
        }
    }

    ofs.close();
    printf("***\nWROTE TO PPM\n***\n");

    delete[] image;
}



void generate_rays(Scene scene, Options option)
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Event event;
    SDL_Window *window = SDL_CreateWindow("Raytracer", 100, 100, scene.width, scene.height, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

    glm::vec3 **image = new glm::vec3*[scene.height];

    for(int i = 0; i < scene.height; i++)
    {
        image[i] = new glm::vec3[scene.width];
    }

//    #pragma omp parallel for
    for (int y = 0; y < scene.height; y++)
	{
        for (int x = 0; x < scene.width; x++)
		{
            float inv_width = 1 / float(scene.width);
            float inv_height = 1 / float(scene.height);
            float aspect_ratio = scene.width / float(scene.height);
            float angle = tan(M_PI * 0.5 * option.fov / 180.);

            if(option.grid_size > 0)
            {
                for(int i = 0; i < option.grid_size; i++)
                {
                    for(int j = 0; j < option.grid_size; j++)
                    {
                        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        float u = (2 * ((x + r) * inv_width) - 1) * angle * aspect_ratio;
                        float v = (1 - 2 * ((y + r) * inv_height)) * angle;

            			glm::vec3 ray_dir(scene.camera.direction + u * scene.camera.right + v * scene.camera.up);
            			glm::normalize(ray_dir);
            			Ray ray = Ray(scene.camera.position, ray_dir);

                        image[y][x] += shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces);
                    }
                }
                image[y][x] /= (option.grid_size * option.grid_size);
            }

            else
            {
                float u = (2 * ((x + 0.5) * inv_width) - 1) * angle * aspect_ratio;
                float v = (1 - 2 * ((y + 0.5) * inv_height)) * angle;

                glm::vec3 ray_dir(scene.camera.direction + u * scene.camera.right + v * scene.camera.up);
                glm::normalize(ray_dir);
                Ray ray = Ray(scene.camera.position, ray_dir);

                image[y][x] = shade(ray, scene, option.max_depth, option.monte_carlo, option.num_path_traces);
            }
            SDL_SetRenderDrawColor(renderer,    (unsigned char)(std::min(float(1), image[y][x].x) * 255),
                                                (unsigned char)(std::min(float(1), image[y][x].y) * 255),
                                                (unsigned char)(std::min(float(1), image[y][x].z) * 255), 255);
            SDL_RenderDrawPoint(renderer, x, y);

            if(SDL_PollEvent(&event) && event.type == SDL_QUIT)
            {
                SDL_DestroyRenderer(renderer);
                SDL_DestroyWindow(window);
                SDL_Quit();
            }
        }

        SDL_RenderPresent(renderer);
    }

    std::ofstream ofs("raytrace.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << scene.width << " " << scene.height << "\n255\n";

	for (int i = 0; i < scene.height; i++)
	{
        for(int j = 0; j < scene.width; j++)
        {
            ofs << (unsigned char)(std::min(float(1), image[i][j].x) * 255) <<
                   (unsigned char)(std::min(float(1), image[i][j].y) * 255) <<
                   (unsigned char)(std::min(float(1), image[i][j].z) * 255);
        }
    }
    ofs.close();


    printf("***\nWROTE TO PPM\n***\n");

    while(1)
    {
        if(SDL_PollEvent(&event) && event.type == SDL_QUIT)
        {
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
        }
    }



    delete[] image;
}


int main(int argc, char *argv[])
{

    Options option;
    Scene scene;
    int width = scene.width;
    int height = scene.height;
    char *path;
    bool path_passed = false;

    for(int i = 0; i < argc; i++)
    {
        if(strcmp(argv[i], "--gillum") == 0)
        {
            if(i + 1 < argc)
            {
                option.monte_carlo = true;
                option.num_path_traces = atoi(argv[i + 1]);
            }

            else
            {
                std::cerr << "gillum takes an int after flag for the number of paths traced" << std::endl;
            }
        }

        if(strcmp(argv[i], "--fov") == 0)
        {
            if(i + 1 < argc)
            {
                option.fov = atof(argv[i + 1]);
            }

            else
            {
                std::cerr << "fov takes a float (degrees) after flag for the field of view" << std::endl;
                return 0;
            }
        }

        if(strcmp(argv[i], "--jsample") == 0)
        {
            if(i + 1 < argc)
            {
                option.grid_size = atoi(argv[i + 1]);
            }

            else
            {
                std::cerr << "jsample takes an int after flag for the supersampling grid size" << std::endl;
                return 0;
            }
        }

        if(strcmp(argv[i], "--width") == 0)
        {
            if(i + 1 < argc)
            {
                width = atoi(argv[i + 1]);
            }

            else
            {
                std::cerr << "width takes an int after flag for the width" << std::endl;
                return 0;
            }
        }

        if(strcmp(argv[i], "--height") == 0)
        {
            if(i + 1 < argc)
            {
                height = atoi(argv[i + 1]);
            }

            else
            {
                std::cerr << "height takes an int after flag for the width" << std::endl;
                return 0;
            }
        }

        if(strcmp(argv[i], "--depth") == 0)
        {
            if(i + 1 < argc && atoi(argv[i + 1]) > 0)
            {
                option.max_depth = atoi(argv[i + 1]);
            }

            else
            {
                std::cerr << "depth takes a positive int after flag for the max depth" << std::endl;
                return 0;
            }
        }

        if(strcmp(argv[i], "--parallel") == 0)
        {
            if(i + 1 < argc && strcmp(argv[i + 1], "true") == 0)
            {
                option.visual = false;
            }

            if(i + 1 < argc && strcmp(argv[i + 1], "false") == 0)
            {
                option.visual = true;
            }
        }

        if(strcmp(argv[i], "--path") == 0)
        {
            if(i + 1 < argc)
            {
                path = argv[i + 1];
                path_passed = true;
            }

            else
            {
                std::cerr << "path must be passed after --path" << std::endl;
                return 0;
            }
        }
    }

    if(!path_passed)
    {
        std::cerr << "no scene file was passed. Pass with --path path_to_scn" << std::endl;
        return 0;
    }

    scene = parseScene(path);
    scene.width = width;
    scene.height = height;

    option.to_string();

    srand((unsigned)time(0));

    if(option.visual)
    {
        generate_rays(scene, option);
    }

    else
    {
        generate_rays_parallel(scene, option);
    }

	return 0;
}
