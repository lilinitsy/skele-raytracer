# Raytracer
Minimalist raytracer with global illumination via Monte-Carlo path-tracing support

# Branches
1. raytracer-original
Parallelizing only the ray generation, such that there is only one ray devoted to one thread, and then each thread will in turn operate over all intersections.
2. raytracer-shared-const-mem
Using multiple threads for calculating collisions and the usage of shared / constant memory.
3. raytracer-input-binning
Use input binning.

# Features
Blinn-Phong shading with diffuse shading and specular highlights is featured. Reflections, refractions, and the fresnel effect are also included as per the Blinn-Phong shading model.

Monte-Carlo path tracing was used to provide global illumination, although the technique is inherently slow and prone to noise.
Parallelization with OpenMP dramatically speeds this up.

Jittered supersampling in an n by n grid is also implemented, and turned on with a command-line option. The advantage of this is that it reduces aliasing, at the cost of speed. Parallelization dramatically speeds this up.

There is optionally a continuously updating visual display, using SDL 2.0. As the image is developed, it will show up on-screen. This comes at the cost of both a large speed slowdown, as well as an even greater one because it's hard to parallelize this.

# Command-line Options
Compile with ``make clean``, ``make``, and run ``./raytracer --path path --output destination`` to run. Renders are written in the PPM format, which can be viewed with the ``display`` command.

Many features are enabled to run the raytracer with different options.
Minimally, a path to the scene file and an output destination must be specified. For example,

	./raytracer --path scenes/spheres1.scn --output renders/cool_sphere.ppm

The options followed by their argument types are::
```
--path		// string path to the .scn
--output	// string destination for render ppm
--width 	// int width of the render and SDL display. Default 1920
--height 	// int height of the render and SDL display. Default 1080
--fov		// float field of view of the scene. Default 60.0
--gillum	// int number of paths traced per ray follows
--jsample 	// int size of grid for jittered supersampling
--depth		// int max depth for reflection rays tracing. Default 1 (1 reflection)
--parallel	// pass true to tur on parallelization and turns off the visual display
--shadow	// nothing passed after, turns shadows on.
```