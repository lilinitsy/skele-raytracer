# Team Member
Ville Cantory, Yihan Zhou, Zitao Yang, Kun Xue, Roman Woolery

# Raytracer
Minimalist raytracer with global illumination via Monte-Carlo path-tracing support

# Branches
1. raytracer-original  
Parallelizing only the ray generation, such that there is only one ray devoted to one thread, and then each thread will in turn operate over all intersections.
2. raytracer-shared-const-mem  
Using multiple threads for calculating collisions and the usage of shared / constant memory.
3. raytracer-input-binning  
Using input binning.

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
--parallel	// pass true to tur on parallelization and turns off the visual display
--shadow	// nothing passed after, turns shadows on.
```
