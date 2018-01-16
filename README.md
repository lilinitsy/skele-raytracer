# skele-raytracer
Minimalist raytracer with global illumination via Monte-Carlo path-tracing support

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
--parallel	// nothing passed after, turns on parallelization and turns off the visual display
--shadow	// nothing passed after, turns shadows on.
```

# Example Images
Multiple renders are provided in renders/. Default scene files are found in /scenes. PNG's of some selected images are found in the example_pngs folder in both renders/shadows and rendres/no_shadows. Times given in the time_notes.txt file are running on a quadcore laptop CPU at 2.5ghz.

![bp_jsample5] (https://github.com/lilinitsy/skele-raytracer/blob/master/renders/no_shadow/sample_pngs/bp_jsample5.png)
Above is an example of a rendering of scenes/spheres2.scn without shadows. With OpenMP enabled, rendering at a resolution of 1920x1080 pixels, and 25 sample / pixel jittered supersampling, it finished in ``3 minutes 47.85 seconds``.

The same scene without any supersampling finished rendering in ``9.642`` seconds. Aliasing is much more apparent.

![bp-gillum16] (https://github.com/lilinitsy/skele-raytracer/blob/master/renders/no_shadow/sample_pngs/bp_gillum16.png)
Here's the same render with Monte-Carlo path traced global illumination, shot with 16 paths traced. It rendered at 1920x1080 in ``4 minutes 36.87 seconds``.

![bp-jsample5-shadows] (https://github.com/lilinitsy/skele-raytracer/blob/master/renders/shadows/sample_pngs/bp_jsample5_parallel_shadows.png)
With shadows enabled, this render at 25 samples / pixel jittered supersampling finished in ``6 minutes 54.15 seconds``.  The same render without supersampling, but with shadows, finished in ``16.137 seconds``.

![bp-gillum16-shadows] (https://github.com/lilinitsy/skele-raytracer/blob/master/renders/shadows/sample_pngs/bp_parallel_shadows_gillum16.png)
Comparable to image 2, this render has the same global illumination with 16 paths traced, but shadows are enabled. It finished in ``6 minutes 16.81 seconds``. Interestingly, the issue with shadows is not so apparent in this one, due to the global illumination and noise covering up some of it, besides the center sphere.


# Issues
Shadows aren't 100% working. The edges are sometimes the only portions that appear, but this is inconsistent. This is apparent in the second example image.
