CXX=clang++ -fopenmp -lSDL2
CXXFLAGS=-g -std=c++11 -fopenmp -Wall -I/usr/include/SDL2/

all: raytracer
clean:
	rm -f *.o *.h.gch raytracer
test: raytracer
	./raytracer
	display raytrace.bmp
.PHONY: all clean test

raytracer: main.o scene.o 
	$(CXX) $(LDFLAGS) -o $(@) $(^)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $(@) $(<)
