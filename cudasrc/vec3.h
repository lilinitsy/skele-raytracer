#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>

namespace vecmath
{

	struct vec3
	{
		float x;
		float y;
		float z;


		__host__ __device__ vec3()
		{
		}
		__host__ __device__ vec3(float e0, float e1, float e2)
		{
			x = e0;
			y = e1;
			z = e2;
		}

		__host__ __device__ inline vec3 &operator+()
		{
			return *this;
		}
		__host__ __device__ inline vec3 operator-()
		{
			return vec3(-x, -y, -z);
		}


		__host__ __device__ inline vec3 &operator+=(const vec3 &v2);
		__host__ __device__ inline vec3 &operator-=(const vec3 &v2);
		__host__ __device__ inline vec3 &operator*=(const vec3 &v2);
		__host__ __device__ inline vec3 &operator/=(const vec3 &v2);
		__host__ __device__ inline vec3 &operator*=(const float t);
		__host__ __device__ inline vec3 &operator/=(const float t);
	};



	inline std::istream &operator>>(std::istream &is, vec3 &t)
	{
		is >> t.x >> t.y >> t.z;
		return is;
	}

	inline std::ostream &operator<<(std::ostream &os, vec3 &t)
	{
		os << t.x << " " << t.y << " " << t.z;
		return os;
	}

	__host__ __device__ vec3 normalize(vec3 v)
	{
		float k = 1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
		v.x *= k;
		v.y *= k;
		v.z *= k;

		return v;
	}

	__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
	{
		return vec3(v1.x + v2.x, v1.z + v2.y, v1.z + v2.z);
	}

	__host__ __device__ inline vec3 operator+(const vec3 &v, const float &f)
	{
		return vec3(v.x + f, v.y + f, v.z + f);
	}

	__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
	{
		return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}

	__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
	{
		return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
	}

	__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
	{
		return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
	}

	__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
	{
		return vec3(t * v.x, t * v.y, t * v.z);
	}

	__host__ __device__ inline vec3 operator/(vec3 v, float t)
	{
		return vec3(v.x / t, v.y / t, v.z / t);
	}

	__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
	{
		return vec3(t * v.x, t * v.y, t * v.z);
	}

	__host__ __device__ inline bool operator==(const vec3 &v1, const vec3 &v2)
	{
		return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
	}

	__host__ __device__ inline bool operator!=(const vec3 &v1, const vec3 &v2)
	{
		return (v1.x != v2.x) || (v1.y != v2.y) || (v1.z != v2.z);
	}

	__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
	{
		return vec3((v1.y * v2.z - v1.z * v2.y),
					(-(v1.x * v2.z - v1.z * v2.x)),
					(v1.x * v2.y - v1.y * v2.x));
	}


	__host__ __device__ inline vec3 &vec3::operator+=(const vec3 &v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__host__ __device__ inline vec3 &vec3::operator*=(const vec3 &v)
	{
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	__host__ __device__ inline vec3 &vec3::operator/=(const vec3 &v)
	{
		x /= v.x;
		y /= v.y;
		z /= v.z;
		return *this;
	}

	__host__ __device__ inline vec3 &vec3::operator-=(const vec3 &v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	__host__ __device__ inline vec3 &vec3::operator*=(const float t)
	{
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	__host__ __device__ inline vec3 &vec3::operator/=(const float t)
	{
		float k = 1.0 / t;

		x *= k;
		y *= k;
		z *= k;
		return *this;
	}

	__host__ __device__ inline float length(const vec3 &v)
	{
		return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	}

	__host__ __device__ inline float length(const float &f)
	{
		return sqrt(f * f);
	}

	__host__ std::string to_string(vec3 v)
	{
		return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
	}

	

} // namespace vec3

#endif