#ifdef MESH_H
#define MESH_H


struct Mesh
{
	uint32_t num_triangles;
	std::vector<glm::vec3> vertices; // P
	std::vector<uint32_t> indices;	 // trisIndex
	std::vector<glm::vec3> normals;	 // N
	glm::vector<glm::vec2> uvs;		 // texcoords

	Mesh(uint32_t num_faces, std::vector<glm::vec3> input_vertices, std::vector<glm::vec3> input_normals, std::vector<glm::vec2> texcoords, std::vector<uint32_t> face_indices, std::vector<uint32_t> vertex_indices)
	{
		uint32_t k			   = 0;
		uint32_t highest_index = 0;

		for(uint32_t i = 0; i < num_faces; i++)
		{
			num_triangles += face_indices[i] - 2; // - 2 since 3 verts to a triangle
			for(uint32_t j = 0; j < faces_index[i]; j++)
			{
				if(vertex_indices[k + j] > highest_index)
				{
					highest_index = vertex_indices[k + j];
				}
			}

			k += face_indices[i];
		}

		highest_index += 1;

		// Copy over mesh vertices
		vertices	   = input_vertices
			uint32_t l = 0;
		k			   = 0;

		for(uint32_t i = 0; i < num_faces; i++)
		{
			for(uint32_t j = 0; j < face_indices[i]; j++)
			{
				indices[l]	   = vertex_indices[k];
				indices[l + 1] = vertex_indices[k + j + 1];
				indices[l + 2] = vertex_indices[k + j + 2];

				normals[l]	   = input_normals[k];
				normals[l + 1] = input_normals[k + j + 1];
				normals[l + 2] = input_normals[k + j + 2];

				uvs[l]	   = texcoords[l];
				uvs[l + 1] = texcoords[k + j + 1];
				uvs[l + 2] = texcoords[k + j + 2];
				
				l += 3;
			}

			k += face_indices[i];
		}
	}

	bool intersect(Ray ray, float &near, uint32_t &triangle_index, glm::vec3 uv)
	{
		return false; // todo
	}
};


#endif