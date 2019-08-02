// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <complex>
#include <iostream>
#include <array>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;
GLFWwindow* window2;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>


#include <vector>
#include <deque>
#include <iostream>
#include <random>
#include <tuple>
#include <complex>
#include "Eigen/Dense"
#include <boost/circular_buffer.hpp>

using namespace std;
using namespace Eigen;

typedef std::mt19937 MyRNG;
const int RECS = 4;
const int MAXD = 100;
const float scalee = 100.0f;
int main( void )
{
	// Initialise GLFW
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int windowWidth = 512;
    int windowHeight = 512;
	// Open a window and create its OpenGL context
	window = glfwCreateWindow( windowWidth, windowWidth, "Tutorial 14 - Render To Texture", NULL, NULL);
	window2 = glfwCreateWindow( windowWidth, windowWidth, "waveforms", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		return -1;
	}
	if( window2 == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	//check we got what we wanted
    glfwGetFramebufferSize(window, &windowWidth, &windowHeight);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) 
	{
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited mouvement
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, windowWidth/2, windowWidth/2);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS); 

	// Cull triangles which normal is not towards the camera
	// glEnable(GL_CULL_FACE);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

#if 0
	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders( "StandardShadingRTT.vertexshader", "StandardShadingRTT.fragmentshader" );

	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
	GLuint ViewMatrixID = glGetUniformLocation(programID, "V");
	GLuint ModelMatrixID = glGetUniformLocation(programID, "M");

	// Load the texture
	GLuint Texture = loadDDS("uvmap.DDS");
	
	// Get a handle for our "myTextureSampler" uniform
	GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");

	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;
	bool res = loadOBJ("suzanne.obj", vertices, uvs, normals);

	std::vector<unsigned short> indices;
	std::vector<glm::vec3> indexed_vertices;
	std::vector<glm::vec2> indexed_uvs;
	std::vector<glm::vec3> indexed_normals;
	indexVBO(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);

	// Load it into a VBO
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_vertices.size() * sizeof(glm::vec3), &indexed_vertices[0], GL_STATIC_DRAW);

	GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_uvs.size() * sizeof(glm::vec2), &indexed_uvs[0], GL_STATIC_DRAW);

	GLuint normalbuffer;
	glGenBuffers(1, &normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_normals.size() * sizeof(glm::vec3), &indexed_normals[0], GL_STATIC_DRAW);

	// Generate a buffer for the indices as well
	GLuint elementbuffer;
	glGenBuffers(1, &elementbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);

	// Get a handle for our "LightPosition" uniform
	glUseProgram(programID);
	GLuint LightID = glGetUniformLocation(programID, "LightPosition_worldspace");
#endif


	// ---------------------------------------------
	// Render to Texture - specific code begins here
	// ---------------------------------------------

	// The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
	GLuint FramebufferName = 0;
	glGenFramebuffers(1, &FramebufferName);
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

	// The texture we're going to render to
	GLuint renderedTexture;
	glGenTextures(1, &renderedTexture);
	
	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, renderedTexture);

	// Give an empty image to OpenGL ( the last "0" means "empty" )
	glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, windowWidth, windowHeight, 0,GL_RGB, GL_UNSIGNED_BYTE, 0);

	// Poor filtering
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// The depth buffer
	GLuint depthrenderbuffer;
	glGenRenderbuffers(1, &depthrenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, windowWidth, windowHeight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

	// Set "renderedTexture" as our colour attachement #0
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

	// Set the list of draw buffers.
	GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

	// Always check that our framebuffer is ok
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		return false;

	
	// The fullscreen quad's FBO
	static const GLfloat g_quad_vertex_buffer_data[] = { 
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 1.0f,  1.0f, 0.0f,
	};

	GLuint quad_vertexbuffer;
	glGenBuffers(1, &quad_vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), 	g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	// Create and compile our GLSL program from the shaders
	GLuint quad_programID = LoadShaders( "Shaders/Passthrough.vertexshader", "Shaders/WobblyTexture.fragmentshader" );
	GLuint texID = glGetUniformLocation(quad_programID, "renderedTexture");
	GLuint timeID = glGetUniformLocation(quad_programID, "time");
    
    // Create and compile our GLSL program from the shaders
	GLuint my_programID = LoadShaders( "Shaders/myVertex.vertexshader", "Shaders/myFragment.fragmentshader" );
	GLuint coordsID = glGetUniformLocation(my_programID, "vertices1");
	GLuint colorsID = glGetUniformLocation(my_programID, "colors");
    








	deque<float> receivers[RECS];

	Matrix<float, 1, 2>	sourceVec;
	Matrix<float, RECS, 2>	recPosMat;
	
	sourceVec <<	180,	40;

	recPosMat <<	0,	 50,
					0,	-50,
					0,	-125,
					0,	0;
	
	Matrix<float, RECS, 2>	displacementMat;
	displacementMat = recPosMat.rowwise() - sourceVec;

	cout << sourceVec << endl << endl;
	cout << recPosMat << endl << endl;
	// cout << displacementMat << endl << endl;
	// cout << displacementMat.transpose() << endl << endl;
	
	auto distances = displacementMat.rowwise().norm();
	// cout << distances << endl << endl;

	vector<boost::circular_buffer<float>> cb_data(RECS);

	for (auto& cb : cb_data)
	{
		cb.resize(MAXD);		
	}

	MyRNG rng;
	normal_distribution<float> normal_dist(0, 10);
	normal_distribution<float> noise(0, 0.1);
	

	//init delays
	{
		Matrix<int, RECS, RECS> delayMat;

		for (int i = 0; i < RECS; i++)
		{
			delayMat(i,i) = distances(i);
		}
		for (int i = 0; i < RECS; i++)
		for (int j = 0; j < RECS; j++)
		{
			if (i != j)
				delayMat(i,j) = delayMat(i,i) - delayMat(j,j);
		}
		cout  << delayMat << " : delayMat" << endl << endl;

		//create some initial conditions by adding some delay for each receiver
		for (int r = 0; r < RECS; r++)
		{
			auto delay = delayMat(r,r);
			for (int i = 0; i < delay; i++)
			{
				auto& rec = receivers[r];
				rec.push_back(0);		
			}
		}
	}

	complex<float> rots[RECS][RECS][MAXD] = {0};
	complex<float> mids[RECS][RECS][MAXD] = {0};

	//init rays and rots
	for (int k = 0; k < MAXD; k++)
	{
		// cout << k << "\t: ";
		for (int i = 0; i < RECS; i++)
		for (int j = 0; j < RECS; j++)
		{
			if (i == j)
			{
				continue;
			}

			complex<float> I(0,1);
			
			complex<float> A(recPosMat(i,0), recPosMat(i,1));
			complex<float> B(recPosMat(j,0), recPosMat(j,1));
			
			auto mid = (A + B) / 2.0f;
			auto vec = (A - B);
			
			auto real2 = norm(vec) - (k) * (k);
			// auto real2 = norm(vec) - (k/scalee) * (k/scalee);
			
			auto baseline = sqrt(norm(vec));
			auto dir = vec / baseline * I;

			if (real2 <= 0)
			{
				cout << "\t";
				continue;
			}
			complex<float> rot(sqrtf(real2), k);
			rot /= sqrt(norm(rot));

			auto ray = dir * rot;
			ray += mid;
			if (real(ray) < 0)
			{
				rot = conj(rot);
				ray = {-real(ray), imag(ray)};
			}

			rots[i][j][k] = rot;
			mids[i][j][k] = mid / scalee;
		}
	}





		//calculate beams

		std::complex<float> base{0.5,0};
		std::complex<float> span{0,0.006};
		std::complex<float> scale{15,0};
	    std::array<std::complex<float>,6> beam
	    {
	        base,
	        base + span,
	        (base + span) * scale,
	        base * scale,
	        (base - span) * scale,
	        (base - span)
	    };

	    float values[beam.size() * 2];
	    
static float beamIntensity = 0.1;


	    float colors[] =
	    {
	        1, 1, 1, beamIntensity,
	        1, 1, 1, 0.0,
	        1, 1, 1, 0.0,
	        1, 1, 1, beamIntensity,
	        1, 1, 1, 0.0,
	        1, 1, 1, 0.0
	    };





	//generate vertex buffers
	unsigned int beamVertBuffer;
	glGenBuffers(1, &beamVertBuffer);
	unsigned int beamColorBuffer;
	glGenBuffers(1, &beamColorBuffer);
	unsigned int quadColorBuffer;
	glGenBuffers(1, &quadColorBuffer);
GLuint vbo;

glGenBuffers(1, &vbo);

	float nowData = 0;

	vector<float> data;
	for (int i = 0; i < 512; i++)
	{
		data.emplace_back(0);
	}

	// Check if the ESC key was pressed or the window was closed
	while	( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS 
			&&glfwWindowShouldClose(window) == 0 )
	{



		//get a new sample, and add to propagation delay queues
		{
			// nowData += (normal_dist(rng) - nowData) * 0.5;
			nowData = sin(glfwGetTime() * 10) * 10+sin(glfwGetTime() * sin(glfwGetTime())) * 10;
			data.erase(data.begin());
			data.push_back(nowData);
			data.erase(data.begin());
			data.push_back(nowData);
			
			// cout << nowData << endl;
			for (auto& rec : receivers)
			{
				rec.push_back(nowData + noise(rng));
			}	



		
		}

		//get a new sample at each receiver, and process the recent history
		{
			for (int r = 0; r < RECS; r++)
			{
				auto& rec = receivers[r];
				auto a = rec.front();
				rec.pop_front();
				cb_data[r].push_front(a);
			}
		}

	

		{

			glfwMakeContextCurrent(window2);
			glClearColor(1.0f, 0.0f, 0.0f, 1.0f);		
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);    	

			float points[512*2];
			for (int i = 0; i < 512; i++)
{
	points[i*2] = i/256.0-1;
	points[i*2 + 1] = data[i];
}
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_DYNAMIC_DRAW);

		// glEnableVertexAttribArray(attribute_coord2d);
		glVertexAttribPointer(
		  0,   // attribute
		  2,                   // number of elements per vertex, here (x,y)
		  GL_FLOAT,            // the type of each element
		  GL_FALSE,            // take our values as-is
		  0,                   // no space between values
		  0                    // use the vertex buffer object
		);

		glDrawArrays(GL_LINE_STRIP, 0, 512);


		}
		glfwMakeContextCurrent(window);
		glEnable(GL_BLEND);
	    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		// Render to our framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		glViewport(0,0,windowWidth,windowHeight); // Render on the whole framebuffer, complete from the lower left corner to the upper right

#if 0
		if (0)
		{

			// Clear the screen
			// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glClear(GL_DEPTH_BUFFER_BIT);	

			// Use our shader
			glUseProgram(programID);

			// Compute the MVP matrix from keyboard and mouse input
			computeMatricesFromInputs();
			glm::mat4 ProjectionMatrix = getProjectionMatrix();
			glm::mat4 ViewMatrix = getViewMatrix();
			glm::mat4 ModelMatrix = glm::mat4(1.0);
			glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

			// Send our transformation to the currently bound shader, 
			// in the "MVP" uniform
			glUniformMatrix4fv(MatrixID, 		1, GL_FALSE, &MVP[0][0]);
			glUniformMatrix4fv(ModelMatrixID, 	1, GL_FALSE, &ModelMatrix[0][0]);
			glUniformMatrix4fv(ViewMatrixID, 	1, GL_FALSE, &ViewMatrix[0][0]);

			glm::vec3 lightPos = glm::vec3(4,4,4);
			glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

			// Bind our texture in Texture Unit 0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, Texture);

			// Set our "myTextureSampler" sampler to use Texture Unit 0
			glUniform1i(TextureID, 0);

			// 1rst attribute buffer : vertices
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
			glVertexAttribPointer(
				0,                  // attribute
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized?
				0,                  // stride
				(void*)0            // array buffer offset
			);

			// 2nd attribute buffer : UVs
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
			glVertexAttribPointer(
				1,                                // attribute
				2,                                // size
				GL_FLOAT,                         // type
				GL_FALSE,                         // normalized?
				0,                                // stride
				(void*)0                          // array buffer offset
			);

			// 3rd attribute buffer : normals
			glEnableVertexAttribArray(2);
			glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
			glVertexAttribPointer(
				2,                                // attribute
				3,                                // size
				GL_FLOAT,                         // type
				GL_FALSE,                         // normalized?
				0,                                // stride
				(void*)0                          // array buffer offset
			);

			// Index buffer
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

			// Draw the triangles !
			glDrawElements(
				GL_TRIANGLES,      // mode
				indices.size(),    // count
				GL_UNSIGNED_SHORT, // type
				(void*)0           // element array buffer offset
			);

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(2);


		}


#endif

		{
			glClear(GL_DEPTH_BUFFER_BIT);	
    		glEnable(GL_BLEND);
    		
    		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

			glUseProgram(my_programID);

			GLuint loc_tID = glGetUniformLocation(my_programID, "t");

			glUniform1f(loc_tID, 0 );
	
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);    	

			{
				glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
				glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), 	g_quad_vertex_buffer_data,  GL_STATIC_DRAW);
				glVertexAttribPointer(
					0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
					3,                  // size
					GL_FLOAT,           // type
					GL_FALSE,           // normalized?
					0,                  // stride
					(void*)0            // array buffer offset
				);

			    float quadColors[] =
			    {
			        0, 0, 0, .05121501105,
			        0, 0, 0, .05121501105,
			        0, 0, 0, .051211501105,
			        0, 0, 0, .05121501105,
			        0, 0, 0, .05121501105,
			        0, 0, 0, .0512151105
			    };

				glBindBuffer(GL_ARRAY_BUFFER, quadColorBuffer);
				glBufferData(GL_ARRAY_BUFFER, sizeof(quadColors), 	quadColors,  GL_STATIC_DRAW);
				glVertexAttribPointer(
					1,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
					4,                  // size
					GL_FLOAT,           // type
					GL_FALSE,           // normalized?
					0,                  // stride
					(void*)0            // array buffer offset
				);

				// Draw the triangles !
				glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles
			}


			glBlendFunc(GL_SRC_ALPHA, GL_ONE);
			glClear(GL_DEPTH_BUFFER_BIT);	

			
			for (int i = 0; i < RECS; i++)
			for (int j = 0; j < RECS; j++)
			{
				if (i == j)
				{
					continue;
				}

				auto a = cb_data[i].front();
				for (int k = 0; k < MAXD; k++)
				{
					auto rot = rots[i][j][k];
					auto mid = mids[i][j][k];

					if (rot == 0.0f)
						continue;
				
					auto b = cb_data[j][k];
					auto delta = abs(a - b);
					if (delta == 0)
					{
						continue;	//todo aaron check thsi
						// delta += 0.001;
					}

					if (delta < 1)
					{
						// cout << "d" ;
						delta += 0.001;

					    for (int i = 0; i < beam.size(); i++)
					    {
					        values[2* i + 0] = real(beam[i] * rot + mid-1.0f);
					        values[2* i + 1] = imag(beam[i] * rot + mid);
					    }

					    float ssss = 1/(2.0+ delta*delta*delta*delta);
			        	float var = 10-delta;
			        	colors[3] =  ssss*ssss*ssss*ssss*ssss * beamIntensity;
			        	colors[15] =  colors[3];
			        	

						glClear(GL_DEPTH_BUFFER_BIT);	
			        	
						glBindBuffer(GL_ARRAY_BUFFER, beamVertBuffer);
						glBufferData(GL_ARRAY_BUFFER, sizeof(values), 	values,  GL_STREAM_DRAW);
						glVertexAttribPointer(
							0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
							2,                  // size
							GL_FLOAT,           // type
							GL_FALSE,           // normalized?
							0,                  // stride
							(void*)0            // array buffer offset
						);

						glBindBuffer(GL_ARRAY_BUFFER, beamColorBuffer);
						glBufferData(GL_ARRAY_BUFFER, sizeof(colors), 	colors,  GL_STREAM_DRAW);
						glVertexAttribPointer(
							1,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
							4,                  // size
							GL_FLOAT,           // type
							GL_FALSE,           // normalized?
							0,                  // stride
							(void*)0            // array buffer offset
						);

						// Draw the triangles !
						glDrawArrays(GL_TRIANGLE_FAN, 0, sizeof(values)/sizeof(values[0])/2); // 2*3 indices starting at 0 -> 2 triangles
				
					}
				}
			}

			glDisableVertexAttribArray(0);
		}




    	// glEnable(GL_BLEND);
		glDisable(GL_BLEND);
    	glBlendFunc(GL_SRC_ALPHA, GL_ZERO);

		// Render to the screen
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // Render on the whole framebuffer, complete from the lower left corner to the upper right
		glViewport(0,0,windowWidth,windowHeight);

		// Clear the screen
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use our shader
		glUseProgram(quad_programID);

		// Bind our texture in Texture Unit 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, renderedTexture);
		// Set our "renderedTexture" sampler to use Texture Unit 0
		if (1)
		{
			GLfloat pixels[windowWidth*windowWidth];

			glGetTexImage(GL_TEXTURE_2D, 0, GL_GREEN, GL_FLOAT, pixels);

			int maxX;
			int maxY;
			float maxVal = 0;
			for (size_t y = 0; y < windowWidth; y +=1)
			for (size_t x = 0; x < windowWidth; x ++)
			{

				float val = pixels[x + y * windowWidth];
				if (val > maxVal)
				{
					maxVal = val;
					maxX = x;
					maxY = y;
				}
			}
	
			if (maxVal > 0.95)
			{
				beamIntensity *= 0.99;
				cout << "beam" << beamIntensity;
			}
			if (maxVal < 0.6)
			{
				beamIntensity *= 1.01;
			}
			if (maxVal > 0.6)
			{
				int percentX = (maxX) * scalee/windowWidth;
				int percentY = (maxY) * scalee/windowWidth;
				int x = 			2*scalee/100 * percentX;
				int y = -scalee +		2*scalee/100 * percentY;
				std::cout << "max val : " << maxVal << "\t " <<  x<< "\t" << y << endl;
				
			}
			glTexSubImage2D(GL_TEXTURE_2D, 0,0,0,windowWidth, windowWidth, GL_GREEN, GL_FLOAT, pixels);



			glUniform1i(texID, 0);

			// glUniform1f(timeID, (float)(glfwGetTime()*10.0f) );
			glUniform1f(timeID, 0 );

			// 1rst attribute buffer : vertices
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
			glVertexAttribPointer(
				0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized?
				0,                  // stride
				(void*)0            // array buffer offset
			);

			// Draw the triangles !
			glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles

			glDisableVertexAttribArray(0);

		}
		// Swap buffers
		glfwSwapBuffers(window);
		glfwSwapBuffers(window2);
		glfwPollEvents();

	}
#if 0
	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteBuffers(1, &normalbuffer);
	glDeleteBuffers(1, &elementbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &Texture);

	glDeleteFramebuffers(1, &FramebufferName);
	glDeleteTextures(1, &renderedTexture);
	glDeleteRenderbuffers(1, &depthrenderbuffer);
	glDeleteBuffers(1, &quad_vertexbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);
#endif

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

