#include "mesh.h"
#include "iostream"
#include "string"

int main(){

    try{
        std::string meshName = "meshes/homer.obj";
        Mesh mesh = Mesh::load(meshName);
        std::cout << "loaded " << meshName << "\n"
            << "  vertices: " << mesh.num_vertices() << "\n"
            << "  faces:    " << mesh.num_faces() << "\n";
    } catch (const std::exception& e){
        std::cerr << "error: " << e.what() << "\n";
        return 1;

    }
    return 0;
}