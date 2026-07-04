#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <vector>
#include <array>

int main() {
    // One triangle: three vertices, one face. Hardcoded, no file loading.
    std::vector<std::array<double, 3>> vertices = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    std::vector<std::array<size_t, 3>> faces = {
        {0, 1, 2}
    };

    polyscope::init();
    polyscope::registerSurfaceMesh("my triangle", vertices, faces);
    polyscope::show();

    return 0;
}