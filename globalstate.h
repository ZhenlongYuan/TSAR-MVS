#pragma once

#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "linestate.h"
#include "imageinfo.h"
#include "managed.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>


class GlobalState : public Managed {
   public:
    CameraParameters_cu *cameras;
    LineState *lines;
    LineState *cannylines;
    curandState *cs;
    AlgorithmParameters *params;

    int col;  // 640
    int row;  // 480

    //如果改用数组传递,可以考虑第一个值记录总数

    cudaTextureObject_t imgs[MAX_IMAGES];
    cudaArray *cuArray[MAX_IMAGES];
    // cudaTextureObject_t gradx [MAX_IMAGES];
    // cudaTextureObject_t grady [MAX_IMAGES];
    GlobalState() {
        // printf("GlobalState constructor\n");
        cameras = new CameraParameters_cu;
        lines = new LineState;
        cannylines = new LineState;
    }
    ~GlobalState() {
        // printf("GlobalState destructor\n");
        delete cameras;
        delete lines;
        delete cannylines;
    }
};
