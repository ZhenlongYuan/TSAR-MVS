//#include <helper_math.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdint.h> // for uint8_t
#include "globalstate.h"
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "linestate.h"
#include "imageinfo.h"
#include "config.h"

#include <vector_types.h> // float4
#include <math.h>
#include <cuda.h>
#include <vector>
#include <string>
#include <iostream>
#include <curand_kernel.h>
#include "vector_operations.h"
#include "helper_cuda.h"
//#include <ORUtils/Matrix.h>


#include<cmath> 
#include<iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp> 
using namespace cv;


//#define CENSUS
#define SHARED
//#define NOTEXTURE_CHECK
#define WIN_INCREMENT 2
#define DEPTH_THS_MAX 0
#define DEPTH_THS_MIN 0

__device__ float K[16];
__device__ float K_inv[16];

__device__ FORCEINLINE_GIPUMA float disparityDepthConversion_cu ( const float &f, const float &baseline, const float &d ) {
    return f * baseline / d;
}

__device__ FORCEINLINE_GIPUMA float inbound(int neindex, int cols, int rows) {
    int px = neindex % cols;
    int py = neindex / cols;
    if (px >= 0 && px < cols && py >= 0 && py < rows)
        return true;
    else
        return false;
}

__device__ FORCEINLINE_GIPUMA void get3Dpoint_cu1(float4 *__restrict__ ptX, const Camera_cu &cam, const int2 &p) {
    // in case camera matrix is not normalized: see page 162, then depth might
    // not be the real depth but w and depth needs to be computed from that
    // first
    float4 pt;
    pt.x = (float)p.x - cam.P_col34.x;
    pt.y = (float)p.y - cam.P_col34.y;
    pt.z = 1.0f - cam.P_col34.z;

    matvecmul4(cam.M_inv, pt, ptX);
}


// CHECKED
__device__ FORCEINLINE_GIPUMA static float getD_cu ( const float4 &normal,
                                              const int2 &p,
                                              const float &depth,
                                              const Camera_cu &cam ) {
    /*float4 pt;*/
    /*get3Dpoint_cu ( &pt, cam, (float)x0, (float)y0, depth );*/
    float4 pt,ptX;
    pt.x = depth * (float)(p.x)     - cam.P_col34.x;
    pt.y = depth * (float)(p.y)     - cam.P_col34.y;
    pt.z = depth         - cam.P_col34.z;

    matvecmul4 (cam.M_inv, pt, (&ptX));

    return -(dot4(normal,ptX));
    /*return getPlaneDistance_cu (normal, ptX);*/
}
// CHECKED
__device__ FORCEINLINE_GIPUMA void normalize_cu (float4 * __restrict__ v)
{
    const float normSquared = pow2(v->x) + pow2(v->y) + pow2(v->z);
    const float inverse_sqrt = rsqrtf (normSquared);
    v->x *= inverse_sqrt;
    v->y *= inverse_sqrt;
    v->z *= inverse_sqrt;
}
//CHECKED
__device__ FORCEINLINE_GIPUMA void getViewVector_cu (float4 * __restrict__ v, const Camera_cu &camera, const int2 &p)
{
    get3Dpoint_cu1 (v, camera, p);
    sub((*v), camera.C4);
    normalize_cu(v);
    //v->x=0;
    //v->y=0;
    //v->z=1;
}
__device__ FORCEINLINE_GIPUMA static void vecOnHemisphere_cu ( float4 * __restrict__ v, const float4 &viewVector ) {
    const float dp = dot4 ( (*v), viewVector );
    if ( dp > 0.0f ) {
        negate4(v);
    }
    return;
}
__device__ FORCEINLINE_GIPUMA float curand_between (curandState *cs, const float &min, const float &max)
{
    return (curand_uniform(cs) * (max-min) + min);
}

__device__ FORCEINLINE_GIPUMA static void rndUnitVectorSphereMarsaglia_cu (float4 *v, curandState *cs) {
    float x = 1.0f;
    float y = 1.0f;
    float sum = 2.0f;
    while ( sum>=1.0f ) {
        x = curand_between (cs, -1.0f, 1.0f);
        y = curand_between (cs, -1.0f, 1.0f);
        sum = get_pow2_norm(x,y);
    }
    const float sq = sqrtf ( 1.0f-sum );
    v->x = 2.0f*x*sq;
    v->y = 2.0f*y*sq;
    v->z = 1.0f-2.0f*sum;

}

__device__ FORCEINLINE_GIPUMA static void rndUnitVectorOnHemisphere_cu ( float4 *v, const float4 &viewVector, curandState *cs ) {
    rndUnitVectorSphereMarsaglia_cu (v, cs);
    vecOnHemisphere_cu ( v,viewVector );
};

__device__ FORCEINLINE_GIPUMA float l1_norm(float f) {
    return fabsf(f);
}
__device__ FORCEINLINE_GIPUMA float l1_norm(float4 f) {
    return ( fabsf (f.x) +
             fabsf (f.y) +
             fabsf (f.z))*0.3333333f;

}
__device__ FORCEINLINE_GIPUMA float l1_norm2(float4 f) {
    return ( fabsf (f.x) +
             fabsf (f.y) +
             fabsf (f.z));

}
template< typename T >
__device__ FORCEINLINE_GIPUMA float weight_cu ( const T &c1, const T &c2, const float &gamma )
{
    const float colorDis = l1_norm (  c1 -  c2 );
    return expf ( -colorDis / gamma ); ///[>0.33333333f));
}

__device__ FORCEINLINE_GIPUMA void getCorrespondingPoint_cu ( const int2 &p, const float * __restrict__ H, float4 * __restrict__ ptf ) {
    /*getCorrespondingHomographyPt_cu ( (const float * )H, x , y , pt );*/
    float4 pt;
    pt.x = __int2float_rn (p.x);
    pt.y = __int2float_rn (p.y);
    /*pt.z = 1.0f;*/
    matvecmul4noz(H,pt,ptf); //ptf =  H * pt;
    vecdiv4(ptf,ptf->z); //ptf = ptf / ptf[2];

    return ;
}

template< typename T >
__device__ FORCEINLINE_GIPUMA float pmCostComputation (
                                                const cudaTextureObject_t &l,
                                                const T * __restrict__ tile_left,
                                                const cudaTextureObject_t &r,
                                                const float4 &pt_l,
                                                const float4 &pt_r,
                                                const int &rows,
                                                const int &cols,
                                                const float &tau_color,
                                                const float &tau_gradient,
                                                const float &alpha,
                                                const float &w )
{
    const float colDiff = l1_norm ( tex2D<T>(l,pt_l.x + 0.5f,pt_l.y + 0.5f) - tex2D<T>(r,pt_r.x + 0.5f, pt_r.y + 0.5f) );
    const float colDis = fminf ( colDiff, tau_color );

    const T gx1 = tex2D<T> (l, pt_l.x+1 + 0.5f, pt_l.y   + 0.5f) - tex2D<T> (l, pt_l.x-1 + 0.5f, pt_l.y   + 0.5f);
    const T gy1 = tex2D<T> (l, pt_l.x   + 0.5f, pt_l.y+1 + 0.5f) - tex2D<T> (l, pt_l.x   + 0.5f, pt_l.y-1 + 0.5f);
    const T gx2 = tex2D<T> (r, pt_r.x+1 + 0.5f, pt_r.y   + 0.5f) - tex2D<T> (r, pt_r.x-1 + 0.5f, pt_r.y   + 0.5f);
    const T gy2 = tex2D<T> (r, pt_r.x   + 0.5f, pt_r.y+1 + 0.5f) - tex2D<T> (r, pt_r.x   + 0.5f, pt_r.y-1 + 0.5f);

    const T gradX = (gx1 - gx2);
    const T gradY = (gy1 - gy2);

    //gradient dissimilarity (L1) in x and y direction (multiplication by 0.5 to use tauGrad from PatchMatch stereo paper)
    const float gradDis = fminf ( ( l1_norm ( gradX ) + l1_norm ( gradY ) ) * 0.0625f, tau_gradient );
    //gradient dissimilarity only in x direction
    //float gradDis = min(abs(gradX),tau_gradient);

    const float dis = ( 1.f - alpha ) * colDis + alpha * gradDis;
    return w * dis;
}

__device__ FORCEINLINE_GIPUMA void getHomography_cu ( const Camera_cu &from, const Camera_cu &to,
                                               const float * __restrict__ K1_inv, const float * __restrict__ K2,
                                               const float4 &n, const float &d, float * __restrict__ H )
{
    //if ( !to.reference )
    {
        /*getHomography_real( K1_inv, K2, to.R, to.t4, n, d, H );*/
        /*float tmp[16];*/
        float tmp2[16];
        outer_product4(to.t4, n, H); // tmp = t * n'
        matdivide(H, d); // tmp / d
        matmatsub2(to.R, H); // tmp = R - tmp;
        matmul_cu(H,K1_inv,tmp2); // tmp2=tmp*Kinv
        matmul_cu(K2,tmp2,H);// H = tmp * K2

    }
    return;
}

/*
 * cost computation of different cost functions
 */
template< typename T >
__device__ FORCEINLINE_GIPUMA static float pmCost (
                                            const cudaTextureObject_t &l,
                                            const cudaTextureObject_t &r,
                                            const int &x,
                                            const int &y,
                                            const float4 &normal,
                                            const int &vRad,
                                            const int &hRad,
                                            const AlgorithmParameters &algParam,
                                            const CameraParameters_cu &camParams,
                                            const int &camTo )
{
    float4 pt_c;
    float H[9];
    getHomography_cu ( camParams.cameras[REFERENCE], camParams.cameras[camTo], camParams.cameras[REFERENCE].K_inv, camParams.cameras[camTo].K, normal, normal.w, H );
    getCorrespondingPoint_cu ( make_int2(x, y), H, &pt_c );
    
    float cen_pix = tex2D<float>(l, x + 0.5f, y + 0.5f);
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    float cost_max = 2.0f;

    float sum_ref = 0.0f;
    float sum_ref_ref = 0.0f;
    float sum_src = 0.0f;
    float sum_src_src = 0.0f;
    float sum_ref_src = 0.0f;
    float bilateral_weight_sum = 0.0f;
    float cost = 0;
    for ( int i = -hRad; i < hRad + 1; i+=WIN_INCREMENT ) {
        for ( int j = -vRad; j < vRad + 1; j+=WIN_INCREMENT ) {
            const int2 pl = make_int2(x + i, y + j);
            float ref_pix = tex2D<float>(l, pl.x + 0.5f, pl.y + 0.5f);
            float4 pr;
            getCorrespondingPoint_cu(pl, H, &pr);
            float src_pix = tex2D<float>(r, pr.x + 0.5f, pr.y + 0.5f);
            float spatial_dist = sqrtf(i * i + j * j);
            float color_dist = fabs(ref_pix - cen_pix);
            float weight =  exp(-spatial_dist / (2.0f * sigma_spatial* sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));

            sum_ref += weight * ref_pix;
            sum_ref_ref += weight * ref_pix * ref_pix;
            sum_src += weight * src_pix;
            sum_src_src += weight * src_pix * src_pix;
            sum_ref_src += weight * ref_pix * src_pix;
            bilateral_weight_sum += weight;
        }
    }

    const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
    sum_ref *= inv_bilateral_weight_sum;
    sum_ref_ref *= inv_bilateral_weight_sum;
    sum_src *= inv_bilateral_weight_sum;
    sum_src_src *= inv_bilateral_weight_sum;
    sum_ref_src *= inv_bilateral_weight_sum;

    const float var_ref = sum_ref_ref - sum_ref * sum_ref;
    const float var_src = sum_src_src - sum_src * sum_src;

    const float kMinVar = 1e-5f;
    if (var_ref < kMinVar || var_src < kMinVar) {
        return cost = cost_max;
    } else {
        const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
        const float var_ref_src = sqrtf(var_ref * var_src);
        return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
    }

}

template< typename T >
__device__ FORCEINLINE_GIPUMA static float rlCost(
                                            const cudaTextureObject_t &l,
                                            const cudaTextureObject_t &r,
                                            const int &x,
                                            const int &y,
                                            const float4 &normal,
                                            const int &vRad,
                                            const int &hRad,
                                            const AlgorithmParameters &algParam,
                                            const CameraParameters_cu &camParams,
                                            const int &camTo )
{
    float4 pt_c;
    float H[9];
    getHomography_cu ( camParams.cameras[REFERENCE], camParams.cameras[camTo], camParams.cameras[REFERENCE].K_inv, camParams.cameras[camTo].K, normal, normal.w, H );
    float det = H[0] * H[4] * H[8] + H[1] * H[5] * H[6] + H[2] * H[3] * H[7] -
                H[2] * H[4] * H[6] - H[1] * H[3] * H[8] - H[0] * H[5] * H[7];
    float V[9];
    V[0] = H[4] * H[8] - H[5] * H[7];
    V[1] = H[1] * H[8] - H[2] * H[7];
    V[2] = H[1] * H[5] - H[2] * H[4];
    V[3] = H[3] * H[8] - H[5] * H[6];
    V[4] = H[0] * H[8] - H[2] * H[6];
    V[5] = H[0] * H[5] - H[2] * H[3];
    V[6] = H[3] * H[7] - H[4] * H[6];
    V[7] = H[0] * H[7] - H[1] * H[6];
    V[8] = H[0] * H[4] - H[1] * H[3];

    V[0] = V[0] / det;
    V[1] = -V[1] / det;
    V[2] = V[2] / det;
    V[3] = -V[3] / det;
    V[4] = V[4] / det;
    V[5] = -V[5] / det;
    V[6] = V[6] / det;
    V[7] = -V[7] / det;
    V[8] = V[8] / det;

    getCorrespondingPoint_cu ( make_int2(x, y), H, &pt_c );
    
    float cen_pix = tex2D<float>(r, pt_c.x + 0.5f, pt_c.y + 0.5f);
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    float cost_max = 2.0f;

    float sum_ref = 0.0f;
    float sum_ref_ref = 0.0f;
    float sum_src = 0.0f;
    float sum_src_src = 0.0f;
    float sum_ref_src = 0.0f;
    float bilateral_weight_sum = 0.0f;
    float cost = 0;
    for ( int i = -hRad; i < hRad + 1; i+=WIN_INCREMENT ) {
        for ( int j = -vRad; j < vRad + 1; j+=WIN_INCREMENT ) {
            const int2 pl = make_int2(pt_c.x + i, pt_c.y + j);
            float ref_pix = tex2D<float>(r, pl.x + 0.5f, pl.y + 0.5f);
            float4 pr;
            getCorrespondingPoint_cu(pl, V, &pr);
            float src_pix = tex2D<float>(l, pr.x + 0.5f, pr.y + 0.5f);
            float spatial_dist = sqrtf(i * i + j * j);
            float color_dist = fabs(ref_pix - cen_pix);
            float weight =  exp(-spatial_dist / (2.0f * sigma_spatial* sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));

            sum_ref += weight * ref_pix;
            sum_ref_ref += weight * ref_pix * ref_pix;
            sum_src += weight * src_pix;
            sum_src_src += weight * src_pix * src_pix;
            sum_ref_src += weight * ref_pix * src_pix;
            bilateral_weight_sum += weight;
        }
    }

    const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
    sum_ref *= inv_bilateral_weight_sum;
    sum_ref_ref *= inv_bilateral_weight_sum;
    sum_src *= inv_bilateral_weight_sum;
    sum_src_src *= inv_bilateral_weight_sum;
    sum_ref_src *= inv_bilateral_weight_sum;

    const float var_ref = sum_ref_ref - sum_ref * sum_ref;
    const float var_src = sum_src_src - sum_src * sum_src;

    const float kMinVar = 1e-5f;
    if (var_ref < kMinVar || var_src < kMinVar) {
        return cost = cost_max;
    } else {
        const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
        const float var_ref_src = sqrtf(var_ref * var_src);
        return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
    }

}


template< typename T >
__device__ FORCEINLINE_GIPUMA static float hasImageTexture (
                                                   const cudaTextureObject_t &l,
                                                   const int2 &p,
                                                   const int &vRad,
                                                   const int &hRad,
                                                   const AlgorithmParameters &algParam)
{
    const float gamma = algParam.gamma;

    int count_similar_pixel = 0;
    for ( int i = -hRad; i < hRad + 1; i += WIN_INCREMENT ) {
        for ( int j = -vRad; j < vRad + 1; j += WIN_INCREMENT ) {
            const int xTemp = p.x + i;
            const int yTemp = p.y + j;
            float4 pt_l;
            pt_l.x = __int2float_rn( xTemp );
            pt_l.y = __int2float_rn( yTemp );

            const float w = weight_cu<T> ( tex2D <T> (l, pt_l.x + 0.5f, pt_l.y + 0.5f ), tex2D <T> ( l, p.x + 0.5f, p.y + 0.5f ), gamma);
                if (w > algParam.no_texture_sim)
                    count_similar_pixel++;
        }
    }
    if (count_similar_pixel > hRad*vRad*4/(WIN_INCREMENT * WIN_INCREMENT)*algParam.no_texture_per)
        return false;
    return true;
}


static __device__ FORCEINLINE_GIPUMA void sort_small(float * __restrict__ d,const int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
            d[j] = d[j-1];
        d[j] = tmp;
    }
}

__device__ FORCEINLINE_GIPUMA float getDepthFromPlane3_cu (const Camera_cu &cam,
                                                    const float4 &n,
                                                    const float &d,
                                                    const int2 &p)
{
    return -d*cam.fx/((n.x*(p.x-cam.K[2]))+(n.y*(p.y-cam.K[2+3]))*cam.alpha +n.z*cam.fx);
}

__device__ FORCEINLINE_GIPUMA float getDisparity_cu ( const float4 &normal,
                                               const float &d,
                                               const int2 &p,
                                               const Camera_cu &cam )
{
    if ( d != d )
        return 1000;

    return getDepthFromPlane3_cu (cam, normal, d, p);
}

template< typename T >
__device__ FORCEINLINE_GIPUMA static float pmCostMultiview_cu (int &beview, float &ratio, 
                                                        const cudaTextureObject_t *images,
                                                        const int2 p,
                                                        const float4 &normal,
                                                        const int &vRad,
                                                        const int &hRad,
                                                        const AlgorithmParameters &algParam,
                                                        const CameraParameters_cu &camParams,
                                                        const float4 * __restrict__ state,
                                                        const int point)
{
    float costVector[32];
    float origVector[32];
    int numValidViews = 0;

    int num = camParams.viewSelectionSubsetNumber;
    for (int i = 0; i < num; i++) {
        int idxCurr = camParams.viewSelectionSubset[i];
        {
            float c = 0;
            c = pmCost<T> ( images[REFERENCE],
                            images[idxCurr],
                            p.x, p.y,
                            normal,
                            vRad, hRad,
                            algParam, camParams,
                            idxCurr );

            if ( c < MAXCOST )
                numValidViews++;
            else
                c = MAXCOST; 
            costVector[i] = c;
            origVector[i] = c;
        }
    }
    sort_small(costVector, num);

    int numBest = numValidViews;

    if ( algParam.cost_comb == COMB_BEST_N )
        numBest = min ( numBest, algParam.n_best );

    float cost = 0.0f;
    if (numBest > 0) {
        for (int i = 0; i < numBest; i++) {
            cost += costVector[i];
        }
        cost = cost / ((float)numBest);
        ratio = costVector[0] / costVector[1];
        for (int i = 0; i < num; i++) {
            if (costVector[0] == origVector[i]) {
                beview = camParams.viewSelectionSubset[i];
            }
        }
    } else {
        cost = MAXCOST;
        ratio = 0;
        beview = -1;
    }

    return cost;
}


#define ISDISPDEPTHWITHINBORDERS(disp,camParams,camIdx,algParams) \
disp >= camParams.cameras[REFERENCE].depthMin && disp <= camParams.cameras[REFERENCE].depthMax

template< typename T >
__device__ FORCEINLINE_GIPUMA void spatialPropagation_cu (GlobalState &gs, const cudaTextureObject_t *imgs,
                                                    const int2 &p,
                                                    const int &box_hrad, const int &box_vrad,
                                                    const AlgorithmParameters &algParams,
                                                    const CameraParameters_cu &camParams,
                                                    float *cost_now,
                                                    float4 *norm_now,
                                                    const float4 norm_before,
                                                    float *disp_now,
                                                    const float4 * __restrict__ state,
                                                    const int point, bool final
                                                  )
{
    // previous image values
    const float d_before    = norm_before.w;
    const float disp_before = getDisparity_cu (norm_before, d_before, p, camParams.cameras[REFERENCE] );
    float ratio;
    int beview;
    float cost_before = pmCostMultiview_cu<T> ( beview, ratio, imgs,
                                                p,
                                                norm_before,
                                                box_vrad,
                                                box_hrad,
                                                algParams,
                                                camParams,
                                                state,
                                                point);

    if ( ISDISPDEPTHWITHINBORDERS(disp_before,camParams,REFERENCE,algParams) )
    {
        if ( cost_before < *cost_now ) {
            *disp_now   = disp_before;
            *norm_now   = norm_before;
            *cost_now   = cost_before;
            if (!final) {
                gs.lines->ratio[p.y * gs.col + p.x] = ratio;
                gs.lines->beview[p.y * gs.col + p.x] = beview;
            }
        }
    }
    return;
}

/* compute random disparity and unit vector within given intervals, used for plane refinement step
 * interval is limited by image border and general disparity range [0 maxDisparity]
 * Input: x     - current column x
 *        disp  - old disparity value
 *        norm  - old normal
 *        maxDeltaZ  - range radius for disparity [disp-maxDeltaZ,disp+maxDeltaZ]
 *        maxDeltaN  - range radius for normal
 *        maxDisparity  - maximum disparity value
 *        cols - number of columns of the image
 *        dir - disparity to the left or right of x
 *        limit - defines maximal value for |[nx ny]T| so that only plane tilts to a certain degree are possible
 * Output: dispOut - new disparity
 *         normOut - new normal
 */
__device__ FORCEINLINE_GIPUMA void getRndDispAndUnitVector_cu (
                                                        float disp,
                                                        const float4 norm,
                                                        float &dispOut,
                                                        float4 * __restrict__ normOut,
                                                        const float maxDeltaZ,
                                                        const float maxDeltaN,
                                                        const float minDisparity,
                                                        const float maxDisparity,
                                                        curandState *cs,
                                                        CameraParameters_cu &camParams,
                                                        const float baseline,
                                                        const float4 viewVector) {
    //convert depth to disparity and back for non-rectified approach
    disp = disparityDepthConversion_cu ( camParams.f, baseline, disp );

    //delta min limited by disp=0 and image border
    //delta max limited by disp=maxDisparity and image border
    float minDelta, maxDelta;
    minDelta = -min ( maxDeltaZ, minDisparity + disp ); //limit new disp>=0
    maxDelta = min ( maxDeltaZ, maxDisparity - disp ); //limit new disp < maxDisparity

    /*minDelta ; -minDelta;*/

    float deltaZ = curand_between(cs, minDelta, maxDelta);
    //get new disparity value within valid range [0 maxDisparity]
    dispOut = fminf ( fmaxf ( disp + deltaZ, minDisparity ), maxDisparity );

    dispOut = disparityDepthConversion_cu ( camParams.f, baseline, dispOut );

    //get normal
    normOut->x = norm.x + curand_between (cs, -maxDeltaN, maxDeltaN );
    normOut->y = norm.y + curand_between (cs, -maxDeltaN, maxDeltaN );
    normOut->z = norm.z + curand_between (cs, -maxDeltaN, maxDeltaN );

    normalize_cu ( normOut );
    vecOnHemisphere_cu (  normOut, viewVector );
}

template< typename T >
__device__ FORCEINLINE_GIPUMA static void planeRefinement_cu (GlobalState &gs, 
                                                       const cudaTextureObject_t *images,
                                                       const int2 &p,
                                                       const int &box_hrad,
                                                       const int &box_vrad,
                                                       const AlgorithmParameters &algParams,
                                                       CameraParameters_cu &camParams,
                                                       const int camIdx,
                                                       float * __restrict__ cost_now,
                                                       float4 * __restrict__ norm_now,
                                                       float * __restrict__ disp_now,
                                                       curandState &cs, const float4 *__restrict__ state, bool final) {
    float deltaN = 1.0f;

    float4 viewVector;
    getViewVector_cu (&viewVector, camParams.cameras[0], p);

    float4 norm_temp;
    float dispTemp_L;
    float costTempL;

    const float maxdisp=algParams.max_disparity / 2.0f; // temp variable
   for ( float deltaZ = maxdisp; deltaZ >= 0.01f; deltaZ = deltaZ / 10.0f ) {
        getRndDispAndUnitVector_cu (
                                    *disp_now, *norm_now,
                                    dispTemp_L, &norm_temp,
                                    deltaZ, deltaN,
                                    algParams.min_disparity, algParams.max_disparity,
                                    &cs,
                                    camParams, camParams.cameras[0].baseline,
                                    viewVector);

        norm_temp.w = getD_cu(norm_temp, p, dispTemp_L, camParams.cameras[camIdx]);
        float ratio;
        int beview;
        costTempL = pmCostMultiview_cu<T>(beview, ratio, images,
                                            p,
                                            norm_temp,
                                            box_vrad, box_hrad,
                                            algParams, camParams,
                                            state,
                                            0);

        if ( costTempL < *cost_now ) {
            *cost_now = costTempL;
            *disp_now = dispTemp_L;
            *norm_now = norm_temp;
            if (!final) {
                gs.lines->ratio[p.y * gs.col + p.x] = ratio;
                gs.lines->beview[p.y * gs.col + p.x] = beview;
            }
        }
        deltaN = deltaN / 4.0f;
    }
}

template< typename T >
__global__ void gipuma_init_cu2(GlobalState &gs) {
    const int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;

    // Temporary variables
    Camera_cu &camera = gs.cameras->cameras[REFERENCE];

    const int center = p.y*cols+p.x;
    int box_hrad = gs.params->box_hsize / 2;
    int box_vrad = gs.params->box_vsize / 2;

    float disp_now;
    float4 norm_now;

    curandState localState = gs.cs[p.y*cols+p.x];
    curand_init ( clock64(), p.y, p.x, &localState );

    // Compute random normal on half hemisphere of fronto view vector
    float mind = gs.params->min_disparity;
    float maxd = gs.params->max_disparity;
    float4 viewVector;
    getViewVector_cu ( &viewVector, camera, p);
    //printf("Random number is %f\n", random_number);
    //return;
    disp_now = curand_between(&localState, mind, maxd);

    rndUnitVectorOnHemisphere_cu ( &norm_now, viewVector, &localState );
    disp_now= disparityDepthConversion_cu ( camera.f, camera.baseline, disp_now);

    // Save values
    norm_now.w = getD_cu ( norm_now, p, disp_now,  camera);

    gs.lines->norm4[center] = norm_now;
    float ratio;
    int beview;
    gs.lines->c[center] = pmCostMultiview_cu<T> (beview, ratio, gs.imgs,
                                                 p,
                                                 norm_now,
                                                 box_vrad, box_hrad,
                                                 *(gs.params),
                                                 *(gs.cameras),
                                                 gs.lines->norm4,
                                                 0);
    return;
}

template <typename T>
__global__ void gipuma_get_disp (GlobalState &gs)
{
    const int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;
    CameraParameters_cu &camParams = *(gs.cameras);
    int pindex = p.y*cols+p.x;

    float4 norm = gs.lines->norm4[pindex];
    float4 norm_transformed;
    // Transform back normal to world coordinate
    matvecmul4 ( gs.cameras->cameras[REFERENCE].R_orig, norm, (&norm_transformed));
    gs.lines->norm4[pindex] = norm_transformed;

    float disp_new = disparityDepthConversion_cu(
        camParams.f, camParams.cameras[0].baseline, gs.lines->depth[pindex]);
    gs.lines->norm4[pindex].w = getD_cu(gs.lines->norm4[pindex], p, disp_new,
                                        gs.cameras->cameras[REFERENCE]);
}

template <typename T>
__global__ void gipuma_compute_disp_final (GlobalState &gs)
{
    const int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;
    CameraParameters_cu &camParams = *(gs.cameras);
    
    int pindex = p.y*cols+p.x;

    float depth_now = getDisparity_cu(gs.lines->norm4[pindex], gs.lines->norm4[pindex].w, p, gs.cameras->cameras[REFERENCE]);
    float disp_now = disparityDepthConversion_cu(camParams.f, camParams.cameras[0].baseline, depth_now);

    float depth_org = getDisparity_cu(gs.lines->resize4[pindex], gs.lines->resize4[pindex].w, p, gs.cameras->cameras[REFERENCE]);
    float disp_org = disparityDepthConversion_cu(camParams.f, camParams.cameras[0].baseline, depth_org);

    if ((abs(disp_now - disp_org) > 6 && gs.lines->text[pindex] == 1) || gs.lines->text[pindex] == -1) {
        gs.lines->norm4[pindex] = gs.lines->resize4[pindex];
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float disp = getDisparity_cu(gs.lines->norm4[pindex], gs.lines->norm4[pindex].w, p, gs.cameras->cameras[REFERENCE]);

    if (disp > camParams.cameras[REFERENCE].depthMax) {
        gs.lines->norm4[pindex].w = getD_cu(gs.lines->norm4[pindex], p,
                                        camParams.cameras[REFERENCE].depthMax,
                                        gs.cameras->cameras[REFERENCE]);
    }
    if (disp < camParams.cameras[REFERENCE].depthMin) {
        gs.lines->norm4[pindex].w = getD_cu(
            gs.lines->norm4[pindex], p, camParams.cameras[REFERENCE].depthMin,
            gs.cameras->cameras[REFERENCE]);
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    gs.lines->depth[pindex] = getDisparity_cu(gs.lines->norm4[pindex], gs.lines->norm4[pindex].w, p, gs.cameras->cameras[REFERENCE]);

    float4 norm = gs.lines->norm4[pindex];
    float4 norm_transformed;
    // Transform back normal to world coordinate
    matvecmul4 ( gs.cameras->cameras[REFERENCE].R_orig_inv, norm, (&norm_transformed));
    if (gs.lines->c[pindex] != MAXCOST)
        norm_transformed.w = getDisparity_cu (norm, norm.w, p, gs.cameras->cameras[REFERENCE] );
    else
        norm_transformed.w = 0;
    gs.lines->norm4[pindex] = norm_transformed;
    return;
}

__global__ void gipuma_compute_disp (GlobalState &gs)
{
    const int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;
    CameraParameters_cu &camParams = *(gs.cameras);
    const int center = p.y*cols+p.x;
    
    int pindex = p.y*cols+p.x;
    
    //if (gs.lines->scale[pindex] == 0) {
    //     gs.lines->norm4[pindex].x = 0;
    //    gs.lines->norm4[pindex].y = 0;
    //    gs.lines->norm4[pindex].z = -1;
    //    gs.lines->norm4[pindex].w = getD_cu(gs.lines->norm4[pindex], p,
    //                                    camParams.cameras[REFERENCE].depthMax,
    //                                    gs.cameras->cameras[REFERENCE]);
    //}
    
    float4 norm = gs.lines->norm4[center];
    float4 norm_transformed;
    // Transform back normal to world coordinate
    matvecmul4 ( gs.cameras->cameras[REFERENCE].R_orig_inv, norm, (&norm_transformed));
    if (gs.lines->c[center] != MAXCOST)
        norm_transformed.w = getDisparity_cu (norm, norm.w, p, gs.cameras->cameras[REFERENCE] );
    else
        norm_transformed.w = 0;
    gs.lines->norm4[center] = norm_transformed;
    return;
}

template< typename T >
__device__ FORCEINLINE_GIPUMA void gipuma_checkerboard_spatialProp_cu(GlobalState &gs, int2 p, bool final)
{
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;
    if (final == true && gs.lines->text[pindex] == -1) return;

    int box_hrad = (gs.params->box_hsize-1) / 2;
    int box_vrad = (gs.params->box_vsize-1) / 2;
    AlgorithmParameters &algParams = *(gs.params);
    CameraParameters_cu &camParams = *(gs.cameras);
    const cudaTextureObject_t *imgs = gs.imgs;

    
    
    float cost_now = gs.lines->c[pindex];
    float4 norm_now = gs.lines->norm4[pindex];
    float disp_now = getDisparity_cu (norm_now, norm_now.w, p, camParams.cameras[REFERENCE] );

    float4 *norm = gs.lines->norm4;

    #define SPATIALPROPAGATION(point) spatialPropagation_cu<T> (gs, imgs, p, box_hrad, box_vrad, algParams, camParams, &cost_now, &norm_now, norm[point], &disp_now, norm, point, final)

    int left_near = pindex - 1;
    int left_far = pindex - 3;
    int right_near = pindex + 1;
    int right_far = pindex + 3;
    int up_near = pindex - gs.col;
    int up_far = pindex - 3 * gs.col;
    int down_near = pindex + gs.col;
    int down_far = pindex + 3 * gs.col;

    bool flag[8] = {false};
    int num_valid_pixels = 0;
    float costMin;
    int costMinPoint;

    // up_far
    if (p.y > 2) {
        costMin = gs.lines->c[up_far];
        costMinPoint = up_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y > 2 + 2 * i) {
                int pointTemp = up_far - 2 * i * gs.col;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        SPATIALPROPAGATION(costMinPoint);
    }

    // down_far
    if (p.y < gs.row - 3) {
        costMin = gs.lines->c[up_far];
        costMinPoint = down_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y < gs.row - 3 - 2 * i) {
                int pointTemp = down_far + 2 * i * gs.col;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        SPATIALPROPAGATION(costMinPoint);
    }

    // left_far
    if (p.x > 2) {
        costMin = gs.lines->c[left_far];
        costMinPoint = left_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x > 2 + 2 * i) {
                int pointTemp = left_far - 2 * i;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        SPATIALPROPAGATION(costMinPoint);
    }

    // right_far
    if (p.x < gs.col - 3) {
        costMin = gs.lines->c[right_far];
        costMinPoint = right_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x < gs.col - 3 - 2 * i) {
                int pointTemp = right_far + 2 * i;
                if (costMin < gs.lines->c[pointTemp]) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        SPATIALPROPAGATION(costMinPoint);
    }

    // up_near
    if (p.y > 0) {
        costMin = gs.lines->c[up_near];
        costMinPoint = up_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y > 1 + i && p.x > i) {
                int pointTemp = up_near - (1 + i) * gs.col - i;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.y > 1 + i && p.x < gs.col - 1 - i) {
                int pointTemp = up_near - (1 + i) * gs.col + i;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        SPATIALPROPAGATION(costMinPoint);
    }

    // down_near
    if (p.y < gs.row - 1) {
        costMin = gs.lines->c[down_near];
        costMinPoint = down_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y < gs.row - 2 - i && p.x > i) {
                int pointTemp = down_near + (1 + i) * gs.col - i;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.y < gs.row - 2 - i && p.x < gs.col - 1 - i) {
                int pointTemp = down_near + (1 + i) * gs.col + i;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        SPATIALPROPAGATION(costMinPoint);
    }

    // left_near
    if (p.x > 0) {
        costMin = gs.lines->c[left_near];
        costMinPoint = left_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x > 1 + i && p.y > i) {
                int pointTemp = left_near - (1 + i) - i * gs.col;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.x > 1 + i && p.y < gs.row - 1 - i) {
                int pointTemp = left_near - (1 + i) + i * gs.col;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        SPATIALPROPAGATION(costMinPoint);
    }

    // right_near
    if (p.x < gs.col - 1) {
        costMin = gs.lines->c[right_near];
        costMinPoint = right_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x < gs.col - 2 - i && p.y > i) {
                int pointTemp = right_near + (1 + i) - i * gs.col;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.x < gs.col - 2 - i && p.y < gs.row - 1 - i) {
                int pointTemp = right_near + (1 + i) + i * gs.col;
                if (gs.lines->c[pointTemp] < costMin) {
                    costMin = gs.lines->c[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        SPATIALPROPAGATION(costMinPoint);
    }
    if (p.y * gs.col + p.x == 190430) {
        printf("after prop : %f \n", cost_now);
    }

    gs.lines->c[pindex] = cost_now;
    gs.lines->norm4[pindex] = norm_now;
    return;
}


template< typename T >
__device__ FORCEINLINE_GIPUMA void gipuma_checkerboard_planeRefinement_cu(GlobalState &gs, int2 p, bool final)
{
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;
    if (final == true && gs.lines->text[pindex] == -1) return;

    int box_hrad = (gs.params->box_hsize - 1) / 2;
    int box_vrad = (gs.params->box_vsize - 1) / 2;
    AlgorithmParameters &algParams = *(gs.params);
    CameraParameters_cu &camParams = *(gs.cameras);
    const cudaTextureObject_t *imgs = gs.imgs;

    float cost_now = gs.lines->c[pindex];
    float4 norm_now = gs.lines->norm4[pindex];
    float disp_now = getDisparity_cu(norm_now, norm_now.w, p, camParams.cameras[REFERENCE]);

    float4 *norm = gs.lines->norm4;
    curandState localState = gs.cs[p.y * cols + p.x];
    curand_init(clock64(), p.y, p.x, &localState);

    planeRefinement_cu<T> (gs, imgs,
                        p,
                        box_hrad, box_vrad,
                        algParams, camParams,
                        REFERENCE,
                        &cost_now,
                        &norm_now,
                        &disp_now,
                        localState,
                        norm, final);

    gs.lines->c[pindex] = cost_now;
    gs.lines->norm4[pindex] = norm_now;
    
    return;
}

template< typename T >
__global__ void gipuma_black_spatialProp_cu(GlobalState &gs, bool final)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2;
    else
        p.y = p.y*2 + 1;
    gipuma_checkerboard_spatialProp_cu<T>(gs, p, final);
}

template< typename T >
__global__ void gipuma_black_planeRefine_cu(GlobalState &gs, bool final)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2;
    else
        p.y = p.y*2 + 1;
    gipuma_checkerboard_planeRefinement_cu<T>(gs, p, final);
}

template< typename T >
__global__ void gipuma_red_spatialProp_cu(GlobalState &gs, bool final)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2 + 1;
    else
        p.y = p.y*2;
    gipuma_checkerboard_spatialProp_cu<T>(gs, p, final);
}

template< typename T >
__global__ void gipuma_red_planeRefine_cu(GlobalState &gs, bool final)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2 + 1;
    else
        p.y = p.y*2;
    gipuma_checkerboard_planeRefinement_cu<T>(gs, p, final);
}

template <typename T>
__global__ void gipuma_dptow(GlobalState &gs) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;         //每个点像素下标
    CameraParameters_cu &camParams = *(gs.cameras);

    float disp = disparityDepthConversion_cu(
        camParams.f, camParams.cameras[0].baseline, gs.lines->depth[pindex]);

    gs.lines->norm4[pindex].w =
        getD_cu(gs.lines->norm4[pindex], p, disp, gs.cameras->cameras[REFERENCE]);
}

template <typename T>
__global__ void gipuma_getlrdiff(GlobalState &gs) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;  //每个点像素下标
    int idxCurr = gs.lines->beview[pindex];
    AlgorithmParameters &algParam = *(gs.params);
    CameraParameters_cu &camParams = *(gs.cameras);
    const cudaTextureObject_t *images = gs.imgs;
    int hrad = (gs.params->box_hsize - 1) / 2;
    int vrad = (gs.params->box_vsize - 1) / 2;
    float4 normal = gs.lines->norm4[pindex];
    

    float rcost = rlCost<T>(images[REFERENCE], images[idxCurr], p.x, p.y, normal, vrad, hrad,
              algParam, camParams, idxCurr);
    gs.lines->lrdiff[pindex] = abs(gs.lines->c[pindex] - rcost);
    if (gs.lines->lrdiff[pindex] > 1) {
        gs.lines->lrdiff[pindex] = 1;
    }
}

template <typename T>
__global__ void gipuma_getview(GlobalState &gs) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;         //每个点像素下标

    //gs.lines->confid[pindex] = ((2 - gs.lines->c[pindex]) / 2 + gs.lines->ratio[pindex] + (1 - gs.lines->lrdiff[pindex])) / 3;
    gs.lines->confid[pindex] = ((2 - gs.lines->c[pindex]) / 2 + (1 - gs.lines->lrdiff[pindex])) / 2;

    CameraParameters_cu &camParams = *(gs.cameras);
    //if (gs.lines->scale[pindex] == 1 && gs.lines->confid[pindex] < 0.8) {   
    //    gs.lines->scale[pindex] = 0;
    //}

    gs.lines->depth[pindex] = getDisparity_cu(gs.lines->norm4[pindex], gs.lines->norm4[pindex].w, p,
                                              gs.cameras->cameras[REFERENCE]);
    gs.lines->depth[pindex] = disparityDepthConversion_cu(
        camParams.f, camParams.cameras[0].baseline, gs.lines->depth[pindex]);

}

template <typename T>
__global__ void gipuma_update_scale(GlobalState &gs) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;         //每个点像素下标
    int canny = gs.lines->canny[pindex];
    CameraParameters_cu &camParams = *(gs.cameras);
    Camera_cu &camera = gs.cameras->cameras[REFERENCE];

    if (gs.cannylines->text[canny] == -1) {//所有弱纹理

        gs.lines->c[pindex] = 0;

        gs.lines->scale[pindex] = 1;
        float4 viewVector;
        getViewVector_cu(&viewVector, camera, p);  //获得viewVector

        float4 norm_now = gs.cannylines->norm4[canny];
        float dp  = norm_now.x * viewVector.x + norm_now.y * viewVector.y + norm_now.z * viewVector.z; 
        if (dp > 0.0f) {
            norm_now.x *= -1;
            norm_now.y *= -1;
            norm_now.z *= -1;
            norm_now.w *= -1;
        }
        gs.lines->norm4[pindex] = norm_now;
    } 
    /*
    else if (gs.cannylines->text[canny] == 1) {
        if (gs.lines->ransa[pindex] == 1) {
            gs.lines->scale[pindex] = 1;
        }
    }
    */
    float disp_now = getDisparity_cu(gs.lines->norm4[pindex], gs.lines->norm4[pindex].w, p,
                        gs.cameras->cameras[REFERENCE]);
    gs.lines->depth[pindex] = disparityDepthConversion_cu(
        camParams.f, camParams.cameras[0].baseline, disp_now);
}

template <typename T>
__global__ void gipuma_update_scale_2(GlobalState &gs) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;         //每个点像素下标
    int canny = gs.lines->canny[pindex];
    CameraParameters_cu &camParams = *(gs.cameras);
    Camera_cu &camera = gs.cameras->cameras[REFERENCE];

    if (gs.cannylines->text[canny] == -1) {//所有弱纹理

        float4 viewVector;
        getViewVector_cu(&viewVector, camera, p);  //获得viewVector

        float4 norm_now = gs.cannylines->norm4[canny];
        float dp  = norm_now.x * viewVector.x + norm_now.y * viewVector.y + norm_now.z * viewVector.z; 
        if (dp > 0.0f) {
            norm_now.x *= -1;
            norm_now.y *= -1;
            norm_now.z *= -1;
            norm_now.w *= -1;
        }
        gs.lines->fakedepth[pindex] = getDisparity_cu(norm_now, norm_now.w, p, gs.cameras->cameras[REFERENCE]);
    }

}

template <typename T>
__global__ void gipuma_WMF_Final(GlobalState &gs, int iter) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;  //每个点像素下标
    int canny = gs.lines->canny[pindex];
    CameraParameters_cu &camParams = *(gs.cameras);
    AlgorithmParameters &algParams = *(gs.params);
    const cudaTextureObject_t *imgs = gs.imgs;

    if (pindex == 100) {
        printf("wmf: %d \n", iter);
    }

    float w[12 * 12] = {0};
    float d[12 * 12] = {0};
    int n[12 * 12] = {0};
    float x[12 * 12] = {0};
    float y[12 * 12] = {0};
    float z[12 * 12] = {0};
    float w1[12 * 12] = {0};
    float w2[12 * 12] = {0};
    float w3[12 * 12] = {0};

    int po = pow(2, iter); // 1 2 4 8 16 32

    int radius = 5 * po;
    int gap = 1 * po;

    //float sigma_spatial = 5.0f;
    //float sigma_color = 3.0f;
    float sigma_spatial = 2.0f;
    float sigma_color = 3.0f;

    int num = 0;
    int cnt = 0;
    if (gs.cannylines->text[canny] == 1 && gs.lines->scale[pindex] == 0) {
        for (int i = -radius; i <= radius;) {
            for (int j = -radius; j <= radius;) {
                int2 pnei = make_int2(p.x + i, p.y + j);
                int neindex = pnei.y * gs.col + pnei.x;
                if (pnei.x >= 0 && pnei.x < gs.col && pnei.y >= 0 && pnei.y < gs.row) {
                    if (gs.lines->scale[neindex] == 1) { 
                        float cen_pix = tex2D<float>(imgs[REFERENCE], p.x + 0.5f, p.y + 0.5f);
                        float ref_pix = tex2D<float>(imgs[REFERENCE], pnei.x + 0.5f, pnei.y + 0.5f);
                        float color_dist = fabs(ref_pix - cen_pix);
                        float spatial_dist = sqrtf(i * i + j * j) / po;
                        float weight =  exp(-spatial_dist / (sigma_spatial* sigma_spatial)) * exp(- color_dist / (sigma_color * sigma_color));
                        w[num] = weight;
                        d[num] = gs.lines->depth[neindex];
                        n[num] = neindex;
                        x[num] = gs.lines->norm4[neindex].x;
                        y[num] = gs.lines->norm4[neindex].y;
                        z[num] = gs.lines->norm4[neindex].z;
                        w1[num] = weight;
                        w2[num] = weight;
                        w3[num] = weight;
                        num++;
                    }
                    cnt++;
                }
                j += gap;
            }
            i += gap;
        }

        //        double ratio = (double)num / (double)cnt;
        int ths = 32 / po; // 6次，如果4次改成8：  1 2 4 8

        if (num >= ths) {
            for (int i = 0; i < num; i++) {
                for (int j = 0; j < num - i; j++) {
                    if (d[j] > d[j + 1]) {
                        float ww = w[j];
                        w[j] = w[j + 1];
                        w[j + 1] = ww;

                        float dd = d[j];
                        d[j] = d[j + 1];
                        d[j + 1] = dd;

                        int nn = n[j];
                        n[j] = n[j + 1];
                        n[j + 1] = nn;
                    }

                    if (x[j] > x[j + 1]) {
                        float xx = x[j];
                        x[j] = x[j + 1];
                        x[j + 1] = xx;

                        float ww = w1[j];
                        w1[j] = w1[j + 1];
                        w1[j + 1] = ww;
                    }

                    if (y[j] > y[j + 1]) {
                        float yy = y[j];
                        y[j] = y[j + 1];
                        y[j + 1] = yy;

                        float ww = w2[j];
                        w2[j] = w2[j + 1];
                        w2[j + 1] = ww;
                    }

                    if (z[j] > z[j + 1]) {
                        float zz = z[j];
                        z[j] = z[j + 1];
                        z[j + 1] = zz;

                        float ww = w3[j];
                        w3[j] = w3[j + 1];
                        w3[j + 1] = ww;
                    }
                }
            }

            float wAcc = 0.f, wSum = 0.f;
            for (int i = 0; i < num; i++) {
                wSum += w[i];
            }

            float4 norm_mid;
            wAcc = 0;
            for (int i = 0; i < num; i++) {
                wAcc += w1[i];
                if (wAcc >= wSum / 2.f) {
                    norm_mid.x = x[i];
                    break;
                }
            }
            wAcc = 0;
            for (int i = 0; i < num; i++) {
                wAcc += w2[i];
                if (wAcc >= wSum / 2.f) {
                    norm_mid.y = y[i];
                    break;
                }
            }
            wAcc = 0;
            for (int i = 0; i < num; i++) {
                wAcc += w3[i];
                if (wAcc >= wSum / 2.f) {
                    norm_mid.z = z[i];
                    break;
                }
            }

            wAcc = 0;
            for (int i = 0; i < num; i++) {
                wAcc += w[i];
                if (wAcc >= wSum / 2.f) {
                    int weimid = n[i];
                    // norm_mid = gs.lines->norm4[weimid];

                    float disp_mid = disparityDepthConversion_cu(
                        camParams.f, camParams.cameras[0].baseline,
                        gs.lines->depth[weimid]);
                    int2 pmid = make_int2(weimid % gs.col, weimid / gs.col);

                    double xyzsqr = sqrtf(norm_mid.x * norm_mid.x +
                                          norm_mid.y * norm_mid.y +
                                          norm_mid.z * norm_mid.z);
                    norm_mid.x /= xyzsqr;
                    norm_mid.y /= xyzsqr;
                    norm_mid.z /= xyzsqr;
                    norm_mid.w = getD_cu(norm_mid, pmid, disp_mid,
                                         gs.cameras->cameras[REFERENCE]);

                    gs.lines->norm4[pindex] = norm_mid;

                    float disp_now = getDisparity_cu(
                        gs.lines->norm4[pindex], gs.lines->norm4[pindex].w, p,
                        gs.cameras->cameras[REFERENCE]);

                    gs.lines->depth[pindex] = disparityDepthConversion_cu(
                        camParams.f, camParams.cameras[0].baseline, disp_now);

                    if (gs.lines->depth[pindex] <= algParams.min_disparity ||
                        gs.lines->depth[pindex] >= algParams.max_disparity) {
                        gs.lines->scale[pindex] = 0;
                        gs.lines->depth[pindex] = algParams.min_disparity;
                    } else {
                        gs.lines->scale[pindex] = 1;
                    }

                    break;
                }
            }
        }

        
        
        

    }
}

template <typename T>
__global__ void gipuma_WMF(GlobalState &gs, int iter) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x >= cols) return;
    if (p.y >= rows) return;

    int pindex = p.y * cols + p.x;  //每个点像素下标
    int canny = gs.lines->canny[pindex];
    CameraParameters_cu &camParams = *(gs.cameras);
    AlgorithmParameters &algParams = *(gs.params);
    const cudaTextureObject_t *imgs = gs.imgs;
    
    if (pindex == 100) {
        printf("wmf: %d \n", iter);
    }

    float w[12 * 12] = {0};
    float d[12 * 12] = {0};
    int n[12 * 12] = {0};
    float x[12 * 12] = {0};
    float y[12 * 12] = {0};
    float z[12 * 12] = {0};
    float w1[12 * 12] = {0};
    float w2[12 * 12] = {0};
    float w3[12 * 12] = {0};

    int po = pow(2, iter); // 1 2 4 8
    int repo = pow(2, 3 - iter); // 8 4 2 1

    int radius = 80 / po;
    int gap = 16 / po;

    // float sigma_spatial = 5.0f;
    // float sigma_color = 3.0f;
    float sigma_spatial = 2.0f;
    float sigma_color = 3.0f;

    int num = 0;
    for (int i = -radius; i <= radius;) {
        for (int j = -radius; j <= radius;) {
            int2 pnei = make_int2(p.x + i, p.y + j);
            int neindex = pnei.y * gs.col + pnei.x;
            if (pnei.x >= 0 && pnei.x < gs.col && pnei.y >= 0 && pnei.y < gs.row && gs.lines->scale[neindex] == 1) {
                float cen_pix = tex2D<float>(imgs[REFERENCE], p.x + 0.5f, p.y + 0.5f);
                float ref_pix = tex2D<float>(imgs[REFERENCE], pnei.x + 0.5f, pnei.y + 0.5f);
                float color_dist = fabs(ref_pix - cen_pix);
                float spatial_dist = sqrtf(i * i + j * j) / repo;
                float weight =  exp(-spatial_dist / (sigma_spatial* sigma_spatial)) * exp(- color_dist / (sigma_color * sigma_color));
                //float weight =  exp(- color_dist / 10);

                if (true) {
                    //weight = (10 - color_dist) / 10;
                    w[num] = weight;
                    d[num] = gs.lines->depth[neindex];
                    n[num] = neindex;
                    x[num] = gs.lines->norm4[neindex].x;
                    y[num] = gs.lines->norm4[neindex].y;
                    z[num] = gs.lines->norm4[neindex].z;
                    w1[num] = weight;
                    w2[num] = weight;
                    w3[num] = weight;
                    num++;
                }
            }
            j += gap;
        }
        i += gap;
    }

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < num - i; j++) {
            if (d[j] > d[j + 1]) {
                float ww = w[j];
                w[j] = w[j + 1];
                w[j + 1] = ww;
                    
                float dd = d[j];
                d[j] = d[j + 1];
                d[j + 1] = dd;

                int nn = n[j];
                n[j] = n[j + 1];
                n[j + 1] = nn;
            }

            if (x[j] > x[j + 1]) {
                float xx = x[j];
                x[j] = x[j + 1];
                x[j + 1] = xx;

                float ww = w1[j];
                w1[j] = w1[j + 1];
                w1[j + 1] = ww;
            }

            if (y[j] > y[j + 1]) {
                float yy = y[j];
                y[j] = y[j + 1];
                y[j + 1] = yy;

                float ww = w2[j];
                w2[j] = w2[j + 1];
                w2[j + 1] = ww;
            }

            if (z[j] > z[j + 1]) {
                float zz = z[j];
                z[j] = z[j + 1];
                z[j + 1] = zz;

                float ww = w3[j];
                w3[j] = w3[j + 1];
                w3[j + 1] = ww;
            }
        }
    }
        
    float wAcc = 0.f, wSum = 0.f;
    for (int i = 0; i < num; i++) {
        wSum += w[i];
    }

    float4 norm_mid;
    wAcc = 0;
    for (int i = 0; i < num; i++) {
        wAcc += w1[i];
        if (wAcc >= wSum / 2.f) {
            norm_mid.x = x[i];
            break;
        }
    }
    wAcc = 0;
    for (int i = 0; i < num; i++) {
        wAcc += w2[i];
        if (wAcc >= wSum / 2.f) {
            norm_mid.y = y[i];
            break;
        }
    }
    wAcc = 0;
    for (int i = 0; i < num; i++) {
        wAcc += w3[i];
        if (wAcc >= wSum / 2.f) {
            norm_mid.z = z[i];
            break;
        }
    }

    wAcc = 0;
    for (int i = 0; i < num; i++) {
        wAcc += w[i];
        if (wAcc >= wSum / 2.f) {
                
            int weimid = n[i];
            //norm_mid = gs.lines->norm4[weimid];
            
            float disp_mid = disparityDepthConversion_cu(
                camParams.f, camParams.cameras[0].baseline, gs.lines->depth[weimid]);
            int2 pmid = make_int2(weimid % gs.col,weimid / gs.col);

            double xyzsqr = sqrtf(norm_mid.x * norm_mid.x + norm_mid.y * norm_mid.y + norm_mid.z * norm_mid.z);
            norm_mid.x /= xyzsqr;
            norm_mid.y /= xyzsqr;
            norm_mid.z /= xyzsqr;
            norm_mid.w = getD_cu(norm_mid, pmid, disp_mid, gs.cameras->cameras[REFERENCE]);
            
            break;
        }
    }

    int ths = 24 / po;

    if (num > 0) {
        float depth_now = getDisparity_cu(norm_mid, norm_mid.w, p,
                                        gs.cameras->cameras[REFERENCE]);

        float disp_now = disparityDepthConversion_cu(camParams.f, camParams.cameras[0].baseline, depth_now);

        float depth_org = getDisparity_cu(gs.lines->norm4[pindex], gs.lines->norm4[pindex].w, p,
                                                gs.cameras->cameras[REFERENCE]);

        float disp_org = disparityDepthConversion_cu(camParams.f, camParams.cameras[0].baseline, depth_org);

        if (abs(disp_now - disp_org) > ths ||
            abs(depth_org - camParams.cameras[REFERENCE].depthMin) < DEPTH_THS_MIN ||
            abs(depth_org - camParams.cameras[REFERENCE].depthMax) < DEPTH_THS_MAX) {  // 20 15 10 5
            gs.lines->scale[pindex] = 0;
        } else {
            gs.lines->scale[pindex] = 1;
        }
    } else {
        gs.lines->scale[pindex] = 0;
    }


}

template< typename T >
void gipuma_first(GlobalState &gs) {
#ifdef SHARED
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#else
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif

    int rows = gs.cameras->rows;
    int cols = gs.cameras->cols;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc ( &gs.cs, rows*cols*sizeof( curandState ) ));

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W/2);

    dim3 grid_size;
    grid_size.x=(cols+BLOCK_W-1)/BLOCK_W;
    grid_size.y=((rows/2)+BLOCK_H-1)/BLOCK_H;
    dim3 block_size;
    block_size.x=BLOCK_W;
    block_size.y=BLOCK_H;

    dim3 grid_size_initrand;
    grid_size_initrand.x=(cols+16-1)/16;
    grid_size_initrand.y=(rows+16-1)/16;
    dim3 block_size_initrand;
    block_size_initrand.x=16;
    block_size_initrand.y=16;

    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    int maxiter=gs.params->iterations;
    printf("Device memory used: %fMB\n", used/1000000.0f);
    printf("Blocksize is %dx%d\n", gs.params->box_hsize,gs.params->box_vsize);

    //gipuma_init_cu2<T><<< grid_size_initrand, block_size_initrand>>>(gs);
    //cudaEventRecord(start);
    printf("Iteration ");
    //for (int it =0;it<maxiter; it++) {
    //    printf("%d ", it+1);
    //    gipuma_black_spatialProp_cu<T><<< grid_size, block_size>>>(gs, false);
    //    cudaDeviceSynchronize();
    //    gipuma_black_planeRefine_cu<T><<<grid_size, block_size>>>(gs, false);
    //    cudaDeviceSynchronize();
    //    gipuma_red_spatialProp_cu<T><<<grid_size, block_size>>>(gs, false);
    //    cudaDeviceSynchronize();
    //    gipuma_red_planeRefine_cu<T><<<grid_size, block_size>>>(gs, false);
    //    cudaDeviceSynchronize();
    //}
    gipuma_get_disp<T><<<grid_size_initrand, block_size_initrand>>>(gs);
    cudaDeviceSynchronize();

    //gipuma_getlrdiff<T><<<grid_size_initrand, block_size_initrand>>>(gs);
    //cudaDeviceSynchronize();

    //gipuma_compute_disp<<<grid_size_initrand, block_size_initrand>>>(gs);
    //cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nTotal time needed for computation: %f seconds\n", milliseconds/1000.f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    // print results to file
    cudaFree (&gs.cs);
}

template <typename T>
void gipuma_slic(GlobalState &gs) {
#ifdef SHARED
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#else
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif

    int rows = gs.cameras->rows;
    int cols = gs.cameras->cols;

    dim3 block_size_slic;
    block_size_slic.x = 16;
    block_size_slic.y = 16;

    dim3 grid_size_initrand;
    grid_size_initrand.x = (cols + 16 - 1) / 16;
    grid_size_initrand.y = (rows + 16 - 1) / 16;
    dim3 block_size_initrand;
    block_size_initrand.x = 16;
    block_size_initrand.y = 16;

    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    int maxiter = gs.params->iterations;

    gipuma_getview<T><<<grid_size_initrand, block_size_initrand>>>(gs);
    cudaDeviceSynchronize();
    
    //for (int iter = 0; iter < 4; iter++) {
    //    gipuma_WMF<T><<<grid_size_initrand, block_size_initrand>>>(gs, iter);  //带入倾斜面
    //    cudaDeviceSynchronize();
    //}

    //gipuma_compute_disp<<<grid_size_initrand, block_size_initrand>>>(gs);
    //cudaDeviceSynchronize();
}


template <typename T>
void gipuma_fill(GlobalState &gs) {
#ifdef SHARED
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#else
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif

    int rows = gs.cameras->rows;
    int cols = gs.cameras->cols;

    dim3 grid_size_initrand;
    grid_size_initrand.x = (cols + 16 - 1) / 16;
    grid_size_initrand.y = (rows + 16 - 1) / 16;
    dim3 block_size_initrand;
    block_size_initrand.x = 16;
    block_size_initrand.y = 16;

    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    int maxiter = gs.params->iterations;
    gipuma_update_scale<T><<<grid_size_initrand, block_size_initrand>>>(gs);
    cudaDeviceSynchronize();
    //for (int iter = 0; iter < 6; iter++) {
    //    gipuma_WMF_Final<T><<<grid_size_initrand, block_size_initrand>>>(gs, iter);
    //    cudaDeviceSynchronize();
    //}
    gipuma_compute_disp<<<grid_size_initrand, block_size_initrand>>>(gs);
    cudaDeviceSynchronize();
}

template <typename T>
void gipuma_fake(GlobalState &gs) {
#ifdef SHARED
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#else
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif

    int rows = gs.cameras->rows;
    int cols = gs.cameras->cols;

    dim3 grid_size_initrand;
    grid_size_initrand.x = (cols + 16 - 1) / 16;
    grid_size_initrand.y = (rows + 16 - 1) / 16;
    dim3 block_size_initrand;
    block_size_initrand.x = 16;
    block_size_initrand.y = 16;

    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    int maxiter = gs.params->iterations;
    gipuma_update_scale_2<T><<<grid_size_initrand, block_size_initrand>>>(gs);
    cudaDeviceSynchronize();
}

int firstcuda(GlobalState &gs) {
    //printf("Run cuda\n");
    if(gs.params->color_processing)
        gipuma_first<float4>(gs);
    else
        gipuma_first<float>(gs);
    return 0;
}

int sliccuda(GlobalState &gs) {
    // printf("Run cuda\n");
    if (gs.params->color_processing)
        gipuma_slic<float4>(gs);
    else
        gipuma_slic<float>(gs);
    return 0;
};

int fillcuda(GlobalState &gs) {
    // printf("Run cuda\n");
    if (gs.params->color_processing)
        gipuma_fill<float4>(gs);
    else
        gipuma_fill<float>(gs);
    return 0;
};

int fakecuda(GlobalState &gs) {
    // printf("Run cuda\n");
    if (gs.params->color_processing)
        gipuma_fake<float4>(gs);
    else
        gipuma_fake<float>(gs);
    return 0;
};
