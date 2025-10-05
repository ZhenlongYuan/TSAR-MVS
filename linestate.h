#pragma once
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include <vector>
#include "managed.h"
#include <string.h>        // memset()
#include <vector_types.h>  // float4


class __align__(128) LineState : public Managed {
   public:
    float4 *norm4;  // n.x * x + n.y * y + n.z * z + n.w = 0
    float *c;       // cost
    float *depth; //记录视差
    float *fakedepth;
    float4 *resize4;
    float *canny;  //记录所属的纹理下标

    int *cenxi;  //中心点位置x
    int *cenyi;  //中心点位置y

    int *nein;
    int **neip;

    int *nump;
    int **eacp;
    int **ranp;

    int *pind;
    int *borlen;
    float *depdif;

    float *scale;
    float *ransa;

    float *text; 

    float3 *XYZ;

    float *ratio;
    int *beview;
    float *lrdiff;
    float *confid;

    float *ranumax;

    float *size;

    int n;
    int s;  // stride
    int l;  // length

    void bigresize(int n)  // gs.lines
    {
        cudaMallocManaged(&c, sizeof(float) * n);
        memset(c, 0, sizeof(float) * n);

        cudaMallocManaged(&depth, sizeof(float) * n);
        memset(depth, 0, sizeof(float) * n);

        cudaMallocManaged(&norm4, sizeof(float4) * n);
        memset(norm4, 0, sizeof(float4) * n);

        cudaMallocManaged(&resize4, sizeof(float4) * n); //记录放大后的原结果
        memset(resize4, 0, sizeof(float4) * n);

        cudaMallocManaged(&text, sizeof(float) * n);  //记录纹理是否是弱纹理
        memset(text, 0, sizeof(float) * n);
    }

    void resize(int n)  // gs.lines
    {
        cudaMallocManaged(&c, sizeof(float) * n);
        memset(c, 0, sizeof(float) * n);

        cudaMallocManaged(&depth, sizeof(float) * n);
        memset(depth, 0, sizeof(float) * n);

        cudaMallocManaged(&fakedepth, sizeof(float) * n);
        memset(fakedepth, 0, sizeof(float) * n);

        cudaMallocManaged(&norm4, sizeof(float4) * n);
        memset(norm4, 0, sizeof(float4) * n);

        cudaMallocManaged(&XYZ, sizeof(float3) * n);  //记录每个像素在真实世界的XYZ
        memset(XYZ, 0, sizeof(float3) * n);

        cudaMallocManaged(&scale, sizeof(float) * n); //记录是否过滤该像素点
        memset(scale, 0, sizeof(float) * n);
        cudaMallocManaged(&ransa, sizeof(float) * n); //记录是否过滤该像素点
        memset(ransa, 0, sizeof(float) * n);

        cudaMallocManaged(&canny, sizeof(float) * n);
        memset(canny, 0, sizeof(float) * n);

        cudaMallocManaged(&ratio, sizeof(float) * n); //记录成本最小和第二小的比值
        memset(ratio, 0, sizeof(float) * n);

        cudaMallocManaged(&beview, sizeof(int) * n); //记录成本最小对应的视角
        memset(beview, 0, sizeof(int) * n);

        cudaMallocManaged(&lrdiff, sizeof(float) * n); //记录原视角 和 成本最小对应的视角 互相转化ref src后的“新成本”
        memset(lrdiff, 0, sizeof(float) * n);

        cudaMallocManaged(&confid, sizeof(float) * n); //记录置信度
        memset(confid, 0, sizeof(float) * n);

        cudaMallocManaged(&pind, sizeof(int) * n);  //记录置信度
        memset(pind, 0, sizeof(int) * n);
    }

    void SLICresize(int n) {  // gs.splines
        cudaMallocManaged(&depth, sizeof(float) * n);
        memset(depth, 0, sizeof(float) * n);

        cudaMallocManaged(&norm4, sizeof(float4) * n);
        memset(norm4, 0, sizeof(float4) * n);

        cudaMallocManaged(&cenxi, sizeof(int) * n);
        memset(cenxi, 0, sizeof(int) * n);

        cudaMallocManaged(&cenyi, sizeof(int) * n);
        memset(cenyi, 0, sizeof(int) * n);

        cudaMallocManaged(&nein, sizeof(int) * n); //记录每个超像素邻居数量
        memset(nein, 0, sizeof(int) * n);

        cudaMallocManaged(&nump, sizeof(int) * n); //记录每个超像素里的像素数量
        memset(nump, 0, sizeof(int) * n);

        cudaMallocManaged(&text, sizeof(float) * n); //记录超像素RANSAC后是否是"好"超像素
        memset(text, 0, sizeof(float) * n);

        cudaMallocManaged(&canny, sizeof(float) * n);
        memset(canny, 0, sizeof(float) * n);

        cudaMallocManaged(&ranumax, sizeof(float) * n); //记录每个超像素ransac得到的最大像素数
        memset(ranumax, 0, sizeof(float) * n);
    }

    void NeipAndEacp(int pnum, int spnum, std::vector<std::vector<int> > A, std::vector<std::vector<int> > B) {
        cudaMallocManaged(&neip, sizeof(int) * pnum);
        for (int i = 0; i < spnum; i++) {
            cudaMallocManaged(&neip[i], sizeof(int) * A[i].size());//记录每个超像素邻居
            memset(neip[i], 0, sizeof(int) * A[i].size());
        }

        cudaMallocManaged(&eacp, sizeof(int) * pnum);
        for (int i = 0; i < spnum; i++) {
            cudaMallocManaged(&eacp[i], sizeof(int) * B[i].size());;//记录每个超像素里的像素
            memset(eacp[i], 0, sizeof(int) * B[i].size());
        }

        cudaMallocManaged(&ranp, sizeof(int) * pnum);
        for (int i = 0; i < spnum; i++) {
            cudaMallocManaged(&ranp[i], sizeof(int) * B[i].size());//记录每个超像素里的用于RANSAC的好超像素
            memset(ranp[i], 0, sizeof(int) * B[i].size());
        }


    }

    void Cannyresize(int n) {
        cudaMallocManaged(&norm4, sizeof(float4) * n);
        memset(norm4, 0, sizeof(float4) * n);

        cudaMallocManaged(&text, sizeof(float) * n); //记录纹理是否是弱纹理
        memset(text, 0, sizeof(float) * n);

        cudaMallocManaged(&cenxi, sizeof(int) * n);
        memset(cenxi, 0, sizeof(int) * n);

        cudaMallocManaged(&cenyi, sizeof(int) * n);
        memset(cenyi, 0, sizeof(int) * n);

        cudaMallocManaged(&size, sizeof(float) * n);
        memset(size, 0, sizeof(float) * n);

        cudaMallocManaged(&borlen, sizeof(int) * n);  //记录纹理边缘像素长度
        memset(borlen, 0, sizeof(int) * n);

        cudaMallocManaged(&depdif, sizeof(float) * n);  //记录纹理边缘像素长度
        memset(depdif, 0, sizeof(float) * n);
    }

    ~LineState() {
        cudaFree(borlen);
        cudaFree(depdif);

        cudaFree(norm4);
        cudaFree(XYZ);
        cudaFree(c);
        cudaFree(depth);
        cudaFree(fakedepth);
        cudaFree(cenxi);
        cudaFree(cenyi);

        cudaFree(nein);
        cudaFree(neip);
        cudaFree(nump);
        cudaFree(eacp);
        cudaFree(ranp);
        cudaFree(pind);


        cudaFree(scale);
        cudaFree(ransa);
        cudaFree(text);

        cudaFree(canny);

        cudaFree(ratio);
        cudaFree(beview);
        cudaFree(lrdiff);
        cudaFree(confid);

        cudaFree(ranumax);

        cudaFree(size);
    }
};
