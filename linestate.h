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
    float *depth; //��¼�Ӳ�
    float *fakedepth;
    float4 *resize4;
    float *canny;  //��¼�����������±�

    int *cenxi;  //���ĵ�λ��x
    int *cenyi;  //���ĵ�λ��y

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

        cudaMallocManaged(&resize4, sizeof(float4) * n); //��¼�Ŵ���ԭ���
        memset(resize4, 0, sizeof(float4) * n);

        cudaMallocManaged(&text, sizeof(float) * n);  //��¼�����Ƿ���������
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

        cudaMallocManaged(&XYZ, sizeof(float3) * n);  //��¼ÿ����������ʵ�����XYZ
        memset(XYZ, 0, sizeof(float3) * n);

        cudaMallocManaged(&scale, sizeof(float) * n); //��¼�Ƿ���˸����ص�
        memset(scale, 0, sizeof(float) * n);
        cudaMallocManaged(&ransa, sizeof(float) * n); //��¼�Ƿ���˸����ص�
        memset(ransa, 0, sizeof(float) * n);

        cudaMallocManaged(&canny, sizeof(float) * n);
        memset(canny, 0, sizeof(float) * n);

        cudaMallocManaged(&ratio, sizeof(float) * n); //��¼�ɱ���С�͵ڶ�С�ı�ֵ
        memset(ratio, 0, sizeof(float) * n);

        cudaMallocManaged(&beview, sizeof(int) * n); //��¼�ɱ���С��Ӧ���ӽ�
        memset(beview, 0, sizeof(int) * n);

        cudaMallocManaged(&lrdiff, sizeof(float) * n); //��¼ԭ�ӽ� �� �ɱ���С��Ӧ���ӽ� ����ת��ref src��ġ��³ɱ���
        memset(lrdiff, 0, sizeof(float) * n);

        cudaMallocManaged(&confid, sizeof(float) * n); //��¼���Ŷ�
        memset(confid, 0, sizeof(float) * n);

        cudaMallocManaged(&pind, sizeof(int) * n);  //��¼���Ŷ�
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

        cudaMallocManaged(&nein, sizeof(int) * n); //��¼ÿ���������ھ�����
        memset(nein, 0, sizeof(int) * n);

        cudaMallocManaged(&nump, sizeof(int) * n); //��¼ÿ�������������������
        memset(nump, 0, sizeof(int) * n);

        cudaMallocManaged(&text, sizeof(float) * n); //��¼������RANSAC���Ƿ���"��"������
        memset(text, 0, sizeof(float) * n);

        cudaMallocManaged(&canny, sizeof(float) * n);
        memset(canny, 0, sizeof(float) * n);

        cudaMallocManaged(&ranumax, sizeof(float) * n); //��¼ÿ��������ransac�õ������������
        memset(ranumax, 0, sizeof(float) * n);
    }

    void NeipAndEacp(int pnum, int spnum, std::vector<std::vector<int> > A, std::vector<std::vector<int> > B) {
        cudaMallocManaged(&neip, sizeof(int) * pnum);
        for (int i = 0; i < spnum; i++) {
            cudaMallocManaged(&neip[i], sizeof(int) * A[i].size());//��¼ÿ���������ھ�
            memset(neip[i], 0, sizeof(int) * A[i].size());
        }

        cudaMallocManaged(&eacp, sizeof(int) * pnum);
        for (int i = 0; i < spnum; i++) {
            cudaMallocManaged(&eacp[i], sizeof(int) * B[i].size());;//��¼ÿ���������������
            memset(eacp[i], 0, sizeof(int) * B[i].size());
        }

        cudaMallocManaged(&ranp, sizeof(int) * pnum);
        for (int i = 0; i < spnum; i++) {
            cudaMallocManaged(&ranp[i], sizeof(int) * B[i].size());//��¼ÿ���������������RANSAC�ĺó�����
            memset(ranp[i], 0, sizeof(int) * B[i].size());
        }


    }

    void Cannyresize(int n) {
        cudaMallocManaged(&norm4, sizeof(float4) * n);
        memset(norm4, 0, sizeof(float4) * n);

        cudaMallocManaged(&text, sizeof(float) * n); //��¼�����Ƿ���������
        memset(text, 0, sizeof(float) * n);

        cudaMallocManaged(&cenxi, sizeof(int) * n);
        memset(cenxi, 0, sizeof(int) * n);

        cudaMallocManaged(&cenyi, sizeof(int) * n);
        memset(cenyi, 0, sizeof(int) * n);

        cudaMallocManaged(&size, sizeof(float) * n);
        memset(size, 0, sizeof(float) * n);

        cudaMallocManaged(&borlen, sizeof(int) * n);  //��¼�����Ե���س���
        memset(borlen, 0, sizeof(int) * n);

        cudaMallocManaged(&depdif, sizeof(float) * n);  //��¼�����Ե���س���
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
