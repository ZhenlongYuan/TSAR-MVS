// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_seg_engine.h"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace gSLICr;
using namespace gSLICr::objects;
using namespace gSLICr::engines;


seg_engine::seg_engine(const objects::settings& in_settings)
{
	gSLICr_settings = in_settings;
}


seg_engine::~seg_engine()
{
	if (source_img != NULL) delete source_img;
	if (cvt_img != NULL) delete cvt_img;
	if (idx_img != NULL) delete idx_img;
	if (spixel_map != NULL) delete spixel_map;
}

void seg_engine::Perform_Segmentation(UChar4Image* in_img, GlobalState* gs) {
	source_img->SetFrom(in_img, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CUDA);
	Cvt_Img_Space(source_img, cvt_img, gSLICr_settings.color_space);

	Init_Cluster_Centers();
	Find_Center_Association();

	for (int i = 0; i < gSLICr_settings.no_iters; i++)
	{
		Update_Cluster_Center();
		Find_Center_Association();
	}

	if(gSLICr_settings.do_enforce_connectivity) Enforce_Connectivity();
	cudaThreadSynchronize();

	idx_img->SetFrom(idx_img, ORUtils::MemoryBlock<int>::CUDA_TO_CPU);
    int* p1 = idx_img->data_cpu;
        int pnum = gs->row * gs->col;
        int spnum = spixel_map->noDims.x * spixel_map->noDims.y;
        int sml_col = idx_img->noDims.x;
        int sml_row = idx_img->noDims.y;

        std::vector<std::vector<int> > A; //记录邻居
        std::vector<std::vector<int> > B; //记录每个像素
        std::vector<std::vector<int> > C; //记录邻居长度

        for (int i = 0; i < spnum; i++) A.push_back(std::vector<int>());
        for (int i = 0; i < spnum; i++) B.push_back(std::vector<int>());
        for (int i = 0; i < spnum; i++) C.push_back(std::vector<int>());

        std::cout << "sml_row: " << sml_row << endl;
        std::cout << "sml_col: " << sml_col << endl;
        std::cout << "gs->row: " << gs->row << endl;
        std::cout << "gs->col: " << gs->col << endl;

        for (int i = 0; i < pnum; i++) {
            int sx = i % gs->col;
            int sy = i / gs->col;

            int x = sx / 2;
            int y = sy / 2;
            int spindex;
            if (x == sml_col && y != sml_row) {
                x -= 1;
            } else if (x != sml_col && y == sml_row) {
                y -= 1;
            } else if (x == sml_col && y == sml_row) {
                x -= 1;
                y -= 1;
            }

            spindex = p1[y * sml_col + x]; //这里x，y代表超像素小图， sx,sy代表原大图

            B[spindex].push_back(i);

            if (x > 0) {
                int spneig = p1[y * sml_col + x - 1];
                if (spindex != spneig) {
                    vector<int>::iterator it = find(A[spindex].begin(), A[spindex].end(), spneig);
                    if (it == A[spindex].end()) {
                        A[spindex].push_back(spneig);
                        C[spindex].push_back(1);
                    } else {
                        int inde = it - A[spindex].begin();
                        C[spindex][inde]++;
                    }
                }
            }
            if (x < sml_col - 1) {
                int spneig = p1[y * sml_col + x + 1];
                if (spindex != spneig) {
                    vector<int>::iterator it = find(A[spindex].begin(), A[spindex].end(), spneig);
                    if (it == A[spindex].end()) {
                        A[spindex].push_back(spneig);
                        C[spindex].push_back(1);
                    } else {
                        int inde = it - A[spindex].begin();
                        C[spindex][inde]++;
                    }
                }
            }
            if (y > 0) {
                int spneig = p1[(y - 1) * sml_col + x];
                if (spindex != spneig) {
                    vector<int>::iterator it = find(A[spindex].begin(), A[spindex].end(), spneig);
                    if (it == A[spindex].end()) {
                        A[spindex].push_back(spneig);
                        C[spindex].push_back(1);
                    } else {
                        int inde = it - A[spindex].begin();
                        C[spindex][inde]++;
                    }
                }
            }
            if (y < sml_row - 1) {
                int spneig = p1[(y + 1) * sml_col + x];
                if (spindex != spneig) {
                    vector<int>::iterator it = find(A[spindex].begin(), A[spindex].end(), spneig);
                    if (it == A[spindex].end()) {
                        A[spindex].push_back(spneig);
                        C[spindex].push_back(1);
                    } else {
                        int inde = it - A[spindex].begin();
                        C[spindex][inde]++;
                    }
                }
            }
        }

        int neigmax = 0;
        int numpmax = 0;
        for (int i = 0; i < spnum; i++) {
            neigmax = max(neigmax, A[i].size());
            numpmax = max(numpmax, B[i].size());
        }
        for (int i = 0; i < spnum; i++) {
            if (neigmax == A[i].size()) cout << "neigmax: " << neigmax << " neigmax index: " << i << endl;
            if (numpmax == B[i].size()) cout << "numpmax: " << numpmax << " numpmax index: " << i << endl;
        }
}