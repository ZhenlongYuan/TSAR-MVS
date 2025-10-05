#include "main.h"

#include <ctime>
#include <math.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

// CUDA helper functions
#include "helper_cuda.h"  // helper functions for CUDA error check

#include <sys/stat.h>   // mkdir
#include <sys/types.h>  // mkdir

#ifdef WIN32
#include <win32_dirent.h>  // opendir()
#else
#include <dirent.h>  // opendir()
#endif

#include "algorithmparameters.h"
#include "gipuma.h"

#include "cameraGeometryUtils.h"
#include "displayUtils.h"
#include "fileIoUtils.h"
#include "groundTruthUtils.h"
#include "mathUtils.h"

// SLIC
#include "NVTimer.h"
#include "gSLICr_Lib/gSLICr.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/opencv.hpp"
#include <time.h>


#include <iostream>
#include <random>

using namespace cv;
using namespace std;
const float weakths = 0;

const int Robthr = 4;
const int Houthr = 110;//小于这个被丢弃，越大保留越少
const int minLineLength = 160;//小于这个被丢弃，越大保留越少
const int maxLineGap = 18;//保留小于这个距离阈值的点，越大保留越多
const int weaktextnum = 5000;
const int sizerat = 2.5;

int readNormalDmb (const std::string file_path, cv::Mat_<cv::Vec3f> &normal)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

        cout << " h: " << h << " w: " << w << " type: " << type << " nb: " << nb
         << endl;

    int32_t dataSize = h*w*nb;

    float* data;
    data = (float*) malloc (sizeof(float)*dataSize);
    fread(data,sizeof(float),dataSize,inimage);

    normal = cv::Mat(h,w,CV_32FC3,data);

    fclose(inimage);
    return 0;
}

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    //cout << " h: " << h << " w: " << w << " type: " << type
    //     << " nb: " << nb << endl;

    int32_t dataSize = h*w*nb;

    float* data;
    data = (float*) malloc (sizeof(float)*dataSize);
    fread(data,sizeof(float),dataSize,inimage);

    depth = cv::Mat(h,w,CV_32F,data);

    fclose(inimage);
    return 0;
}

void refinement(GlobalState* gs, vector<Point3f> ptSet, double &a, double &b, double &c, double &d, double &maximum, double residual_error) {
    
	return;
}

void calcLinePara(vector<Point3f> pts, double& A, double& B, double& C, double& D)
{
    double x1 = pts[0].x;
    double y1 = pts[0].y;
    double z1 = pts[0].z;
    double x2 = pts[1].x;
    double y2 = pts[1].y;
    double z2 = pts[1].z;
    double x3 = pts[2].x;
    double y3 = pts[2].y;
    double z3 = pts[2].z;

	A = (y3 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
	B = (x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1);
	C = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
	D = -(A * x1 + B * y1 + C * z1);
	return;
}

void getSample(vector<int> set, vector<int>& sset)
{
	int i[3];
	if (set.size() > 3){
		while (true){
			for (int n = 0; n < 3; n++) {
                int index = rand() % set.size();
				i[n] = index;
			}
			if (i[0] == i[1] || i[0] == i[2] || i[1] == i[2]) {
				continue;
			}
			else {
				sset.push_back(i[0]);
				sset.push_back(i[1]);
				sset.push_back(i[2]);
				break;
			}
		}
	}
    return; 
}

void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg) {
    gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < outimg->noDims.y; y++)
        for (int x = 0; x < outimg->noDims.x; x++) {
            int idx = x + y * outimg->noDims.x;
            outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
            outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
            outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
        }
}

void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg) {
    const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < inimg->noDims.y; y++)
        for (int x = 0; x < inimg->noDims.x; x++) {
            int idx = x + y * inimg->noDims.x;
            outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
            outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
            outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
        }
}

// roberts算子实现
cv::Mat roberts(cv::Mat srcImage) {
	cv::Mat dstImage = srcImage.clone();
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			int t1 = 0;
			int t2 = 0;
			if (i > 0 && i < dstImage.rows - 1 && j > 0 && j < dstImage.cols - 1) {
				t1 = (srcImage.at<uchar>(i, j) -
					srcImage.at<uchar>(i + 1, j + 1)) *
					(srcImage.at<uchar>(i, j) -
						srcImage.at<uchar>(i + 1, j + 1));
				t2 = (srcImage.at<uchar>(i + 1, j) -
					srcImage.at<uchar>(i, j + 1)) *
					(srcImage.at<uchar>(i + 1, j) -
						srcImage.at<uchar>(i, j + 1));
			}
			else {
				t1 = 100 * 50;
				t2 = t1;
			}

			dstImage.at<uchar>(i, j) = (uchar)sqrt(t1 + t2);
		}
	}

	return dstImage;
}

void Connect(Mat dstImage, Mat &labImg, std::vector<int> &labelcnt, std::vector<int> &weaklabel) {
	std::vector<std::vector<int>> leftnei(dstImage.rows);
	std::vector<std::vector<int>> upnei(dstImage.rows);

	for (int y = 0; y < dstImage.rows; y++) {
		for (int x = 0; x < dstImage.cols; x++) {
			if (x == 0) {
				leftnei[y].push_back(0);
			}
			else {
				if (dstImage.at<uchar>(y, x) == 0 && dstImage.at<uchar>(y, x - 1) == 0) {
					leftnei[y].push_back(1);
				}
				else {
					leftnei[y].push_back(0);
				}
			}
		}
	}

	for (int y = 0; y < dstImage.rows; y++) {  //上减下
		for (int x = 0; x < dstImage.cols; x++) {
			if (y == 0) {
				upnei[y].push_back(0);
			}
			else {
				if (dstImage.at<uchar>(y, x) == 0 && dstImage.at<uchar>(y - 1, x) == 0) {
					upnei[y].push_back(1);
				}
				else {
					upnei[y].push_back(0);
				}
			}
		}
	}


	int cnt = 1;
	std::vector<int> connection;
	connection.push_back(0);
	for (int y = 0; y < dstImage.rows; y++) {
		for (int x = 0; x < dstImage.cols; x++) {
			if (dstImage.at<uchar>(y, x) == 255) {
				labImg.at<int>(y, x) = 0;
			}
			else {
				bool left = false;
				bool up = false;
				int index = y * dstImage.cols + x;
				if (leftnei[y][x] == 1) {
					labImg.at<int>(y, x) = labImg.at<int>(y, x - 1);
					left = true;
				}
				if (upnei[y][x] == 1) {
					labImg.at<int>(y, x) = labImg.at<int>(y - 1, x);
					up = true;
				}
				if (left == false && up == false) {
					labImg.at<int>(y, x) = cnt;
					connection.push_back(cnt);
					cnt++;
				}
				else if (left == true && up == true) {
					int left_label = labImg.at<int>(y, x - 1);
					int up_label = labImg.at<int>(y - 1, x);
					if (left_label > up_label) {
						connection[left_label] = up_label;
						labImg.at<int>(y, x) = labImg.at<int>(y - 1, x);
					}
					else if (left_label < up_label) {
						connection[up_label] = left_label;
						labImg.at<int>(y, x) = labImg.at<int>(y, x - 1);
					}
				}
			}

		}
	}

	for (size_t i = 1; i < connection.size(); i++) {
		int curLabel = connection[i];
		int prelabel = connection[curLabel];
		while (prelabel != curLabel) {
			curLabel = prelabel;
			prelabel = connection[prelabel];
		}
		connection[i] = curLabel;
	}

	int labelnum = 1;
	std::vector<int> mapping;
	mapping.push_back(0);
	for (size_t i = 1; i < connection.size(); i++) {
		mapping.push_back(0);
		if (connection[i] == i) {
			mapping[i] = labelnum;
			labelnum++;  //标签总数
		}
	}

	for (size_t i = 1; i < connection.size(); i++) {
		connection[i] = mapping[connection[i]];
	}

	for (int i = 0; i < labelnum; i++) {
		labelcnt.push_back(0);
	}

	for (int y = 0; y < dstImage.rows; y++) {  //全变为最小值
		for (int x = 0; x < dstImage.cols; x++) {
			int label = labImg.at<int>(y, x);
			labImg.at<int>(y, x) = connection[label];
			labelcnt[connection[label]]++;
		}
	}
	for (int i = 1; i < labelcnt.size(); i++) {
		if (labelcnt[i] > weaktextnum) {
			weaklabel.push_back(i);
		}
	}
}


void texture(InputFiles& inputFiles, GlobalState* gs) {
    clock_t startTime, endTime;
    startTime = clock();
    string prename = "./" + inputFiles.images_folder;
    string imgname = inputFiles.img_filenames[0];
    string allname = prename + imgname;
    std::cout << "Name:" << endl;
    std::cout << allname << endl << endl;
    std::cout << "Start Weak Texture Detection" << endl;

    Mat srcImage = imread(allname, 0);
    Mat down_2;
    pyrDown(srcImage, down_2, Size(srcImage.cols / 2, srcImage.rows / 2));
    Mat down_4;
    pyrDown(down_2, down_4, Size(down_2.cols / 2, down_2.rows / 2));

    Mat dstImage = roberts(down_4);
    
    cv::threshold(dstImage, dstImage, Robthr, 255, cv::THRESH_BINARY);
    //imwrite("./Canny/Rob.jpg", dstImage);

    cv::Mat labImg0(dstImage.rows, dstImage.cols, CV_32S);
    std::vector<int> labelcnt0;
    std::vector<int> weaklabel0;
    Connect(dstImage, labImg0, labelcnt0, weaklabel0);

    for (int k = 0; k < weaklabel0.size(); k++) {
        int weakindex = weaklabel0[k];
        cv::Mat img_weak(dstImage.rows, dstImage.cols, CV_8UC3);
        for (int y = 0; y < img_weak.rows; y++) {
            for (int x = 0; x < img_weak.cols; x++) {
                img_weak.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                int label = labImg0.at<int>(y, x);
                if (label != weakindex) {
                    if (x > 0) {
                        int neibel = labImg0.at<int>(y, x - 1);
                        if (neibel == weakindex)
                            img_weak.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                    }
                    if (x < img_weak.cols - 1) {
                        int neibel = labImg0.at<int>(y, x + 1);
                        if (neibel == weakindex)
                            img_weak.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                    }
                    if (y > 0) {
                        int neibel = labImg0.at<int>(y - 1, x);
                        if (neibel == weakindex)
                            img_weak.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                    }
                    if (y < img_weak.rows - 1) {
                        int neibel = labImg0.at<int>(y + 1, x);
                        if (neibel == weakindex)
                            img_weak.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                    }
                }
            }
        }
        Mat grayImage;
        cvtColor(img_weak, grayImage, COLOR_BGR2GRAY);
        vector<Vec4i> lines;
        HoughLinesP(grayImage, lines, 1, CV_PI / 180, Houthr, minLineLength,
                    maxLineGap);
        for (size_t i = 0; i < lines.size(); i++) {
            Vec4i I = lines[i];
            double x1 = I[0];
            double y1 = I[1];
            double x2 = I[2];
            double y2 = I[3];
            line(dstImage, Point2d(x1, y1), Point2d(x2, y2), Scalar(255, 255, 255), 1);
        }
    }

    cv::Mat labImg(dstImage.rows, dstImage.cols, CV_32S);
    std::vector<int> labelcnt;
    std::vector<int> weaklabel;

    for (int y = 0; y < dstImage.rows; y++) {
		if (dstImage.data[y * dstImage.cols + 1] == 0)
			dstImage.data[y * dstImage.cols] = 0;
		if (dstImage.data[y * dstImage.cols + dstImage.cols - 2] == 0)
			dstImage.data[y * dstImage.cols + dstImage.cols - 1] = 0;
	}
	for (int x = 0; x < dstImage.cols; x++) {
		if (dstImage.data[1 * dstImage.cols + x] == 0)
			dstImage.data[0 * dstImage.cols + x] = 0;
		if (dstImage.data[(dstImage.rows - 2) * dstImage.cols + x] == 0)
			dstImage.data[(dstImage.rows - 1) * dstImage.cols + x] = 0;
	}

    Connect(dstImage, labImg, labelcnt, weaklabel);

    int labelnum = labelcnt.size();
    std::vector<cv::Vec3b> colors(labelnum);
    colors[0] = cv::Vec3b(0, 0, 0);
    int labelcntmax = 0;
    for (int i = 1; i < labelnum; i++) {
        labelcntmax = max(labelcntmax, labelcnt[i]);
        if (labelcnt[i] > 50) {
            colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
        } else {
            colors[i] = cv::Vec3b(0, 0, 0);
        }
    }

    cv::Mat img_connect(dstImage.rows, dstImage.cols, CV_8UC3);
    for (int y = 0; y < img_connect.rows; y++) {
        for (int x = 0; x < img_connect.cols; x++) {
            img_connect.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            int label = labImg.at<int>(y, x);
            img_connect.at<cv::Vec3b>(y, x) = colors[label];
        }
    }
    string connectcname = inputFiles.connect_folder;
    connectcname.append(imgname);
    cv::imwrite(connectcname, img_connect);

    std::vector<int> xmax;
    std::vector<int> xmin;
    std::vector<int> ymax;
    std::vector<int> ymin;
    for (int i = 0; i < weaklabel.size(); i++) {
        xmax.push_back(0);
        xmin.push_back(dstImage.cols - 1);
        ymax.push_back(0);
        ymin.push_back(dstImage.rows - 1);
    }


    std::vector<int> labelx(labelnum);
    std::vector<int> labely(labelnum);
    for (int y = 0; y < dstImage.rows; y++) {  //全变为最小值
        for (int x = 0; x < dstImage.cols; x++) {
            int label = labImg.at<int>(y, x);
            labelx[label] += x;
            labely[label] += y;
            if (labelcnt[label] > weaktextnum) {
                for (int i = 0; i < weaklabel.size(); i++) {
                    if (label == weaklabel[i]) {
                        xmax[i] = max(xmax[i], x);
                        ymax[i] = max(ymax[i], y);
                        xmin[i] = min(xmin[i], x);
                        ymin[i] = min(ymin[i], y);
                    }
                }
            }
        }
    }

    cout << "Trueweak index:" << endl;
    std::vector<int> trueweak;
    std::vector<int> maxsize;
    for (int i = 0; i < weaklabel.size(); i++) {
        int xsize = xmax[i] - xmin[i];
        int ysize = ymax[i] - ymin[i];
        int msize = max(xsize, ysize);
        int xysize = xsize * ysize;

        //cout << xmax[i] << " " << xmin[i] << endl;
        //cout << ymax[i] << " " << ymin[i] << endl;

        int label = weaklabel[i];
        if (xysize < sizerat * labelcnt[label] || labelcnt[label] > 100000) {
            trueweak.push_back(label);
            maxsize.push_back(msize);
            //cout << label << " " << endl;
            cout << label << " ";
        }
    }
    cout << endl;
    cv::Mat img_filter(dstImage.rows, dstImage.cols, CV_8UC3);
    for (int y = 0; y < img_filter.rows; y++) {
        for (int x = 0; x < img_filter.cols; x++) {
            img_filter.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            int label = labImg.at<int>(y, x);
            for (int i = 0; i < trueweak.size(); i++) {
                if (label == trueweak[i]) {
                    img_filter.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                }
            }
        }
    }

    string weakname = inputFiles.weak_folder;
    weakname.append(imgname);
    cv::imwrite(weakname, img_filter);
    //imwrite("./Canny/Weak.jpg", img_filter);

    cout << "Allocation Begin" << endl;
    {
        gs->lines->n = srcImage.cols * srcImage.rows;  // srcImage.cols * srcImage.rows
        gs->lines->resize(srcImage.cols * srcImage.rows);
    }
    cout << "Allocation Over" << endl;

    for (int y = 0; y < srcImage.rows; y++) {
        for (int x = 0; x < srcImage.cols; x++) {
            int pindex = y * srcImage.cols + x;
            int sx = x / 4;
            int sy = y / 4;
            if (sx >= dstImage.cols) sx--;
            if (sy >= dstImage.rows) sy--;
            gs->lines->canny[pindex] = labImg.at<int>(sy, sx);
        }
    }

    {
        gs->cannylines->n = labelnum;
        gs->cannylines->Cannyresize(labelnum);
    }

    gs->cannylines->text[0] = 1;
    for (int tind = 1; tind < labelnum; tind++) {
        labelx[tind] *= 4;
        labely[tind] *= 4;
        labelx[tind] /= labelcnt[tind];
        labely[tind] /= labelcnt[tind];

        gs->cannylines->cenxi[tind] = labelx[tind];
        gs->cannylines->cenyi[tind] = labely[tind];

        gs->cannylines->text[tind] = 1;

        for (int i = 0; i < trueweak.size(); i++) {
            if (tind == trueweak[i]) {
                gs->cannylines->text[tind] = -1;
                gs->cannylines->size[tind] = maxsize[i];
            }
        }
    }
    endTime = clock();
    std::cout << "Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl << endl;
}

void gslic(InputFiles& inputFiles, GlobalState* gs) {
    std::cout << "Start SuperPixel Generating" << endl;
    
    clock_t startTime, endTime;
    startTime = clock();

    string prename = "./" + inputFiles.images_folder;
    string imgname = inputFiles.img_filenames[0];
    string allname = prename + imgname;

    gSLICr::objects::settings my_settings;
	my_settings.no_segs = 4256;
	my_settings.spixel_size = 20;
	my_settings.coh_weight = 5.0f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::CIELAB; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = false; // whether or not run the enforce connectivity step

    Mat_<Vec3b> img_color;

    Mat_<Vec3b> Color_1 = imread(allname, IMREAD_COLOR);
    Mat_<Vec3b> Color_2;
    pyrDown(Color_1, Color_2, Size(Color_1.cols / 2, Color_1.rows / 2));
    pyrDown(Color_2, img_color, Size(Color_2.cols / 2, Color_2.rows / 2));

    gs->col = Color_1.cols;
    gs->row = Color_1.rows;

    my_settings.img_size.x = img_color.cols;
    my_settings.img_size.y = img_color.rows;

    //my_settings.no_segs = img_color.cols * img_color.rows / 36;

    // instantiate a core_engine
    gSLICr::engines::core_engine* gSLICr_engine =
        new gSLICr::engines::core_engine(my_settings);

    // gSLICr takes gSLICr::UChar4Image as input and out put
    gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

    load_image(img_color, in_img);

    Size s(img_color.cols, img_color.rows);
    Mat boundry_draw_frame;
    boundry_draw_frame.create(s, CV_8UC3);

    gSLICr_engine->Process_Frame(in_img, gs);

    endTime = clock();
    std::cout << "Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl << endl;

    gSLICr_engine->Draw_Segmentation_Result(out_img);

    load_image(out_img, boundry_draw_frame);
    // imshow("segmentation", boundry_draw_frame);
    string slicname = inputFiles.slic_folder;
    string resname = slicname + imgname;
    cv::imwrite(resname, boundry_draw_frame);
    
    delete gSLICr_engine;
    delete in_img;
    delete out_img;
}

static void print_help(char** argv) {
    printf("\nUsage: %s <im1> <im2> ... [--parameter=<parameter>]\n", argv[0]);
}

static void get_directory_entries(const char* dirname,
                                  vector<string>& directory_entries) {
    DIR* dir;
    struct dirent* ent;

    // Open directory stream
    dir = opendir(dirname);
    if (dir != NULL) {
        // std::cout << "Dirname is " << dirname << endl;
        // std::cout << "Dirname type is " << ent->d_type << endl;
        // std::cout << "Dirname type DT_DIR " << DT_DIR << endl;

        // Print all files and directories within the directory
        while ((ent = readdir(dir)) != NULL) {
            // std::cout << "INSIDE" << endl;
            // if(ent->d_type == DT_DIR)
            {
                char* name = ent->d_name;
                if (strcmp(name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                    continue;
                // printf ("dir %s/\n", name);
                directory_entries.push_back(string(name));
            }
        }

        closedir(dir);

    } else {
        // Could not open directory
        printf("Cannot open directory %s\n", dirname);
        exit(EXIT_FAILURE);
    }
    sort(directory_entries.begin(), directory_entries.end());
}

/* process command line arguments
 * Input: argc, argv - command line arguments
 * Output: inputFiles, outputFiles, parameters, gt_parameters, - algorithm
 * parameters
 */
static int getParametersFromCommandLine(int argc, char** argv,
                                        InputFiles& inputFiles,
                                        OutputFiles& outputFiles,
                                        AlgorithmParameters& algParams,
                                        GTcheckParameters& gt_parameters) {
    int camera_idx = 0;
    const char* algorithm_opt = "--algorithm=";
    const char* maxdisp_opt = "--max-disparity=";
    const char* blocksize_opt = "--blocksize=";
    const char* cost_tau_color_opt = "--cost_tau_color=";
    const char* cost_tau_gradient_opt = "--cost_tau_gradient=";
    const char* cost_alpha_opt = "--cost_alpha=";
    const char* cost_gamma_opt = "--cost_gamma=";
    const char* disparity_tolerance_opt = "--disp_tol=";
    const char* normal_tolerance_opt = "--norm_tol=";
    const char* border_value = "--border_value=";
    const char* gtDepth_divFactor_opt = "--gtDepth_divisionFactor=";
    const char* gtDepth_tolerance_opt = "--gtDepth_tolerance=";
    const char* gtDepth_tolerance2_opt = "--gtDepth_tolerance2=";
    const char* colorProc_opt = "-color_processing";
    const char* num_iterations_opt = "--iterations=";
    const char* self_similariy_n_opt = "--ss_n=";
    const char* ct_epsilon_opt = "--ct_eps=";
    const char* cam_scale_opt = "--cam_scale=";
    const char* num_img_processed_opt = "--num_img_processed=";
    const char* n_best_opt = "--n_best=";
    const char* cost_comb_opt = "--cost_comb=";
    const char* cost_good_factor_opt = "--good_factor=";
    const char* depth_min_opt = "--depth_min=";
    const char* depth_max_opt = "--depth_max=";
    //    const char* scale_opt         = "--scale=";
    const char* outputPath_opt = "-output_folder";
    const char* calib_opt = "-calib_file";
    const char* gt_opt = "-gt";
    const char* gt_nocc_opt = "-gt_nocc";
    const char* occl_mask_opt = "-occl_mask";
    const char* gt_normal_opt = "-gt_normal";
    const char* images_input_folder_opt = "-images_folder";
    const char* mslp_input_folder_opt = "-mslp_folder";
    const char* p_input_folder_opt = "-p_folder";
    const char* krt_file_opt = "-krt_file";
    const char* camera_input_folder_opt = "-camera_folder";
    const char* bounding_folder_opt = "-bounding_folder";
    const char* viewSelection_opt = "-view_selection";
    const char* initial_seed_opt = "--initial_seed";
    const char* min_angle_opt = "--min_angle=";
    const char* max_angle_opt = "--max_angle=";
    const char* no_texture_sim_opt = "--no_texture_sim";
    const char* no_texture_per_opt = "--no_texture_per";
    const char* max_views_opt = "--max_views=";
    const char* pmvs_folder_opt = "--pmvs_folder";
    const char* camera_idx_opt = "--camera_idx=";

    // read in arguments
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            inputFiles.img_filenames.push_back(argv[i]);
        } else if (strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) ==
                   0) {
            char* _alg = argv[i] + strlen(algorithm_opt);
            algParams.algorithm =
                strcmp(_alg, "pm") == 0        ? PM_COST
                : strcmp(_alg, "ct") == 0      ? CENSUS_TRANSFORM
                : strcmp(_alg, "sct") == 0     ? SPARSE_CENSUS
                : strcmp(_alg, "ct_ss") == 0   ? CENSUS_SELFSIMILARITY
                : strcmp(_alg, "adct") == 0    ? ADCENSUS
                : strcmp(_alg, "adct_ss") == 0 ? ADCENSUS_SELFSIMILARITY
                : strcmp(_alg, "pm_ss") == 0   ? PM_SELFSIMILARITY
                                               : -1;
            if (algParams.algorithm < 0) {
                printf(
                    "Command-line parameter error: Unknown stereo "
                    "algorithm\n\n");
                print_help(argv);
                return -1;
            }
        } else if (strncmp(argv[i], cost_comb_opt, strlen(cost_comb_opt)) ==
                   0) {
            char* _alg = argv[i] + strlen(algorithm_opt);
            algParams.cost_comb = strcmp(_alg, "all") == 0      ? COMB_ALL
                                  : strcmp(_alg, "best_n") == 0 ? COMB_BEST_N
                                  : strcmp(_alg, "angle") == 0  ? COMB_ANGLE
                                  : strcmp(_alg, "good") == 0   ? COMB_GOOD
                                                                : -1;
            if (algParams.cost_comb < 0) {
                printf(
                    "Command-line parameter error: Unknown cost combination "
                    "method\n\n");
                print_help(argv);
                return -1;
            }
        } else if (strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0) {
            if (sscanf(argv[i] + strlen(maxdisp_opt), "%f",
                       &algParams.max_disparity) != 1 ||
                algParams.max_disparity < 1) {
                printf(
                    "Command-line parameter error: The max disparity "
                    "(--maxdisparity=<...>) must be a positive integer \n");
                print_help(argv);
                return -1;
            }
        } else if (strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) ==
                   0) {
            int k_size;
            if (sscanf(argv[i] + strlen(blocksize_opt), "%d", &k_size) != 1 ||
                k_size < 1 || k_size % 2 != 1) {
                printf(
                    "Command-line parameter error: The block size "
                    "(--blocksize=<...>) must be a positive odd number\n");
                return -1;
            }
            algParams.box_hsize = k_size;
            algParams.box_vsize = k_size;
        } else if (strncmp(argv[i], cost_good_factor_opt,
                           strlen(cost_good_factor_opt)) == 0) {
            sscanf(argv[i] + strlen(cost_good_factor_opt), "%f",
                   &algParams.good_factor);
        } else if (strncmp(argv[i], cost_tau_color_opt,
                           strlen(cost_tau_color_opt)) == 0) {
            sscanf(argv[i] + strlen(cost_tau_color_opt), "%f",
                   &algParams.tau_color);
        } else if (strncmp(argv[i], cost_tau_gradient_opt,
                           strlen(cost_tau_gradient_opt)) == 0) {
            sscanf(argv[i] + strlen(cost_tau_gradient_opt), "%f",
                   &algParams.tau_gradient);
        } else if (strncmp(argv[i], cost_alpha_opt, strlen(cost_alpha_opt)) ==
                   0) {
            sscanf(argv[i] + strlen(cost_alpha_opt), "%f", &algParams.alpha);
        } else if (strncmp(argv[i], cost_gamma_opt, strlen(cost_gamma_opt)) ==
                   0) {
            sscanf(argv[i] + strlen(cost_gamma_opt), "%f", &algParams.gamma);
        } else if (strncmp(argv[i], border_value, strlen(border_value)) == 0) {
            sscanf(argv[i] + strlen(border_value), "%d",
                   &algParams.border_value);
        } else if (strncmp(argv[i], num_iterations_opt,
                           strlen(num_iterations_opt)) == 0) {
            sscanf(argv[i] + strlen(num_iterations_opt), "%d",
                   &algParams.iterations);
        } else if (strncmp(argv[i], disparity_tolerance_opt,
                           strlen(disparity_tolerance_opt)) == 0) {
            sscanf(argv[i] + strlen(disparity_tolerance_opt), "%f",
                   &algParams.dispTol);
        } else if (strncmp(argv[i], normal_tolerance_opt,
                           strlen(normal_tolerance_opt)) == 0) {
            sscanf(argv[i] + strlen(normal_tolerance_opt), "%f",
                   &algParams.normTol);
        } else if (strncmp(argv[i], self_similariy_n_opt,
                           strlen(self_similariy_n_opt)) == 0) {
            sscanf(argv[i] + strlen(self_similariy_n_opt), "%d",
                   &algParams.self_similarity_n);
        } else if (strncmp(argv[i], ct_epsilon_opt, strlen(ct_epsilon_opt)) ==
                   0) {
            sscanf(argv[i] + strlen(ct_epsilon_opt), "%f",
                   &algParams.census_epsilon);
        } else if (strncmp(argv[i], cam_scale_opt, strlen(cam_scale_opt)) ==
                   0) {
            sscanf(argv[i] + strlen(cam_scale_opt), "%f", &algParams.cam_scale);
        } else if (strncmp(argv[i], num_img_processed_opt,
                           strlen(num_img_processed_opt)) == 0) {
            sscanf(argv[i] + strlen(num_img_processed_opt), "%d",
                   &algParams.num_img_processed);
        } else if (strncmp(argv[i], n_best_opt, strlen(n_best_opt)) == 0) {
            sscanf(argv[i] + strlen(n_best_opt), "%d", &algParams.n_best);
        } else if (strncmp(argv[i], gtDepth_divFactor_opt,
                           strlen(gtDepth_divFactor_opt)) == 0) {
            sscanf(argv[i] + strlen(gtDepth_divFactor_opt), "%f",
                   &gt_parameters.divFactor);
        } else if (strncmp(argv[i], gtDepth_tolerance_opt,
                           strlen(gtDepth_tolerance_opt)) == 0) {
            sscanf(argv[i] + strlen(gtDepth_tolerance_opt), "%f",
                   &gt_parameters.dispTolGT);
        } else if (strncmp(argv[i], gtDepth_tolerance2_opt,
                           strlen(gtDepth_tolerance2_opt)) == 0) {
            sscanf(argv[i] + strlen(gtDepth_tolerance2_opt), "%f",
                   &gt_parameters.dispTolGT2);
        } else if (strncmp(argv[i], depth_min_opt, strlen(depth_min_opt)) ==
                   0) {
            sscanf(argv[i] + strlen(depth_min_opt), "%f", &algParams.depthMin);
        } else if (strncmp(argv[i], depth_max_opt, strlen(depth_max_opt)) ==
                   0) {
            sscanf(argv[i] + strlen(depth_max_opt), "%f", &algParams.depthMax);
        } else if (strncmp(argv[i], min_angle_opt, strlen(min_angle_opt)) == 0)
            sscanf(argv[i] + strlen(min_angle_opt), "%f", &algParams.min_angle);
        else if (strncmp(argv[i], max_angle_opt, strlen(max_angle_opt)) == 0) {
            sscanf(argv[i] + strlen(max_angle_opt), "%f", &algParams.max_angle);
        } else if (strncmp(argv[i], pmvs_folder_opt, strlen(pmvs_folder_opt)) ==
                   0) {
            inputFiles.pmvs_folder = argv[++i];
        } else if (strncmp(argv[i], max_views_opt, strlen(max_views_opt)) == 0)
            sscanf(argv[i] + strlen(max_views_opt), "%u", &algParams.max_views);
        else if (strncmp(argv[i], no_texture_sim_opt,
                         strlen(no_texture_sim_opt)) == 0)
            sscanf(argv[i] + strlen(no_texture_sim_opt), "%f",
                   &algParams.no_texture_sim);
        else if (strncmp(argv[i], no_texture_per_opt,
                         strlen(no_texture_per_opt)) == 0)
            sscanf(argv[i] + strlen(no_texture_per_opt), "%f",
                   &algParams.no_texture_per);
        else if (strcmp(argv[i], viewSelection_opt) == 0)
            algParams.viewSelection = true;
        else if (strcmp(argv[i], colorProc_opt) == 0)
            algParams.color_processing = true;
        else if (strcmp(argv[i], "-o") == 0)
            outputFiles.disparity_filename = argv[++i];
        else if (strcmp(argv[i], outputPath_opt) == 0)
            outputFiles.parentFolder = argv[++i];
        else if (strcmp(argv[i], calib_opt) == 0)
            inputFiles.calib_filename = argv[++i];
        else if (strcmp(argv[i], gt_opt) == 0)
            inputFiles.gt_filename = argv[++i];
        else if (strcmp(argv[i], gt_nocc_opt) == 0)
            inputFiles.gt_nocc_filename = argv[++i];
        else if (strcmp(argv[i], occl_mask_opt) == 0)
            inputFiles.occ_filename = argv[++i];
        else if (strcmp(argv[i], gt_normal_opt) == 0)
            inputFiles.gt_normal_filename = argv[++i];
        else if (strcmp(argv[i], images_input_folder_opt) == 0)
            inputFiles.images_folder = argv[++i];
        else if (strcmp(argv[i], mslp_input_folder_opt) == 0)
            inputFiles.mslp_folder = argv[++i];
        else if (strcmp(argv[i], p_input_folder_opt) == 0)
            inputFiles.p_folder = argv[++i];
        else if (strcmp(argv[i], krt_file_opt) == 0)
            inputFiles.krt_file = argv[++i];
        else if (strcmp(argv[i], camera_input_folder_opt) == 0)
            inputFiles.camera_folder = argv[++i];
        else if (strcmp(argv[i], initial_seed_opt) == 0)
            inputFiles.seed_file = argv[++i];
        else if (strcmp(argv[i], bounding_folder_opt) == 0)
            inputFiles.bounding_folder = argv[++i];
        else if (strncmp(argv[i], camera_idx_opt, strlen(camera_idx_opt)) ==
                 0) {
            sscanf(argv[i] + strlen(camera_idx_opt), "%d", &camera_idx);
        } else {
            printf("Command-line parameter warning: unknown option %s\n",
                   argv[i]);
            // return -1;
        }
    }
    // std::cout << "Seed file is " << inputFiles.seed_file  << endl;
    // std::cout << "Min angle is " << algParams.min_angle  << endl;
    if (inputFiles.pmvs_folder.size() > 0) {
        std::cout << "Using pmvs information inside directory "
             << inputFiles.pmvs_folder << endl;
        inputFiles.images_folder = inputFiles.pmvs_folder + "/visualize/";

        inputFiles.img_filenames.clear();
        get_directory_entries(inputFiles.images_folder.c_str(),
                              inputFiles.img_filenames);

        inputFiles.p_folder = inputFiles.pmvs_folder + "/txt/";

        std::cout << "Using image " << inputFiles.img_filenames[camera_idx]
             << " as reference camera" << endl;
        std::swap(inputFiles.img_filenames[0],
                  inputFiles.img_filenames[camera_idx]);
    }
    std::cout << "Input files are: ";
    for (const auto i : inputFiles.img_filenames) std::cout << i << " ";
    std::cout << endl;


     {
        std::stringstream path;
        path << inputFiles.mslp_folder << "Canny/";
        inputFiles.canny_folder = path.str();
        _mkdir(inputFiles.canny_folder.data());
    }
    {
        std::stringstream path;
        path << inputFiles.mslp_folder << "Canny/Connect/";
        inputFiles.connect_folder = path.str();
        _mkdir(inputFiles.connect_folder.data());
    }
    {
        std::stringstream path;
        path << inputFiles.mslp_folder << "Canny/Disp/";
        inputFiles.disp_folder = path.str();
        _mkdir(inputFiles.disp_folder.data());
    }
    {
        std::stringstream path;
        path << inputFiles.mslp_folder << "Canny/Normal/";
        inputFiles.normal_folder = path.str();
        _mkdir(inputFiles.normal_folder.data());
    }
    {
        std::stringstream path;
        path << inputFiles.mslp_folder << "Canny/SLIC/";
        inputFiles.slic_folder = path.str();
        _mkdir(inputFiles.slic_folder.data());
    }
    {
        std::stringstream path;
        path << inputFiles.mslp_folder << "Canny/WEAK/";
        inputFiles.weak_folder = path.str();
        _mkdir(inputFiles.weak_folder.data());
    }


    return 0;
}

static void selectViews(CameraParameters& cameraParams, int imgWidth,
                        int imgHeight, AlgorithmParameters& algParams) {
    vector<Camera>& cameras = cameraParams.cameras;
    Camera ref = cameras[cameraParams.idRef];

    int x = imgWidth / 2;
    int y = imgHeight / 2;

    cameraParams.viewSelectionSubset.clear();

    Vec3f viewVectorRef = getViewVector(ref, x, y);

    // TODO hardcoded value makes it a parameter
    float minimum_angle_degree = algParams.min_angle;
    float maximum_angle_degree = algParams.max_angle;

    unsigned int maximum_view = algParams.max_views;
    float minimum_angle_radians = minimum_angle_degree * M_PI / 180.0f;
    float maximum_angle_radians = maximum_angle_degree * M_PI / 180.0f;
    float min_depth = 9999;
    float max_depth = 0;
    //if (algParams.viewSelection)
    //    printf(
    //        "Accepting intersection angle of central rays from %f to %f "
    //        "degrees, use --min_angle=<angle> and --max_angle=<angle> to "
    //        "modify them\n",
    //        minimum_angle_degree, maximum_angle_degree);
    for (size_t i = 1; i < cameras.size(); i++) {
        // if ( !algParams.viewSelection ) { //select all views, dont perform
        // selection cameraParams.viewSelectionSubset.push_back ( i ); continue;
        //}
        Vec3f vec = getViewVector(cameras[i], x, y);

        float baseline = static_cast<float>(norm(cameras[0].C, cameras[i].C));
        float angle = getAngle(viewVectorRef, vec);
        //         if ( angle > minimum_angle_radians &&
        //              angle < maximum_angle_radians ) //0.6 select if angle
        //              between 5.7 and 34.8 (0.6) degrees (10 and 30 degrees
        //              suggested by some paper)
        //         {
        if (true) {
            if (algParams.viewSelection) {
                cameraParams.viewSelectionSubset.push_back(static_cast<int>(i));
                // printf("\taccepting camera %ld with angle\t %f degree (%f
                // radians) and baseline %f\n", i, angle*180.0f/M_PI, angle,
                // baseline);
            }
            float min_range =
                (baseline / 2.0f) / sin(maximum_angle_radians / 2.0f);
            float max_range =
                (baseline / 2.0f) / sin(minimum_angle_radians / 2.0f);
            min_depth = std::min(min_range, min_depth);
            max_depth = std::max(max_range, max_depth);
            // printf("Min max ranges are %f %f\n", min_range, max_range);
            // printf("Min max depth are %f %f\n", min_depth, max_depth);
        }
        // else
        // printf("Discarding camera %ld with angle\t %f degree (%f radians) and
        // baseline, %f\n", i, angle*180.0f/M_PI, angle, baseline);
    }

    if (algParams.depthMin == -1) algParams.depthMin = min_depth;
    if (algParams.depthMax == -1) algParams.depthMax = max_depth;

    if (!algParams.viewSelection) {
        cameraParams.viewSelectionSubset.clear();
        for (size_t i = 1; i < cameras.size(); i++)
            cameraParams.viewSelectionSubset.push_back(static_cast<int>(i));
        return;
    }
    if (cameraParams.viewSelectionSubset.size() >= maximum_view) {
        printf(
            "Too many camera, randomly selecting only %d of them (modify with "
            "--max_views=<number>)\n",
            maximum_view);
        std::srand(unsigned(std::time(0)));
        std::random_shuffle(
            cameraParams.viewSelectionSubset.begin(),
            cameraParams.viewSelectionSubset.end());  // shuffle elements of v
        cameraParams.viewSelectionSubset.erase(
            cameraParams.viewSelectionSubset.begin() + maximum_view,
            cameraParams.viewSelectionSubset.end());
    }
    // for (auto i : cameraParams.viewSelectionSubset )
    // printf("\taccepting camera %d\n", i);
}

static void delTexture(int num, cudaTextureObject_t texs[],
                       cudaArray* cuArray[]) {
    for (int i = 0; i < num; i++) {
        cudaFreeArray(cuArray[i]);
        cudaDestroyTextureObject(texs[i]);
    }
}

static void addImageToTextureUint(vector<Mat_<uint8_t> >& imgs,
                                  cudaTextureObject_t texs[],
                                  cudaArray* cuArray[]) {
    for (size_t i = 0; i < imgs.size(); i++) {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with uint8_t point type
        cudaChannelFormatDesc channelDesc =
            // cudaCreateChannelDesc (8,
            // 0,
            // 0,
            // 0,
            // cudaChannelFormatKindUnsigned);
            cudaCreateChannelDesc<char>();
        // Allocate array with correct size and number of channels
        checkCudaErrors(cudaMallocArray(&cuArray[i], &channelDesc, cols, rows));

        checkCudaErrors(cudaMemcpy2DToArray(
            cuArray[i], 0, 0, imgs[i].ptr<uint8_t>(), imgs[i].step[0],
            cols * sizeof(uint8_t), rows, cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        // cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(
            cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        // texs[i] = texObj;
    }
    return;
}
static void addImageToTextureFloatColor(vector<Mat>& imgs,
                                        cudaTextureObject_t texs[],
                                        cudaArray* cuArray[]) {
    for (size_t i = 0; i < imgs.size(); i++) {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Allocate array with correct size and number of channels
        // cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray[i], &channelDesc, cols, rows));

        checkCudaErrors(cudaMemcpy2DToArray(
            cuArray[i], 0, 0, imgs[i].ptr<float>(), imgs[i].step[0],
            cols * sizeof(float) * 4, rows, cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        // cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(
            cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
    }
    return;
}

static void addImageToTextureFloatGray(vector<Mat>& imgs,
                                       cudaTextureObject_t texs[],
                                       cudaArray* cuArray[]) {
    for (size_t i = 0; i < imgs.size(); i++) {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        checkCudaErrors(cudaMallocArray(&cuArray[i], &channelDesc, cols, rows));

        checkCudaErrors(cudaMemcpy2DToArray(
            cuArray[i], 0, 0, imgs[i].ptr<float>(), imgs[i].step[0],
            cols * sizeof(float), rows, cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        // cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(
            cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        // texs[i] = texObj;
    }
    return;
}

static void selectCudaDevice() {
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "There is no cuda capable device!\n");
        exit(EXIT_FAILURE);
    }
    std::cout << "Detected " << deviceCount << " devices!" << endl;
    std::vector<int> usableDevices;
    std::vector<std::string> usableDeviceNames;
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 3 && prop.minor >= 0) {
                usableDevices.push_back(i);
                usableDeviceNames.push_back(string(prop.name));
            } else {
                std::cout << "CUDA capable device " << string(prop.name)
                     << " is only compute cabability " << prop.major << '.'
                     << prop.minor << endl;
            }
        } else {
            std::cout << "Could not check device properties for one of the cuda "
                    "devices!"
                 << endl;
        }
    }
    if (usableDevices.empty()) {
        fprintf(stderr, "There is no cuda device supporting gipuma!\n");
        exit(EXIT_FAILURE);
    }
    std::cout << "Detected gipuma compatible device: " << usableDeviceNames[0]
         << endl;
    ;
    checkCudaErrors(cudaSetDevice(usableDevices[0]));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 128);
}

static int runGipuma(InputFiles& inputFiles, OutputFiles& outputFiles,
                     AlgorithmParameters& algParams,
                     GTcheckParameters& gtParameters, Results& results,
                     GlobalState* gs) {
    

            // create folder to store result images
    time_t timeObj;
    time ( &timeObj );
    tm *pTime = localtime ( &timeObj );

    //#if defined(_WIN32)
//    _mkdir ( outputFiles.parentFolder );
//#else
//    mkdir ( outputFiles.parentFolder, 0777 );
//#endif
//    char outputFolder[256];
//    if(inputFiles.img_filenames.empty())
//    {
//      throw std::runtime_error("There was a problem finding the input files!");
//    }
//    string ref_name = inputFiles.img_filenames[0].substr ( 0, inputFiles.img_filenames[0].length() - 4 );
//    sprintf ( outputFolder, "%s/%04d%02d%02d_%02d%02d%02d_%s", outputFiles.parentFolder, pTime->tm_year + 1900, pTime->tm_mon + 1, pTime->tm_mday, pTime->tm_hour, pTime->tm_min, pTime->tm_sec, ref_name.c_str () );
//#if defined(_WIN32)
//    _mkdir ( outputFolder );
//#else
//    mkdir ( outputFolder, 0777 );
//#endif

    size_t numImages = inputFiles.img_filenames.size();

    vector<Mat_<Vec3b> > img_color(numImages);
    vector<Mat_<uint8_t> > img_grayscale(numImages);
    for (size_t i = 0; i < numImages; i++) {
        img_grayscale[i] = imread((inputFiles.images_folder + inputFiles.img_filenames[i]), IMREAD_GRAYSCALE);
        if (algParams.color_processing) {
            img_color[i] = imread((inputFiles.images_folder + inputFiles.img_filenames[i]), IMREAD_COLOR);
        }
    }

    uint32_t rows = img_grayscale[0].rows;
    uint32_t cols = img_grayscale[0].cols;
    uint32_t numPixels = rows * cols;

    size_t avail;
    size_t used;
    size_t total;

    CameraParameters cameraParams = getCameraParameters(algParams, *(gs->cameras), inputFiles, algParams.cam_scale, true);

    cout << "algParams.depthMin: " << algParams.depthMin << endl;
    cout << "algParams.depthMax: " << algParams.depthMax << endl;

        Mat testImg_display;
    {
        int sizeN = cols / 8;
        float halfSize = (float)sizeN / 2.0f;
        Mat_<Vec3f> normalTestImg = Mat::zeros(sizeN, sizeN, CV_32FC3);
        for (int i = 0; i < sizeN; i++) {
            for (int j = 0; j < sizeN; j++) {
                float y = (float)i / halfSize - 1.0f;
                float x = (float)j / halfSize - 1.0f;
                float xy = pow(x, 2) + pow(y, 2);
                if (xy <= 1.0f) {
                    float z = sqrt(1.0f - xy);
                    normalTestImg(sizeN - 1 - i, sizeN - 1 - j) =
                        Vec3f(-x, -y, -z);
                }
            }
        }

        normalTestImg.convertTo(testImg_display, CV_16U, 32767, 32767);
        cvtColor(testImg_display, testImg_display, COLOR_RGB2BGR);
    }

    //selectViews(cameraParams, cols, rows, algParams);
    
    cameraParams.viewSelectionSubset.clear();

    string sel = inputFiles.img_filenames[0];
    string numsel = sel.substr(4, 8);
    int camera_id = atoi(numsel.c_str());

    ifstream images;
    string viewfile = inputFiles.mslp_folder;
    viewfile.append("pair.txt");
    std::cout << "viewfile: " << viewfile << endl; 
    images.open(viewfile, ifstream::in);
    string line;
    getline(images, line);
    for (int i = 0; i <= camera_id; i++) {
        getline(images, line);
        getline(images, line);
    }

    stringstream ss(line);

    int viewnum;
    ss >> viewnum;
    for (int j = 0; j < viewnum; ++j) {
        int image_index;
        float score;
        ss >> image_index >> score;
        if (image_index > camera_id) {
            cameraParams.viewSelectionSubset.push_back(image_index);
        } else {
            cameraParams.viewSelectionSubset.push_back(image_index + 1);
        }
    }

    std::cout << "camera_id: " << camera_id << endl;
    std::cout << "view: " << endl;
    size_t numSelViews = cameraParams.viewSelectionSubset.size();
    for (int i = 0; i < numSelViews; i++) {
        gs->cameras->viewSelectionSubset[i] = cameraParams.viewSelectionSubset[i];
        std::cout << cameraParams.viewSelectionSubset[i] << " ";
    }

    for (int i = 0; i < algParams.num_img_processed; i++) {
        cameraParams.cameras[i].depthMin = algParams.depthMin;
        cameraParams.cameras[i].depthMax = algParams.depthMax;

        gs->cameras->cameras[i].depthMin = algParams.depthMin;
        gs->cameras->cameras[i].depthMax = algParams.depthMax;

        algParams.min_disparity = disparityDepthConversion(
            cameraParams.f, cameraParams.cameras[i].baseline,
            cameraParams.cameras[i].depthMax);
        algParams.max_disparity = disparityDepthConversion(
            cameraParams.f, cameraParams.cameras[i].baseline,
            cameraParams.cameras[i].depthMin);
    }

    //std::cout << "cameraParams.cameras[i].depthMax:" << cameraParams.cameras[0].depthMax << endl;
    //std::cout << "cameraParams.cameras[i].depthMin:" << cameraParams.cameras[0].depthMin << endl;
    //std::cout << "algParams.min_disparity:" << algParams.min_disparity << endl;
    //std::cout << "algParams.max_disparity:" << algParams.max_disparity << endl;

    // run gpu run
    // Init parameters
    gs->params = &algParams;

    gs->cameras->viewSelectionSubsetNumber = static_cast<int>(numSelViews);

    // Init ImageInfo
    gs->cameras->cols = cols;
    gs->cameras->rows = rows;
    gs->params->cols = cols;
    gs->params->rows = rows;

    vector<Mat> img_grayscale_float(numImages);
    vector<Mat> img_color_float(numImages);
    vector<Mat> img_color_float_alpha(numImages);
    vector<Mat_<uint16_t> > img_grayscale_uint(numImages);
    for (size_t i = 0; i < numImages; i++) {
        img_grayscale[i].convertTo(img_grayscale_float[i],
                                   CV_32FC1);  // or CV_32F works (too)
        img_grayscale[i].convertTo(img_grayscale_uint[i],
                                   CV_16UC1);  // or CV_32F works (too)
        if (algParams.color_processing) {
            vector<Mat_<float> > rgbChannels(3);
            img_color_float_alpha[i] = Mat::zeros(
                img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC4);
            img_color[i].convertTo(img_color_float[i],
                                   CV_32FC3);  // or CV_32F works (too)
            Mat alpha(img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC1);

            split(img_color_float[i], rgbChannels);
            rgbChannels.push_back(alpha);
            merge(rgbChannels, img_color_float_alpha[i]);
        }
    }
    int64_t t = getTickCount();

    cudaMemGetInfo(&avail, &total);
    used = total - avail;

    if (algParams.color_processing)
        addImageToTextureFloatColor(img_color_float_alpha, gs->imgs,
                                    gs->cuArray);
    else
        addImageToTextureFloatGray(img_grayscale_float, gs->imgs, gs->cuArray);

    //cudaMemGetInfo(&avail, &total);
    //used = total - avail;
    //printf("Device memory used: %fMB\n", used / 1000000.0f);

    std::cout << endl;

    std::cout << "Begin reading input from baseline: " << endl;

    string imgname = inputFiles.img_filenames[0];
    string numind = imgname.substr(0, 8);
    std::stringstream result_path;
     result_path << inputFiles.mslp_folder << "APD" << "/" << numind;
    std::string result_folder = result_path.str();
    //std::cout << result_folder << endl;
    std::string depth_path = result_folder + "/depths_geom.dmb";
    std::string normal_path = result_folder + "/normals.dmb";
    cv::Mat_<float> ref_depth;
    cv::Mat_<cv::Vec3f> ref_normal;
    cv::Mat_<float> ref_cost;
    std::cout << "depth_path" << depth_path << endl;
    std::cout << "normal_path" << normal_path << endl;

    readDepthDmb(depth_path, ref_depth);
    readNormalDmb(normal_path, ref_normal);

    cout << "gs->row: " << gs->row << endl;
    cout << "gs->col: " << gs->col << endl;

    for (int y = 0; y < gs->row; y++) {
        for (int x = 0; x < gs->col; x++) {
            int center = y * gs->col + x;
            gs->lines->norm4[center].x = ref_normal(y, x)[0];
            gs->lines->norm4[center].y = ref_normal(y, x)[1];
            gs->lines->norm4[center].z = ref_normal(y, x)[2];
            gs->lines->c[center] = 1.0f;
            //gs->lines->depth[center] = ref_depth(y, x);
            float depth_ = ref_depth(y, x);
            gs->lines->depth[center] = disparityDepthConversion(cameraParams.f, cameraParams.cameras[0].baseline,depth_);
        }
    }

    std::cout << endl << "Start Upsampling Pixel PatchMatch" << endl;
    firstcuda(*gs);
    
    std::cout << endl << "Start Disparity Continuity Calculation" << endl;
    cout << "gs->row: " << gs->row << endl;
    cout << "gs->col: " << gs->col << endl;

    std::string reliable_path = result_folder + "/weak.png";
    Mat srcImage = imread(reliable_path, cv::IMREAD_COLOR);
    for (int y = 0; y < srcImage.rows; y++) {
        for (int x = 0; x < srcImage.cols; x++) {
            int pindex = y * srcImage.cols + x;
            if (srcImage.at<cv::Vec3b>(y, x) == cv::Vec3b(255, 255, 255)) {
                gs->lines->scale[pindex] = 1;
            }
            else if (srcImage.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 255, 0)) {
                gs->lines->scale[pindex] = 1;
            } 
            else if (srcImage.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 255)) {
                gs->lines->scale[pindex] = 1;
            }
        }
    }

    std::cout << endl << "Iteratively Start Superpixel RANSAC and Weighted Median Filtering" << endl;
    sliccuda(*gs);

    std::cout << endl << "Start Large Textureness Area RANSAC" << endl;
    if (true) {
        //剩下的好点，强纹理DBSCAN，弱纹理mean square
        // std::vector<std::vector<float>> D(gs->cannylines->n);
        std::vector<std::vector<int>> I(gs->cannylines->n);

        std::vector<int> Robert;

        for (int j = 0; j < gs->row; j++) {
            for (int i = 0; i < gs->col; i++) {
                int pindex = j * gs->col + i;
                int canny = gs->lines->canny[pindex];
                if (gs->lines->scale[pindex] == 1 &&
                    gs->cannylines->text[canny] == -1) {
                    I[canny].push_back(pindex);
                }
            }
        }

        for (int textindex = 0; textindex < gs->cannylines->n; textindex++) {
            if (gs->cannylines->text[textindex] == -1) {
                if (I[textindex].size() > 50000) {
                    unsigned seed = std::chrono::system_clock::now()
                                        .time_since_epoch()
                                        .count();
                    std::shuffle(I[textindex].begin(), I[textindex].end(),
                                 std::default_random_engine(seed));
                    while (I[textindex].size() >= 50000) {
                        I[textindex].pop_back();
                    }
                }

                float depth_abs =
                    0.0003 * sqrtf(gs->cannylines->size[textindex] / 20);

                float4 norm_this;
                norm_this.x = gs->cannylines->cenxi[textindex] - gs->col / 2;
                norm_this.y = gs->cannylines->cenyi[textindex] - gs->row / 2;
                norm_this.z = gs->cameras->cameras[REFERENCE].f;
                float sq = sqrtf(norm_this.x * norm_this.x +
                                 norm_this.y * norm_this.y +
                                 norm_this.z * norm_this.z);
                norm_this.x /= sq;
                norm_this.y /= sq;
                norm_this.z /= sq;

                std::vector<Point3f> ptSet;
                cout << "textindex: " << textindex << endl;
                std::cout << "cenxi: " << gs->cannylines->cenxi[textindex]
                          << "    cenyi: " << gs->cannylines->cenyi[textindex]
                          << endl;
                std::cout << "I[textindex]: " << I[textindex].size() << endl;
                for (int i = 0; i < I[textindex].size(); i++) {  //将好点放入矩阵计算中
                    int pindex = I[textindex][i];
                    // z
                    float disp = disparityDepthConversion(
                        cameraParams.f, cameraParams.cameras[0].baseline,
                        gs->lines->depth[pindex]);
                    float3 pt, ptX;
                    int por_x = pindex % gs->col;
                    int por_y = pindex / gs->col;
                    pt.x = disp * por_x - gs->cameras->cameras[0].P_col34.x;
                    pt.y = disp * por_y - gs->cameras->cameras[0].P_col34.y;
                    pt.z = disp - gs->cameras->cameras[0].P_col34.z;
                    ptX.x = gs->cameras->cameras[0].M_inv[0] * pt.x +
                            gs->cameras->cameras[0].M_inv[1] * pt.y +
                            gs->cameras->cameras[0].M_inv[2] * pt.z;
                    ptX.y = gs->cameras->cameras[0].M_inv[3] * pt.x +
                            gs->cameras->cameras[0].M_inv[4] * pt.y +
                            gs->cameras->cameras[0].M_inv[5] * pt.z;
                    ptX.z = gs->cameras->cameras[0].M_inv[6] * pt.x +
                            gs->cameras->cameras[0].M_inv[7] * pt.y +
                            gs->cameras->cameras[0].M_inv[8] * pt.z;
                    Point3f Point(ptX.x, ptX.y, ptX.z);
                    ptSet.push_back(Point);
                }

                double a = 0;
                double b = 0;
                double c = 1;
                double d = -1;
                double maximum = 0;
                vector<double> u;

                for (int k = 0; k < 10000; k++) {
                    int o = rand() % ptSet.size();
                    int p = rand() % ptSet.size();
                    int q = rand() % ptSet.size();

                    vector<Point3f> pt_sam;
                    pt_sam.push_back(ptSet[o]);
                    pt_sam.push_back(ptSet[p]);
                    pt_sam.push_back(ptSet[q]);

                    double ta, tb, tc, td;
                    calcLinePara(pt_sam, ta, tb, tc, td);
                    double sq = sqrt(ta * ta + tb * tb + tc * tc);
                    ta /= sq;
                    tb /= sq;
                    tc /= sq;
                    td /= sq;

                    double pixel_count = 0;
                    vector<double> x;
                    for (unsigned int i = 0; i < ptSet.size(); i++) {
                        double resid = fabs(ptSet[i].x * ta + ptSet[i].y * tb +
                                            ptSet[i].z * tc + td);
                        x.push_back(resid);
                        if (resid < depth_abs) {
                            pixel_count++;
                        }
                    }
                    //和目前最好的结果比较
                    if (pixel_count >= maximum) {
                        a = ta;
                        b = tb;
                        c = tc;
                        d = td;
                        maximum = pixel_count;
                        u.assign(x.begin(), x.end());
                    }

                    //if (k > 2000 && k % 1000 == 0) {
                    if (k % 1000 == 0) {
                        double rat = maximum / (double)I[textindex].size();
                        if (rat < 0.3 && depth_abs < 0.003) {
                            depth_abs += 0.0001;
                        } else {
                            double maximum_2 = 0;
                            for (unsigned int i = 0; i < ptSet.size(); i++) {
                                double resid =
                                    fabs(ptSet[i].x * a + ptSet[i].y * b +
                                         ptSet[i].z * c + d);
                                if (resid < depth_abs + 0.0001) {
                                    maximum_2++;
                                }
                            }
                            if (maximum_2 > maximum + I[textindex].size() * 0.02) { // 15000 * 2%
                                depth_abs += 0.0001;
                                maximum = maximum_2;
                            }
                        }
                    }
                }

                 //cout << "before refine: " << a << " " << b << " " << c << " "
                 //<< d << " "<< endl;

                int round = 1000;
                for (int i = 0; i < round; i++) {
                    for (int j = 2000; j >= 2; j /= 10) {
                        int med = j / 2;
                        double da = (double)(rand() % j - med) / 10000;
                        double db = (double)(rand() % j - med) / 10000;
                        double dc = (double)(rand() % j - med) / 10000;
                        double dd = (double)(rand() % j - med) / 1000;

                        double ra = a + da;
                        double rb = b + db;
                        double rc = c + dc;
                        double rd = d + dd;
                        double sq = sqrt(ra * ra + rb * rb + rc * rc);
                        ra /= sq;
                        rb /= sq;
                        rc /= sq;
                        rd /= sq;

                        // double nor = ra * norm_this.x + rb * norm_this.y + rc
                        // * norm_this.z; if (abs(nor) < 0.1736) continue;
                        ////cos(80) = 0.1736
                        ////cos(75) = 0.2588

                        double pixel_count = 0;
                        vector<double> x;
                        for (unsigned int i = 0; i < ptSet.size(); i++) {
                            double resid =
                                fabs(ptSet[i].x * ra + ptSet[i].y * rb +
                                     ptSet[i].z * rc + rd);
                            x.push_back(resid);
                            if (resid < depth_abs) {
                                pixel_count++;
                            }
                        }
                        if (pixel_count >= maximum) {
                            a = ra;
                            b = rb;
                            c = rc;
                            d = rd;
                            maximum = pixel_count;
                            u.assign(x.begin(), x.end());
                        }
                    }
                }

                // cout << "after refine: "<< a << " " << b << " " << c << " "
                // << d << " " << endl;
                sort(u.begin(), u.end());
                std::cout << "u.0.5(): " << u[u.size() / 2]
                          << "    u.0.8(): " << u[u.size() * 4 / 5] << endl;
                std::cout << "maximum: " << maximum
                          << "    allnum: " << I[textindex].size()
                          << "    rate: " << (float)maximum / (float)I[textindex].size()
                          << endl;
                std::cout << endl;

                gs->cannylines->norm4[textindex].x = a;
                gs->cannylines->norm4[textindex].y = b;
                gs->cannylines->norm4[textindex].z = c;
                gs->cannylines->norm4[textindex].w = d;
            }
        }
    }
    
    std::cout << endl << "Iteratively Start Pixel Filling" << endl;
    fakecuda(*gs);
    
    if (false) {
        for (int y = 0; y < gs->row; y++) {
            for (int x = 0; x < gs->col; x++) {
                int pindex = y * gs->col + x;  //每个点像素下标
                int canny = gs->lines->canny[pindex];
                if (gs->cannylines->text[canny] == -1) {
                    float dep_A = gs->lines->fakedepth[pindex];
                    //要将覆盖的结果带进去才行
                    int left = pindex - 1;
                    if (x > 0 && gs->lines->canny[left] != canny) {
                        gs->cannylines->borlen[canny]++;
                        float dep_B = disparityDepthConversion(cameraParams.f, cameraParams.cameras[0].baseline,gs->lines->depth[left]);
                        gs->cannylines->depdif[canny] += abs(dep_A - dep_B);
                    }
                    int right = pindex + 1;
                    if (x < gs->col - 1 && gs->lines->canny[right] != canny) {
                        gs->cannylines->borlen[canny]++;
                        float dep_B = disparityDepthConversion(cameraParams.f, cameraParams.cameras[0].baseline,gs->lines->depth[right]);
                        gs->cannylines->depdif[canny] += abs(dep_A - dep_B);
                    }
                    int up = pindex - gs->col;
                    if (y > 0 && gs->lines->canny[up] != canny) {
                        gs->cannylines->borlen[canny]++;
                        float dep_B = disparityDepthConversion(cameraParams.f, cameraParams.cameras[0].baseline,gs->lines->depth[up]);
                        gs->cannylines->depdif[canny] += abs(dep_A - dep_B);
                    }
                    int down = pindex + gs->col;
                    if (y < gs->row - 1 && gs->lines->canny[down] != canny) {
                        gs->cannylines->borlen[canny]++;
                        float dep_B = disparityDepthConversion(cameraParams.f, cameraParams.cameras[0].baseline,gs->lines->depth[down]);
                        gs->cannylines->depdif[canny] += abs(dep_A - dep_B);
                    }
                }
            }
        }
        for (int textindex = 0; textindex < gs->cannylines->n; textindex++) {
            if (gs->cannylines->text[textindex] == -1) {
                gs->cannylines->depdif[textindex] /= gs->cannylines->borlen[textindex];
                cout << "textindex: " << textindex << endl;
                cout << "gs->cannylines->depdif[textindex]: "  << gs->cannylines->depdif[textindex] << endl;
                cout << "gs->cannylines->borlen[textindex]: "  << gs->cannylines->borlen[textindex] << endl;
                /*if (gs->cannylines->depdif[textindex] > 1)
                    gs->cannylines->text[textindex] = 1;*/
            }
        }
    }

    std::cout << endl << "Iteratively Start Pixel Filling" << endl;
    fillcuda(*gs);

    Mat_<Vec3f> norm0 =
        Mat::zeros(img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC3);
    Mat_<float> cudadisp =
        Mat::zeros(img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC1);
    for (int i = 0; i < img_grayscale[0].cols; i++)
        for (int j = 0; j < img_grayscale[0].rows; j++) {
            int center = i + img_grayscale[0].cols * j;
            float4 n = gs->lines->norm4[center];
            norm0(j, i) = Vec3f(n.x, n.y, n.z);
            cudadisp(j, i) = gs->lines->norm4[i + img_grayscale[0].cols * j].w;
        }

    cout << endl;

    printf("cudadisp: %f\n", cudadisp(400, 400));
    Mat_<Vec3f> norm0disp = norm0.clone();
    Mat planes_display, planescalib_display, planescalib_display2;
    getNormalsForDisplay(norm0disp, planes_display);
    testImg_display.copyTo(planes_display(Rect(cols - testImg_display.cols, 0,
        testImg_display.cols, testImg_display.rows)));


    string normalcname = inputFiles.normal_folder;
    const char* folder = normalcname.data();
    string imgname_2 = imgname.substr(0, 8);
    const char* normalname = imgname_2.data();
    writeImageToFile(folder, normalname, planes_display);

    result_path << "/";
    std::string result_folder_2 = result_path.str();
    const char* outputFolder = result_folder_2.data();

    writeImageToFile(outputFolder, "TSAR_normals", planes_display);
    planes_display.release();

    Mat cost_display;
    normalize(cudadisp, cost_display, 0, 65535, NORM_MINMAX, CV_16U);
    printf("cudadisp: %f\n", cudadisp(400, 400));


    Mat_<float> disp0 = cudadisp.clone();

    char outputPath[256];
    sprintf(outputPath, "%s/TSAR_disp.dmb", outputFolder);
    writeDmb(outputPath, disp0);
    sprintf(outputPath, "%s/TSAR_normals.dmb", outputFolder);
    writeDmbNormal(outputPath, norm0);

    Mat_<float> distImg;

    cout << " algParams.num_img_processed: " << algParams.num_img_processed << endl;
    // store 3D coordinates to file
    CameraParameters camParamsNotTransformed = getCameraParameters(algParams, 
        *(gs->cameras), inputFiles, algParams.cam_scale, false);
    char plyFile[256];
    sprintf(plyFile, "%s/TSAR_model.ply", outputFolder, 0);

    storePlyFileBinary(plyFile, disp0, norm0, img_grayscale[0],
                        camParamsNotTransformed.cameras[0], distImg);
    Mat dist_display, dist_display_col;
    getDisparityForDisplay(distImg, dist_display, dist_display_col,
                            algParams.max_disparity);
    //writeImageToFile(outputFolder, "dist", dist_display);
    //writeImageToFile(outputFolder, "dist_col", dist_display_col);

    // cout << "before time output" << endl;
    t = getTickCount() - t;
    double rt = (double)t / getTickFrequency();

    ofstream myfile;
    //store results to file
    char resultsFile[256];
    sprintf ( resultsFile, "%s/TSAR_results.txt", outputFolder );
    myfile.open(resultsFile, ios::out | ios::app);
    myfile << "Total runtime: " << rt << " sec ( " << rt / 60.0f << " min)" << endl;
    myfile.close();
    cout << "Total runtime including disk i/o: " << rt << "sec" << endl;

    delTexture(algParams.num_img_processed, gs->imgs, gs->cuArray);
    delete gs;
    delete &algParams;
    cudaDeviceSynchronize();

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_help(argv);
        return 0;
    }
    GlobalState* gs = new GlobalState;
    InputFiles inputFiles;
    OutputFiles outputFiles;
    AlgorithmParameters* algParams = new AlgorithmParameters;
    GTcheckParameters gtParameters;
;
    int ret = getParametersFromCommandLine(argc, argv, inputFiles, outputFiles,
                                           *algParams, gtParameters);
    if (ret != 0) return ret;

    texture(inputFiles, gs);

    gslic(inputFiles, gs);

    selectCudaDevice();

    Results results;
    ret = runGipuma(inputFiles, outputFiles, *algParams, gtParameters, results, gs);

    return 0;
}