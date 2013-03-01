/*
 * demo.cpp
 *
 *  Created on: Feb 9, 2013
 *      Author: face52
 */

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "image.h"
using namespace cv;
using namespace std;


int main( int argc, char** argv )
{

	Mat Grndtrth;
	Mat Methods;
	Image imgObj (Methods, Grndtrth);
	imgObj.ImageManip();
	imgObj.Normalize();
	imgObj.Labeling();
	imgObj.ImageStats();
	imgObj.CentroidConstraint();
//	imgObj.SetDistanceConstraint();
	imgObj.Display();



}



/*
void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs);

int main(int argc, char **argv)
{
    cv::Mat img = cv::imread("images/test1.png", 0); // force greyscale

    if(!img.data) {
        cout << "File not found" << std::endl;
        return -1;
    }

    namedWindow("binary");
    namedWindow("labelled");

    Mat output = cv::Mat::zeros(img.size(), CV_8UC3);

    Mat binary;
    vector <vector<Point2i> > blobs;

    threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY);

    FindBlobs(binary, blobs);

    // Randomy color the blobs
    unsigned char k = 1;
    for(size_t i=0; i < blobs.size(); i++) {
 //       unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
 //       unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
 //       unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

    	unsigned char r = k;
        unsigned char g = k;
        unsigned char b = k;

        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            output.at<Vec3b>(y,x)[0] = r;
            output.at<Vec3b>(y,x)[1] = g;
            output.at<Vec3b>(y,x)[2] = b;

        }

        k++;
    }

    cv::imshow("binary", img);
    cv::imshow("labelled", output);
    cv::waitKey(0);

    cout << output << endl;

    return 0;
}





void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32FC1); // weird it doesn't support CV_32S!

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < binary.rows; y++) {
        for(int x=0; x < binary.cols; x++) {
            if((int)label_image.at<float>(y,x) != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);

            std::vector <cv::Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if((int)label_image.at<float>(i,j) != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
}
*/
