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
	vector <vector <Point2i> > blobs1;
	Image imgObj (Methods, Grndtrth);
	imgObj.ImageManip();
	imgObj.Normalize();
	imgObj.Labeling();
	imgObj.ImageStats();
	imgObj.CentroidConstraint();
	imgObj.Display();

}


