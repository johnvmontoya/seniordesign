/*
 * SingleImageComps.h
 *
 *  Created on: Feb 10, 2013
 *      Author: face52
 */


#include <cv.h>
#include <highgui.h>
#include <iostream>
#ifndef SINGLEIMAGECOMPS_H_
#define SINGLEIMAGECOMPS_H_
using namespace cv;
using namespace std;


void SingleImageComps(const Mat &binary, vector <vector<Point2i> > &blobs);



#endif /* SINGLEIMAGECOMPS_H_ */
