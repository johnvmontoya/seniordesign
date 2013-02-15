/*
 * SingleImageComps.cpp
 *
 *  Created on: Feb 10, 2013
 *      Author: face52
 */




/*
 * SingleImageComp.cpp
 *
 *  Created on: Nov 3, 2012
 *      Author: face52
 */


#include "SingleImageComps.h"
#include <cv.h>
#include <highgui.h>
#include <iostream>
using namespace cv;
using namespace std;

Mat SingleImageComps(const Mat &binary, vector <vector<Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    Mat label_image;
    binary.convertTo(label_image, CV_8UC1); // weird it doesn't support CV_32S!

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < binary.rows; y++)
       {
        for(int x=0; x < binary.cols; x++)
          {
            if((int)label_image.at<float>(y,x) != 1)
            {
                continue;
            }

            Rect rect;
            floodFill(label_image, Point(x,y), Scalar(label_count), &rect, Scalar(0), Scalar(0), 8);

            vector <Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++)
              {
                for(int j=rect.x; j < (rect.x+rect.width); j++)
                  {
                    if((int)label_image.at<float>(i,j) != label_count)
                     {
                        continue;
                     }

                    blob.push_back(Point2i(j,i));
                  }
               }

            blobs.push_back(blob);

            label_count++;
        }
    }

return label_image;

}


