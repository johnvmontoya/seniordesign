/*
 * image.h
 *
 *  Created on: Feb 9, 2013
 *      Author: face52
 *
 *      Class declaration and definition for Class Image
 */

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <map>
#include "SingleImageComps.h"
#ifndef IMAGE_H_
#define IMAGE_H_
using namespace cv;
using namespace std;


class Image {
	Mat methods;
	Mat grndtrth;
	Mat NormGrndtrth;
	Mat NormMethods;
	Mat NGrndtrth;
	Mat NMethods;
	vector<vector<Point> > contours1;
	vector<Vec4i> hierarchy1;
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy2;
	double CenX1, CenY1, CenX2, CenY2;
	int Map1to2[];

public:
    Image(Mat, Mat);

	void ImageManip();
    void Normalize();
    void Labeling();
    void ImageStats();
    void CentroidConstraint();
//    void SetDistanceConstraint();
	int Display();

};

//----------------------------------------------------------------------------------------------------------------------------

Image::Image(Mat met, Mat grnd)					//   Constructor
	{
		met = imread( "images/test1.png", 0);
		grnd = imread( "images/test2.png", 0);

		  if( !met.data || !grnd.data )    // check and make sure image data exists
		    {
		      printf( "No image data \n" );
		    }


		methods = met;
		grndtrth = grnd;

	}


/*--------------------------------------Functions------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------*/
void Image :: ImageManip()
{
	flip(methods, methods, 0);
	flip(grndtrth, grndtrth, 0);

	resize(methods,methods,grndtrth.size(),CV_INTER_LANCZOS4);
	resize(grndtrth,grndtrth,grndtrth.size(),CV_INTER_LANCZOS4);

	methods.convertTo(methods,CV_32F, 1./255);
	grndtrth.convertTo(grndtrth,CV_32F, 1./255);

}


void Image :: Normalize()
{
	Mat mask;
	mask = grndtrth;
	multiply(mask,methods,methods,1);

//	threshold(methods,methods,0.0,1.0,THRESH_BINARY);
//	threshold(grndtrth,grndtrth,0.5,1.0,THRESH_BINARY);
//	threshold(methods,methods,127,255,THRESH_BINARY);
//	threshold(grndtrth,grndtrth,127,255,THRESH_BINARY);

	methods.convertTo(methods,CV_8U);
	grndtrth.convertTo(grndtrth,CV_8U);
}



void Image :: Labeling()
{
    vector <vector <Point2i> > blobs1;
    vector <vector <Point2i> > blobs2;
    NormGrndtrth = cv::Mat::zeros(grndtrth.size(), CV_8UC3);
    NormMethods = cv::Mat::zeros(methods.size(), CV_8UC3);
    threshold(methods,NMethods,0.0,1.0,THRESH_BINARY);
    threshold(grndtrth,NGrndtrth,0.0,1.0,THRESH_BINARY);
    SingleImageComps(NMethods,blobs2);
    SingleImageComps(NGrndtrth,blobs1);
    unsigned char n,m;
    m = 1;
    n = 1;

    for(size_t i=0; i < blobs1.size(); i++)
      {
         unsigned char r = m;
         unsigned char g = m;
         unsigned char b = m;

         for(size_t j=0; j < blobs1[i].size(); j++)
           {
                int x = blobs1[i][j].x;
                int y = blobs1[i][j].y;

                NormGrndtrth.at<Vec3b>(y,x)[0] = b;
                NormGrndtrth.at<Vec3b>(y,x)[1] = g;
                NormGrndtrth.at<Vec3b>(y,x)[2] = r;
            }
         m++;
        }



    for(size_t i=0; i < blobs2.size(); i++)
      {
          unsigned char r = n;
          unsigned char g = n;
          unsigned char b = n;

          for(size_t j=0; j < blobs2[i].size(); j++)
            {
                int x = blobs2[i][j].x;
                int y = blobs2[i][j].y;

                NormMethods.at<Vec3b>(y,x)[0] = b;
                NormMethods.at<Vec3b>(y,x)[1] = g;
                NormMethods.at<Vec3b>(y,x)[2] = r;
            }
          n++;
        }
}



void Image :: ImageStats()
{
	findContours(grndtrth,contours1,hierarchy1,CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE );
	findContours(methods,contours2,hierarchy2,CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE );

	  int idx1 = 0;
	        for( ; idx1 >= 0; idx1 = hierarchy1[idx1][0] )
	         {
	            Scalar color( rand()&255, rand()&255, rand()&255 );
	            drawContours( grndtrth, contours1, idx1, color, CV_FILLED, 8, hierarchy1 );
	        }


	  int idx2 = 0;
	        for( ; idx2 >= 0; idx2 = hierarchy2[idx2][0] )
	         {
	             Scalar color( rand()&255, rand()&255, rand()&255 );
	             drawContours( methods, contours2, idx2, color, CV_FILLED, 8, hierarchy2 );
	         }
}

void Image :: CentroidConstraint()
{

    /// Get the moments
      vector<Moments> mu1(contours1.size() );
      for( unsigned int i = 0; i < contours1.size(); i++ )
         { mu1[i] = moments( contours1[i], false ); }

      ///  Get the Methods Centroids:
      vector<Point2f> centroid1( contours1.size() );
      for( unsigned int j = 0; j < contours1.size(); j++ )
         { centroid1[j] = Point2f( mu1[j].m10/mu1[j].m00 , mu1[j].m01/mu1[j].m00 ); }


      vector<Moments> mu2(contours2.size() );
      for( unsigned int i = 0; i < contours2.size(); i++ )
         { mu2[i] = moments( contours2[i], false ); }

      ///  Get the GrndtrthCentroids:
      vector<Point2f> centroid2( contours2.size() );
      for( unsigned int j = 0; j < contours2.size(); j++ )
         { centroid2[j] = Point2f( mu2[j].m10/mu2[j].m00 , mu2[j].m01/mu2[j].m00 ); }



      //printf("\t Info: Area and Contour Length \n");
              for( unsigned int k = 0; k< contours1.size(); k++ )
                 {
                 	 cout <<" * Methods Centroids: Component[" << k << "] X coord: " << centroid1[k].x << " Y coord: " << centroid1[k].y << endl;
                 }

                cout << endl;

                for( unsigned int k = 0; k< contours2.size(); k++ )
                   {
                   	   cout <<" * Ground Truth Centroids: Component[" << k << "] X coord: " << centroid2[k].x << " Y coord: " << centroid2[k].y << endl;
                   }


                double MinCent;
                double MinimumCentroid[contours1.size()];
                double TempCent[contours2.size()];
                int TempIndex;
                int index;
                int Map1to2[contours1.size()];

                for(unsigned int a = 0; a < contours1.size(); a++)
                {
                	for (unsigned int  b = 0; b < contours2.size(); b++)
                	   {
                		CenX1 = centroid1[a].x;
                		CenY1 = centroid1[a].y;
                		CenX2 = centroid2[b].x;
                		CenY2 = centroid2[b].y;
                		complex<double> ComplexDist((CenX1-CenX2),(CenY1-CenY2));
                		double CentDist = abs(ComplexDist);
                		cout << "Centroid Distance [" << a << "][" << b << "]: " << CentDist << endl;
            //		   	cout << "Centroid Angle [" << a << "][" << b << "]: " <<  CentAng << endl;
                		cout << endl;
                		TempCent[b] = CentDist;
                   	   }

                	MinCent = TempCent[0];
                	TempIndex = 1;
                	index = 1;

                	for(unsigned int c = 1;c < contours2.size(); c++)
                	{
                   		TempIndex++;
                		if(MinCent > TempCent[c])
                		{
                			MinCent = TempCent[c];
                			index = TempIndex;
                		}
                   	}

                	MinimumCentroid[a] = MinCent;
                	Map1to2[a] = index;
               }

                for(unsigned int k = 0; k < contours1.size(); k++)
                	cout <<"Map1to2: " << Map1to2[k] <<". " << MinimumCentroid[k] << endl;
}
/*
void Image :: SetDistanceConstraint()
{

	Mat grndtrth_labeled = NormGrndtrth;
	Mat methods_labeled = NormMethods;


//	for(unsigned int a = 0;a < contours1.size(); a++)
//		for(unsigned int b = 0;b < contours2.size(); b++)
//		{

//		}
}
*/






int Image :: Display()
{


		namedWindow( "Display window1", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    imshow( "Display window1", methods );  // Show our image inside it.
	    namedWindow( "Display window2", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    imshow( "Display window2", grndtrth );  //


//	    cout << endl << methods << endl;
	    waitKey(0);      // Wait for a keystroke in the window

	    return 0;
}



#endif /* IMAGE_H_ */
