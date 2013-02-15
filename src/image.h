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
#include "SingleImageComps.h"
#ifndef IMAGE_H_
#define IMAGE_H_
using namespace cv;
using namespace std;


class Image {
	Mat methods;
	Mat grndtrth;
	vector<vector<Point> > contours1;
	vector<Vec4i> hierarchy1;
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy2;
	double CenX1, CenY1, CenX2, CenY2;

public:
    Image(Mat, Mat);

	void ImageManip();
    void Normalize();
    void Labeling();
    void ImageStats();
    void CentroidConstraint();
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
	threshold(methods,methods,0.5,1.0,THRESH_BINARY);
	threshold(grndtrth,grndtrth,0.5,1.0,THRESH_BINARY);

}

void Image :: Labeling()
{


    vector <vector <Point2i> > blobs1;
    vector <vector <Point2i> > blobs2;
    Mat NormMethods, NormGrndtrth;
    NormMethods = SingleImageComps(methods,blobs1);
    NormGrndtrth = SingleImageComps(grndtrth,blobs2);

    methods = NormMethods;
    grndtrth = NormGrndtrth;

}

void Image :: ImageStats()
{
	  findContours(methods,contours1,hierarchy1,CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE );
	  findContours(grndtrth,contours2,hierarchy2,CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE );

	  int idx1 = 0;
	        for( ; idx1 >= 0; idx1 = hierarchy1[idx1][0] )
	         {
	            Scalar color( rand()&255, rand()&255, rand()&255 );
	            drawContours( methods, contours1, idx1, color, CV_FILLED, 8, hierarchy1 );
	        }


	  int idx2 = 0;
	        for( ; idx2 >= 0; idx2 = hierarchy2[idx2][0] )
	         {
	             Scalar color( rand()&255, rand()&255, rand()&255 );
	             drawContours( grndtrth, contours2, idx2, color, CV_FILLED, 8, hierarchy2 );

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




                for(unsigned int a = 0; a < contours1.size(); a++)
                	for (unsigned int  b = 0; b < contours2.size(); b++)
                	   {

                		CenX1 = centroid1[a].x;
                		CenY1 = centroid1[a].y;
                		CenX2 = centroid2[b].x;
                		CenY2 = centroid2[b].y;
                		complex<double> ComplexDist((CenX1-CenX2),(CenY1-CenY2));
                		double CentDist = abs(ComplexDist);
            //		   	double CentAng = arg(ComplexDist);

                		cout << "Centroid Distance [" << a << "][" << b << "]: " << CentDist << endl;
            //		   	cout << "Centroid Angle [" << a << "][" << b << "]: " <<  CentAng << endl;
                		cout << endl;



                	   }

}





int Image :: Display()
{
	    namedWindow( "Display window1", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    imshow( "Display window1", methods );  // Show our image inside it.
	    namedWindow( "Display window2", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    imshow( "Display window2", grndtrth );  //
	    waitKey(0);                                          // Wait for a keystroke in the window

	    return 0;
}



#endif /* IMAGE_H_ */
