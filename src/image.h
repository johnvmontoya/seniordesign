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
/*	Mat MapGrndtrth;
	Mat MapMethods;*/
	Mat NGrndtrth;
	Mat NMethods;
	vector <vector <Point2i> > blobs1;
	vector <vector <Point2i> > blobs2;
	vector<vector<Point> > contours1;
	vector<Vec4i> hierarchy1;
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy2;
	double CenX1, CenY1, CenX2, CenY2;
	//int Map1to2[];
	//int HausdorffMap1to2[];

public:
    Image(Mat, Mat);

	void ImageManip();
    void Normalize();
    void Labeling();
    void ImageStats();

    void ColorMap();
    void ColorMapTest(int *Matches,string NameofConstraint);
    void DisplayMatches(int MatchCentroid,int MatchHausdorff, int MatchSet, int MatchAll);
    void CentroidConstraint(int *Map1to2,int ApplyAreaLimiting, double AreaLimit);
    void HausdorffConstraint(int *HausdorffMap1to2,int MinimizeFlag, double MaxDistance);

    double MinDist(const vector<Point>&,const vector<Point>&);
//    void SetDistanceConstraint();
	int Display();

};

//----------------------------------------------------------------------------------------------------------------------------

Image::Image(Mat met, Mat grnd)					//   Constructor
	{
		met = imread( "images/testA.png", 0);
		grnd = imread( "images/testB.png", 0);


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
	//flip(methods, methods, 0);
	//flip(grndtrth, grndtrth, 0);

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
	findContours(grndtrth,contours1,hierarchy1,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE );
	findContours(methods,contours2,hierarchy2,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE );

	  int idx1 = 0;
	        for( ; idx1 >= 0; idx1 = hierarchy1[idx1][0] )
	         {
	            Scalar color(rand()&255, rand()&255, rand()&255);
	            drawContours( grndtrth, contours1, idx1, color, CV_FILLED, 8, hierarchy1 );
	        }


	  int idx2 = 0;
	        for( ; idx2 >= 0; idx2 = hierarchy2[idx2][0] )
	         {
	             Scalar color( rand()&255, rand()&255, rand()&255 );
	             drawContours( methods, contours2, idx2, color, CV_FILLED, 8, hierarchy2 );
	         }
}

void Image :: CentroidConstraint(int *Map1to2,int ApplyAreaLimiting,double AreaLimit)
{
	//int Map1to2[contours1.size()];
	double area1,area2,percentDiff;
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
              /*for( unsigned int k = 0; k< contours1.size(); k++ )
                 {
                 	 cout <<" * Methods Centroids: Component[" << k << "] X coord: " << centroid1[k].x << " Y coord: " << centroid1[k].y << endl;
                 }

                cout << endl;*/

/*                for( unsigned int k = 0; k< contours2.size(); k++ )
                   {
                   	   cout <<" * Ground Truth Centroids: Component[" << k << "] X coord: " << centroid2[k].x << " Y coord: " << centroid2[k].y << endl;
                   }*/


                double MinCent;
                double MinimumCentroid[contours1.size()];
                double TempCent[contours2.size()];
                int TempIndex;
                int index;
                //int Map1to2[contours1.size()]; JM - MERGE?

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
            //    		cout << "Centroid Distance [" << a << "][" << b << "]: " << CentDist << endl;
            //		   	cout << "Centroid Angle [" << a << "][" << b << "]: " <<  CentAng << endl;
            //    		cout << endl;
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

                //Apply Area limit to centroid constraint.
                if (ApplyAreaLimiting == 1)
                {
                	for(unsigned int k = 0; k < contours1.size() ; k++)
                	{
                		 area1=contourArea(contours1[k]);
                		 area2=contourArea(contours2[Map1to2[k]-1]);
                		 percentDiff=(abs((area2 - area1) / ((area1 + area2)/2))*100);
                		 cout << "Area " << area1 <<" vs. " << area2 << endl;
                		 cout << percentDiff <<"% difference" << endl;
                		 if (percentDiff > AreaLimit)
                		 {
                			 Map1to2[k] = 0;
                			 cout << "Discarding Match " << k+1 << endl;
                		 }

                	}

                }





                //ColorMapTest(Map1to2,"Centroid");
                /*for(unsigned int k = 0; k < contours1.size(); k++)
                	cout <<"Map1to2: [" << Map1to2[k] <<"]  " << MinimumCentroid[k] << endl;*/
}
double Image :: MinDist(const vector<Point>& c1,const vector<Point>& c2)
{
	double shortest;
	double hausDist = -1;
	vector<Point>::const_iterator point1;
	vector<Point>::const_iterator point2;
	for(point1 = c1.begin(); point1 != c1.end(); ++point1)
	{
		shortest = DBL_MAX;
		for(point2 = c2.begin(); point2 != c2.end(); ++point2)
		{
			double distIJ = sqrt(((point2->x - point1->x)*(point2->x - point1->x))+((point2->y - point1->y)*(point2->y - point1->y)));
			if(distIJ < shortest)
			{
				shortest = distIJ;
			}
		}
		if (shortest>hausDist)
		{
			hausDist = shortest;
		}

	}

	//cout << " Haus Dist: " << hausDist << endl;
	return hausDist;
}
void Image :: HausdorffConstraint(int *HausdorffMap1to2, int MinimizeFlag,double MaxDistance)
{
	vector<pair<size_t,double> > distances;
	vector<pair<size_t,double> > maxDists1to2;
	vector<pair<size_t,double> > maxDists2to1;
	//int HausdorffMap1to2[contours1.size()];

	for(size_t contour1Index = 0; contour1Index < contours1.size() ; ++contour1Index)
	{

		//cout << "Contour size: " << contour1->size() << endl;
		//cout << "Contour: " << *contour << endl;
		//This gives each contour as an array of points

		vector<Point> currContour1 = contours1[contour1Index];

		size_t minDistanceIndex = 0;

		double minimumDistance = DBL_MAX;

		//cout << "Contour: " << contour1Index <<endl;
		//Iterate through each component and find the max distance to each other component

		for(size_t contour2Index = 0; contour2Index != contours2.size(); ++contour2Index)
		{
			vector<Point> currContour2 = contours2[contour2Index];

			//cout << contour1Index << " to " << contour2Index;
			double currMinDist = MinDist(currContour1,currContour2);
			//cout << contour2Index << " to " << contour1Index;
			double currMinDist2 = MinDist(currContour2,currContour1);

			double hausDist = max(currMinDist,currMinDist2);
			//cout << "Hausdorff Distance: " << hausDist << endl;;

			if(hausDist < minimumDistance)
			{
				minimumDistance = hausDist;
				minDistanceIndex = contour2Index;
			}

			if (contour2Index == contours2.size() - 1)
			{
				//cout << "Match Distance: " << minimumDistance << " @ Index: " << minDistanceIndex+1 << endl;
				maxDists1to2.push_back(pair<size_t,double>(minDistanceIndex+1,minimumDistance));
			}


			//distances.push_back(pair<size_t,double>(maxDistanceIndex,maxDistanceSoFar));
		}
		//distances.push_back(pair<size_t,double>(maxDistanceIndex,minimumDistance));
		//iterate over vector to find the minimum distance and the associated component

	}
	//Apply minimizing (if flagged) and return Maps1to2
	for(vector<pair<size_t,double> >::iterator it = maxDists1to2.begin(); it != maxDists1to2.end(); ++it)
	{


		if (MinimizeFlag == 1)
				{
					if (it->second < MaxDistance)
					{

						HausdorffMap1to2[(it - maxDists1to2.begin())]=it->first;
					}
					else
					{
						HausdorffMap1to2[(it - maxDists1to2.begin())]=0;
					}
				}
		else
				{
					HausdorffMap1to2[(it - maxDists1to2.begin())]=it->first;
				}

	}
    //for(unsigned int k = 0; k < contours1.size(); k++)


	//ColorMapTest(HausdorffMap1to2,"Hausdorff");

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


void Image :: ColorMap()
{

	int CentroidMatch[contours1.size()];
	int HausdorffMatch[contours1.size()];
	//CentroidConstraint(CentroidMatch);
	//HausdorffConstraint(HausdorffMatch,1,200.00);
    SingleImageComps(NMethods,blobs2);
    SingleImageComps(NGrndtrth,blobs1);
    unsigned char r, g, b;
    int x;
    int y = 0;

	for(int i=blobs1.size()-1; i >= 0; i--)
	      {
		     x = CentroidMatch[blobs1.size() - 1 - i];
		     switch(x)
		        {
		        case 1:
		    	    r = 255;	//red
		            g = 0;
		            b = 0;
		            break;
		        case 2:
		    	    r = 0;
		            g = 255;	// green
		            b = 0;
		            break;
		        case 3:
		    	    r = 0;
		            g = 0;
		            b = 255;	// blue
		            break;
		        case 4:
		    	     r = 255;	// yellow
		             g = 255;
		             b = 0;
		             break;
		        case 5:
		    	     r = 255;   // purple
		             g = 0;
		             b = 255;
		             break;
		        case 6:
		    	     r = 0;		// turqoise
		             g = 255;
		             b = 255;
		             break;
		        case 7:
		    	     r = 255;	//  white
		             g = 255;
		             b = 255;
		             break;
		        case 8:
		    	     r = 255;			// Orange
		             g = 153;
		             b = 0;
		             break;
		        case 9:
		    	     r = 255;		// Pink
		             g = 102;
		             b = 153;
		             break;
		        case 0:
		    	     r = 0;			// Black
		             g = 0;
		             b = 0;
		             break;
		        default:
		        	r = 255 * (rand()/(1.0 + RAND_MAX));
		        	g = 255 * (rand()/(1.0 + RAND_MAX));
		        	b = 255 * (rand()/(1.0 + RAND_MAX));
		        	break;
		        }

	         for(size_t j=0; j < blobs1[i].size(); j++)
	           {
	                int x = blobs1[i][j].x;
	                int y = blobs1[i][j].y;

	                NormGrndtrth.at<Vec3b>(y,x)[0] = b;
	                NormGrndtrth.at<Vec3b>(y,x)[1] = g;
	                NormGrndtrth.at<Vec3b>(y,x)[2] = r;
	            }

	        }
//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------
	    for(int i=blobs2.size()-1; i >= 0; i--)
	      {
	    	y++;
	    	switch(y)
		        {
		        case 1:
		    	    r = 255;	//red
		            g = 0;
		            b = 0;
		            break;
		        case 2:
		    	    r = 0;
		            g = 255;	// green
		            b = 0;
		            break;
		        case 3:
		    	    r = 0;
		            g = 0;
		            b = 255;	// blue
		            break;
		        case 4:
		    	     r = 255;	// yellow
		             g = 255;
		             b = 0;
		             break;
		        case 5:
		    	     r = 255;   // purple
		             g = 0;
		             b = 255;
		             break;
		        case 6:
		    	     r = 0;		// turqoise
		             g = 255;
		             b = 255;
		             break;
		        case 7:
		    	     r = 255;	//  white
		             g = 255;
		             b = 255;
		             break;
		        case 8:
		    	     r = 255;			// Orange
		             g = 153;
		             b = 0;
		             break;
		        case 9:
		    	     r = 255;		// Pink
		             g = 102;
		             b = 153;
		             break;
		        case 0:
		    	     r = 0;			// Black
		             g = 0;
		             b = 0;
		             break;
		        default:
		        	r = 255 * (rand()/(1.0 + RAND_MAX));
		        	g = 255 * (rand()/(1.0 + RAND_MAX));
		        	b = 255 * (rand()/(1.0 + RAND_MAX));
		        	break;
		        }


	          for(size_t j=0; j < blobs2[i].size(); j++)
	            {
	                int x = blobs2[i][j].x;
	                int y = blobs2[i][j].y;

	                NormMethods.at<Vec3b>(y,x)[0] = b;
	                NormMethods.at<Vec3b>(y,x)[1] = g;
	                NormMethods.at<Vec3b>(y,x)[2] = r;
	            }
	        }

		for(unsigned int k = 0; k < contours1.size(); k++)
		{
			cout << "Component " << (k+1) << ":" << endl;
			cout << "	Centroid Match: " << CentroidMatch[k] << endl;
	       	cout <<"	HausdorffMatch: " << HausdorffMatch[k] << endl;
		}
}

void Image :: ColorMapTest(int *Matches, string NameofConstraint)
{

	//TODO: Accept vector of arrays as input

	//FIX: Weird match on 20110113 image (Hausdorff) - Related to that image?


	/*				 (255,0,0) //red
		             (0,255,0)	// green
		    	     (0,0,255) //blue
		             (255,255,0)	// yellow
		    	     (255,0,255)   // purple
		             (0,255,255)		// turqoise
		             (255,255,255)	//  white
		             (255,153,0)			// Orange
		             (255,102,153)		// Pink*/
	Scalar colorWheel[]={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(255,0,255),Scalar(0,255,255),Scalar(255,255,255),Scalar(255,153,0),Scalar(255,102,153),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255)};


    Mat MapGrndtrth = cv::Mat::zeros(grndtrth.size(), CV_8UC3);
    Mat MapMethods = cv::Mat::zeros(methods.size(), CV_8UC3);
    Mat Legend = cv::Mat::zeros(methods.size(), CV_8UC3);
    std::stringstream ss;
	for(unsigned int k = 0; k < contours1.size(); k++)
		{
		cout << "[" << (k+1) << "]: ";
		cout << " " + NameofConstraint + " " << Matches[k] << endl;
		}

	  cv::putText(Legend, "Methods", Point(20,25), CV_FONT_HERSHEY_PLAIN,2,Scalar(255,255,255),2,8);
	  cv::putText(Legend, "Ground Truth", Point(280,25), CV_FONT_HERSHEY_PLAIN,2,Scalar(255,255,255),2,8);

	  for( size_t i = 0; i< contours1.size(); i++ )
	     {
	      Scalar color = Scalar (colorWheel[(int)i]);
		  //Scalar color = Scalar(rand()%255,rand()%155,rand()%255);
	      ss.str("");
	      ss.clear();
	      ss << "Component " << i+1;
	      drawContours( MapMethods, contours1, (int)i, color, CV_FILLED, 8, hierarchy1, 0, Point() );
	      putText(Legend, ss.str(), Point(20,(int)i*30+50), CV_FONT_HERSHEY_PLAIN,2,color,2,8);
	      cv::rectangle(Legend,Point(0,(int)i*30+50),Point(20,(int)i*30+30),color,CV_FILLED,8);
	      if (Matches[(int)i] == 0)
	    		   {
	    		   color = Scalar(0,0,0);
	    		   }
	      drawContours( MapGrndtrth, contours2, Matches[(int)i]-1, color, CV_FILLED, 8, hierarchy1, 0, Point() );
	      ss.str("");
	      ss.clear();
	      ss << "Component " << Matches[(int)i];
	      putText(Legend, ss.str(), Point(300,(int)i*30+50), CV_FONT_HERSHEY_PLAIN,2,color,2,8);
	      cv::rectangle(Legend,Point(280,(int)i*30+50),Point(300,(int)i*30+30),color,CV_FILLED,8);
	     }

	  namedWindow( NameofConstraint +" Methods", CV_WINDOW_AUTOSIZE );// Create a window for display.
	  imshow( NameofConstraint +" Methods", MapMethods );  // Show our image inside it.
	  namedWindow( "Ground Truth ("+NameofConstraint+")", CV_WINDOW_AUTOSIZE );// Create a window for display.
	  imshow( "Ground Truth ("+NameofConstraint+")", MapGrndtrth );  //
	  namedWindow( "Legend ("+NameofConstraint+")", CV_WINDOW_AUTOSIZE );// Create a window for display.
	  imshow( "Legend ("+NameofConstraint+")", Legend );  //


//	    cout << endl << methods << endl;


}
void Image :: DisplayMatches(int MatchCentroid, int MatchHausdorff, int MatchSet, int MatchAll)
{
	int HausdorffMap1to2[contours1.size()];
	int CentroidMap1to2[contours1.size()];
	int SetMap1to2[contours1.size()];
	int AllMap1to2[contours1.size()];

	if (MatchCentroid == 1)
	{
		CentroidConstraint(CentroidMap1to2,0,20.00);
		ColorMapTest(CentroidMap1to2,"Centroid");
	}
	if (MatchHausdorff == 1)
	{
		HausdorffConstraint(HausdorffMap1to2,1,75.00);
		ColorMapTest(HausdorffMap1to2,"Hausdorff");
	}

	if (MatchAll == 1)
	{
		if(MatchHausdorff ==0 && MatchCentroid == 0 && MatchSet == 0)
		{
			cout << "No Constraints selected!" << endl;
			for(unsigned int k = 0; k < contours1.size(); k++)
			{
				AllMap1to2[k] = 0;
			}

		}
		else if(MatchHausdorff ==0 && MatchCentroid == 0 && MatchSet == 1)
		{
			for(unsigned int k = 0; k < contours1.size(); k++)
			{
				AllMap1to2[k] = SetMap1to2[k];
			}

		}
		else if(MatchHausdorff ==0 && MatchCentroid == 1 && MatchSet == 0)
		{
			for(unsigned int k = 0; k < contours1.size(); k++)
			{
				AllMap1to2[k] = CentroidMap1to2[k];
			}

		}
		else if(MatchHausdorff ==0 && MatchCentroid == 1 && MatchSet == 1)
		{
			for(unsigned int k = 0; k < contours1.size(); k++)
			{
				if (CentroidMap1to2[k] == SetMap1to2[k])
				{
					AllMap1to2[k] = CentroidMap1to2[k];
				}
				else
				{
					AllMap1to2[k] = 0;
				}
			}

		}
		else if(MatchHausdorff ==1 && MatchCentroid == 0 && MatchSet == 0)
		{
			for(unsigned int k = 0; k < contours1.size(); k++)
			{
				AllMap1to2[k] = HausdorffMap1to2[k];
			}

		}
		else if(MatchHausdorff ==1 && MatchCentroid == 0 && MatchSet == 1)
		{
			for(unsigned int k = 0; k < contours1.size(); k++)
			{
				if (SetMap1to2[k] == CentroidMap1to2[k])
				{
					AllMap1to2[k] = SetMap1to2[k];
				}
				else
				{
					AllMap1to2[k] = 0;
				}
			}

		}
		else if(MatchHausdorff ==1 && MatchCentroid == 1 && MatchSet == 0)
		{
			for(unsigned int k = 0; k < contours1.size(); k++)
			{
				if (HausdorffMap1to2[k] == CentroidMap1to2[k])
				{
					AllMap1to2[k] = HausdorffMap1to2[k];
				}
				else
				{
					AllMap1to2[k] = 0;
				}
			}

		}
		else if(MatchHausdorff ==1 && MatchCentroid == 1 && MatchSet == 1)
		{
			for(unsigned int k = 0; k < contours1.size(); k++)
			{
				if (HausdorffMap1to2[k] == CentroidMap1to2[k] && HausdorffMap1to2[k] == SetMap1to2[k])
				{
					AllMap1to2[k] = HausdorffMap1to2[k];
				}
				else
				{
					AllMap1to2[k] = 0;
				}
			}

		}

		ColorMapTest(AllMap1to2,"All");
	}

}


int Image :: Display()
{


		namedWindow( "Centroid Match2", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    imshow( "Centroid Match2", NormMethods );  // Show our image inside it.
	    namedWindow( "Centroid Match1", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    imshow( "Centroid Match1", NormGrndtrth );  //


//	    cout << endl << methods << endl;
	    waitKey(0);      // Wait for a keystroke in the window

	    return 0;
}



#endif /* IMAGE_H_ */
