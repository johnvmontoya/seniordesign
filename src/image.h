
 /* image.h
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
	vector <vector <Point2i> > blobs1;
	vector <vector <Point2i> > blobs2;
	vector<vector<Point> > gtContours;
	vector<Vec4i> hierarchy1;
	vector<vector<Point> > mContours;
	vector<Vec4i> hierarchy2;
	double CenX1, CenY1, CenX2, CenY2;

public:
    Image(Mat, Mat);

	void ImageManip();
    void Normalize();
    void ImageStats();
    void ColorMap(int *Matches,string NameofConstraint);
    void DisplayMatches(int MatchCentroid,int MatchHausdorff, int MatchSet, int MatchAll);
    void CentroidConstraint(int *CentroidMap1to2,int ApplyAreaLimiting, double AreaLimit);
    void HausdorffConstraint(int *HausdorffMap1to2,int MinimizeFlag, double MaxDistance);
    double MinDist(const vector<Point>&,const vector<Point>&);
//  void SetDistanceConstraint();

};

//----------------------------------------------------------------------------------------------------------------------------

Image::Image(Mat met, Mat grnd)					//   Constructor
	{
		met = imread( "images/contour_GONG_20091014.png", 0);
		grnd = imread( "images/R4_1_drawn_euvi_new_20091014.png", 0);


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




void Image :: ImageStats()
{
	findContours(grndtrth,gtContours,hierarchy1,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE );
	findContours(methods,mContours,hierarchy2,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE );

	  int idx1 = 0;
	        for( ; idx1 >= 0; idx1 = hierarchy1[idx1][0] )
	         {
	            Scalar color(rand()&255, rand()&255, rand()&255);
	            drawContours( grndtrth, gtContours, idx1, color, CV_FILLED, 8, hierarchy1 );
	        }


	  int idx2 = 0;
	        for( ; idx2 >= 0; idx2 = hierarchy2[idx2][0] )
	         {
	             Scalar color( rand()&255, rand()&255, rand()&255 );
	             drawContours( methods, mContours, idx2, color, CV_FILLED, 8, hierarchy2 );
	         }
}

void Image :: CentroidConstraint(int *CentroidMap1to2,int ApplyAreaLimiting,double AreaLimit)
{
// inputs:
// CenroidMap1to2 - pointer to array to store matches
// ApplyAreaLimiting - turn Area Limiting on (1) or off (0)
// AreaLimit - % match of area to apply for AreaLimiting
	int Map1to2[gtContours.size()];
	double area1,area2,percentDiff;
    /// Get the moments
      vector<Moments> mu1(gtContours.size() );
      for( unsigned int i = 0; i < gtContours.size(); i++ )
         { mu1[i] = moments( gtContours[i], false ); }

      ///  Get the Methods Centroids:
      vector<Point2f> centroid1( gtContours.size() );
      for( unsigned int j = 0; j < gtContours.size(); j++ )
         { centroid1[j] = Point2f( mu1[j].m10/mu1[j].m00 , mu1[j].m01/mu1[j].m00 ); }


      vector<Moments> mu2(mContours.size() );
      for( unsigned int i = 0; i < mContours.size(); i++ )
         { mu2[i] = moments( mContours[i], false ); }

      ///  Get the GrndtrthCentroids:
      vector<Point2f> centroid2( mContours.size() );
      for( unsigned int j = 0; j < mContours.size(); j++ )
         { centroid2[j] = Point2f( mu2[j].m10/mu2[j].m00 , mu2[j].m01/mu2[j].m00 ); }



      //printf("\t Info: Area and Contour Length \n");
              /*for( unsigned int k = 0; k< gtContours.size(); k++ )
                 {
                 	 cout <<" * Methods Centroids: Component[" << k << "] X coord: " << centroid1[k].x << " Y coord: " << centroid1[k].y << endl;
                 }

                cout << endl;*/

/*                for( unsigned int k = 0; k< mContours.size(); k++ )
                   {
                   	   cout <<" * Ground Truth Centroids: Component[" << k << "] X coord: " << centroid2[k].x << " Y coord: " << centroid2[k].y << endl;
                   }*/


                double MinCent;
                double MinimumCentroid[gtContours.size()];
                double TempCent[mContours.size()];
                int TempIndex;
                int index;
                //int Map1to2[gtContours.size()]; JM - MERGE?

                for(unsigned int a = 0; a < gtContours.size(); a++)
                {
                	for (unsigned int  b = 0; b < mContours.size(); b++)
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

                	for(unsigned int c = 1;c < mContours.size(); c++)
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
                	for(unsigned int k = 0; k < gtContours.size() ; k++)
                	{
                		 area1=contourArea(gtContours[k]);
                		 area2=contourArea(mContours[Map1to2[k]-1]);
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






                for( size_t i = 0; i< gtContours.size(); i++ )
                	cout <<"Map1to2: [" << Map1to2[i] <<"]  " << MinimumCentroid[i] << endl;
                for( size_t j = 0; j< mContours.size(); j++ )
                {
                	CentroidMap1to2[j] = 0;
                }
                for( size_t k = 0; k< gtContours.size(); k++ )
                {
                	cout <<  " k: " << k << " Map1to2: " << Map1to2[k] << endl;
                	CentroidMap1to2[Map1to2[k]-1]=k+1;

                }


}
double Image :: MinDist(const vector<Point>& c1,const vector<Point>& c2)
{
// Computes minimum Euclidean distance between two points
// Used by Hausdorff Constraint
	double shortest;
	double hausDist = -1; //place holder for Hausdorff Distance to return
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

	return hausDist; //Return hausdorff distance
}
void Image :: HausdorffConstraint(int *HausdorffMap1to2, int MinimizeFlag,double MaxDistance)
{
// Finds Hausdorff distance between all components in each input contour
// After computing Hausdorff distance, selects minimum Hausdorff distance as best match for each component
// inputs:
// HausdorffMap1to2 - pointer to array to store matches
// MinimizeFlag - turn distance minimizing on (1) or off (0)
// MaxDistance - Distance to apply for distance minimization
	vector<pair<size_t,double> > distances;
	vector<pair<size_t,double> > maxDists1to2;
	vector<pair<size_t,double> > maxDists2to1;
	//int HausdorffMap1to2[gtContours.size()];

	for(size_t mContourIndex = 0; mContourIndex < mContours.size() ; ++mContourIndex)
	{

		vector<Point> currContour1 = mContours[mContourIndex];

		size_t minDistanceIndex = 0;

		double minimumDistance = DBL_MAX;

		//Iterate through each component and find the max distance to each other component

		for(size_t gtContourIndex = 0; gtContourIndex != gtContours.size(); ++gtContourIndex)
		{
			vector<Point> currContour2 = gtContours[gtContourIndex];

			double currMinDist = MinDist(currContour1,currContour2);
			double currMinDist2 = MinDist(currContour2,currContour1);
			double hausDist = max(currMinDist,currMinDist2);

			if(hausDist < minimumDistance)
			{
				minimumDistance = hausDist;
				minDistanceIndex = gtContourIndex;
			}

			if (gtContourIndex == gtContours.size() - 1)
			{
				cout << "Match Distance: " << minimumDistance << " @ Index: " << minDistanceIndex+1 << endl;
				maxDists1to2.push_back(pair<size_t,double>(minDistanceIndex+1,minimumDistance));
			}


		}

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

}

/*
void Image :: SetDistanceConstraint()
{

	Mat grndtrth_labeled = NormGrndtrth;
	Mat methods_labeled = NormMethods;


//	for(unsigned int a = 0;a < gtContours.size(); a++)
//		for(unsigned int b = 0;b < mContours.size(); b++)
//		{

//		}
}
*/



void Image :: ColorMap(int *Matches, string NameofConstraint)
{
// Apply color map to matching components and displays output image
// Runs per constraint, input array of matches and name of constraint (for labelling)
// Improve this by ensuring all components that match multiple components have the same colors

	//colorWheel is a list of colors to iterate through. The last elements are random and should be defined as specific colors to prevent
	//high numbered components from having different colors
	Scalar colorWheel[]={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(255,0,255),Scalar(0,255,255),Scalar(255,255,255),Scalar(255,153,0),Scalar(255,102,153),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255),Scalar(rand()%255,rand()%255,rand()%255)};


    Mat MapGrndtrth = cv::Mat::zeros(grndtrth.size(), CV_8UC3);
    Mat MapMethods = cv::Mat::zeros(methods.size(), CV_8UC3);
    Mat Legend = cv::Mat::zeros(methods.size(), CV_8UC3);
    std::stringstream ss;
	for(unsigned int k = 0; k < mContours.size(); k++)
		{
		cout << "[" << (k+1) << "]: ";
		cout << " " + NameofConstraint + " " << Matches[k] << endl;
		}

	  cv::putText(Legend, "Methods", Point(20,25), CV_FONT_HERSHEY_PLAIN,2,Scalar(255,255,255),2,8);
	  cv::putText(Legend, "Ground Truth", Point(280,25), CV_FONT_HERSHEY_PLAIN,2,Scalar(255,255,255),2,8);

	  for( size_t i = 0; i< mContours.size(); i++ )
	     {
	      Scalar color = Scalar (colorWheel[(int)i]); //choose next color from colorWheel

	      ss.str("");
	      ss.clear();
	      ss << "Component " << i+1;
	      drawContours( MapMethods, mContours, (int)i, color, CV_FILLED, 8, hierarchy1, 0, Point() );
	      putText(Legend, ss.str(), Point(20,(int)i*30+50), CV_FONT_HERSHEY_PLAIN,2,color,2,8);
	      cv::rectangle(Legend,Point(0,(int)i*30+50),Point(20,(int)i*30+30),color,CV_FILLED,8);
	      if (Matches[(int)i] != 0)
	    		   {
	    	  	  	  drawContours( MapGrndtrth, gtContours, Matches[(int)i]-1, color, CV_FILLED, 8, hierarchy1, 0, Point() );
	    		      ss.str("");
	    		      ss.clear();
	    		      ss << "Component " << Matches[(int)i];
	    		      putText(Legend, ss.str(), Point(300,(int)i*30+50), CV_FONT_HERSHEY_PLAIN,2,color,2,8);
	    		      cv::rectangle(Legend,Point(280,(int)i*30+50),Point(300,(int)i*30+30),color,CV_FILLED,8);

	    		   }
	     }

	  namedWindow( NameofConstraint +" Methods", CV_WINDOW_AUTOSIZE );// Create a window for display.
	  imshow( NameofConstraint +" Methods", MapMethods );  // Show our image inside it.
	  namedWindow( "Ground Truth ("+NameofConstraint+")", CV_WINDOW_AUTOSIZE );// Create a window for display.
	  imshow( "Ground Truth ("+NameofConstraint+")", MapGrndtrth );  //
	  namedWindow( "Legend ("+NameofConstraint+")", CV_WINDOW_AUTOSIZE );// Create a window for display.
	  imshow( "Legend ("+NameofConstraint+")", Legend );  //

	  //Uncomment the following lines to write the images to files
	  //imwrite("methods-"+NameofConstraint+".png",MapMethods);
	  //imwrite("grndtrth-"+NameofConstraint+".png",MapGrndtrth);
	  //imwrite("legend-"+NameofConstraint+".png",Legend);

}
void Image :: DisplayMatches(int MatchCentroid, int MatchHausdorff, int MatchSet, int MatchAll)
{
// Calls ColorMap to display matches for selected constraints
// Also computes matches for all selected constraints
// Constraint parameters are selected in here!

	int HausdorffMap1to2[mContours.size()];
	int CentroidMap1to2[mContours.size()];
	int SetMap1to2[mContours.size()];
	int AllMap1to2[mContours.size()];

	if (MatchCentroid == 1)
	{
		//Set Centroid parameters here!
		CentroidConstraint(CentroidMap1to2,0,20.00);
		ColorMap(CentroidMap1to2,"Centroid");
	}
	if (MatchHausdorff == 1)
	{
		//Set Hausdorff parameters here!
		HausdorffConstraint(HausdorffMap1to2,0,75.00);
		ColorMap(HausdorffMap1to2,"Hausdorff");
	}
/*	if (MatchSet == 1)
	{
		//Set Set Distance parameters here!
		SetDistanceConstraint(SetMap1to2,0,75.00);
		ColorMap(SetMap1to2,"Set Distance");
	}*/


	if (MatchAll == 1)
	{
		if(MatchHausdorff ==0 && MatchCentroid == 0 && MatchSet == 0)
		{
			cout << "No Constraints selected!" << endl;
			for(unsigned int k = 0; k < mContours.size(); k++)
			{
				AllMap1to2[k] = 0;
			}

		}
		else if(MatchHausdorff ==0 && MatchCentroid == 0 && MatchSet == 1)
		{
			for(unsigned int k = 0; k < mContours.size(); k++)
			{
				AllMap1to2[k] = SetMap1to2[k];
			}

		}
		else if(MatchHausdorff ==0 && MatchCentroid == 1 && MatchSet == 0)
		{
			for(unsigned int k = 0; k < mContours.size(); k++)
			{
				AllMap1to2[k] = CentroidMap1to2[k];
			}

		}
		else if(MatchHausdorff ==0 && MatchCentroid == 1 && MatchSet == 1)
		{
			for(unsigned int k = 0; k < mContours.size(); k++)
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
			for(unsigned int k = 0; k < mContours.size(); k++)
			{
				AllMap1to2[k] = HausdorffMap1to2[k];
			}

		}
		else if(MatchHausdorff ==1 && MatchCentroid == 0 && MatchSet == 1)
		{
			for(unsigned int k = 0; k < mContours.size(); k++)
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
			for(unsigned int k = 0; k < mContours.size(); k++)
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
			for(unsigned int k = 0; k < mContours.size(); k++)
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

		ColorMap(AllMap1to2,"All");
	}

}





#endif /* IMAGE_H_ */
