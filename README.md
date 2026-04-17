# MS285
Class Project

Future Predictions of the Monterey Canyon head with machine learning 

Motivation

My thesis research involves repeat bathymetry surveys over the canyon head to analyze sediment transport processes over the course of a year. Multiple groups have surveyed this similar region over the course of the last 20 years with enough data fora sample size to train the model on how the shapes of the canyon have been changing. This could give us insight on large features that could be ready to slide or sluff causing major turbitdity currents down canyon. A direct example would be the satalite imaging that USGS has of the big sur coast that turns out could have predicted the area of Regents slide failing. 

Data

This data will be in the form of KML file type or a xyz file depending on with the past groups used as their eporting raster file for editing in ArcGIS. This Data will then be complied into one folder with each file name being the data at which the survey was conducted. This will help the model recognize the evolution of the canyon wit htime stamps.  

Model 

I will use a classification model to learn and understand how the nearshore bathymetry has changed ove the course of 20 years then use that data for it to then output future bathymetry profiles. I'll prompt the model to output me files for each year for the next 20 years. 

Analysis

After the model has ran I will then overlay the new output KML files Google Earth for viewing and targeting large failures that have occured. Once i have found the main erosional events (assuming that this canyon truely is in a state of erosion and had not reach a new equalibrium) I will do a raster subtraction to quatify cubic meters of sediment lost into the canyon.  
