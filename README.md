This is a set of scripts (currently) to interrogate the output from QuakeMigrate, an earthquake detection and location algorithm. The primary aim of the code is to facilitate the setting of parameters for triggering, defining a set of earthquakes for location, manual pick refinement, earthquake relocation using NonLinLoc and a quick focal mechanism solver. 

So far, only a structure of the code is present. GUIs are defined and in many cases functionality is implemented. However, many buttons and check boxes still don't do anything. 

todo:
- Implement the ability to retrigger sections of data
- Write out triggered earthquake catalogues for relocation using locate. 
- enable the ability to use archive waveforms rather than the saved waveforms from QM
- implement instrument removal and the convolution of a Wood-Anderson response
- implement amplitude picking
- implement magnitude calculation
- implement a better system of warnings and errors rather than print to screen
- bugs, bugs, bugs
- upload to github
- probably many coding issues....
