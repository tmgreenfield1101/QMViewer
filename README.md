This is a set of scripts (currently) to interrogate the output from QuakeMigrate, an earthquake detection and location algorithm. The primary aim of the code is to facilitate the setting of parameters for triggering, defining a set of earthquakes for location, manual pick refinement, earthquake relocation using NonLinLoc and a quick focal mechanism solver. 

So far, only a structure of the code is present. GUIs are defined and in many cases functionality is implemented. However, many buttons and check boxes still don't do anything. 

todo:
- Implement the ability to retrigger sections of data
- Write out triggered earthquake catalogues for relocation using locate. 
- select events in the detect/trigger window
- enable the ability to use archive waveforms rather than the saved waveforms from QM
- enable add_new_A0 functionality - do we even need this?
- implement a better system of warnings and errors rather than print to screen
- some faff with calculating magnitudes means it might be easier to use QM amplitude picking functionality. Should we bother with saving amplitude picks for real (may be more useful) and raw (probably not useful) traces?
- currently user has to go to the magnitude page to calculate magnitude. Should a button be present on the picking page for calculating magnitude.
- same as above the nonlinloc relocation
- enable the ability to manually remove an amplitude pick by clicking on it.
- issue a warning to save picks 
- use ctrl+s for saving picks
- bugs, bugs, bugs
- upload to github
- probably many coding issues....
