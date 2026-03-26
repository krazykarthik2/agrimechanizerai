main.py contains script that is stable and independent of other python scripts other scripts are test

precise.py is the new script that will be osciallation aware and calculate the next ones and then predict the next step and control sprayers

calibrate_wiper2.py is the script that will calibrate cameras with respect to arc made by the sprayer. it fits the arc curve into parameters of ml algo ( function ) and you can simply use that to predict the next  point with respect to theta and control loop delay + physical fluid delay

