# uhh_controller_pkg (this is the most updated branch)

## In src/node/robot_controller:
- plate_detector_SIFT.py is the script used during the competition which uses SIFT to isolate license plates in an image and make license plate and parking ID predictions
- plate_detector.py is the non-SIFT version of the above script which did not work as well
- run.py is the script used during the competition to control robot movement using imitation learning
- robot_controller.py is essentially plate_detector_SIFT.py and run.py in one script, which did not run well
