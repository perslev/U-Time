History
-------

1.0.0 (2022-04-11)
--------------------
This version includes a major refactoring of the utime package in several areas:

* No longer relies on code from the mpunet package (https://github.com/perslev/MultiPlanarUNet).
* All code related to loading/processing/saving of PSG data has been separated into the 'psg_utils' package (https://github.com/perslev/psg-utils).
* Hyperparameter file loading/manipulation/saving has been separated into the 'yamlhparams' package (https://github.com/perslev/yamlhparams).
* Logging has been overhauled and standardized across all utime scripts (invoked with ut <script> command / in utime/bin).
* Many other smaller changes to both functions and classes
* Some hyperparameter configuration attributes have been renamed or changed/removed entirely.

In addition, this update includes various bug fixes (e.g., issue 43 - non changing validation loss).
Some rarely used loss functions and metrics have been removed (these are still available in the mpunet package).

All in all, this version includes many API changes that are incompatible with earlier versions. However, all utime
scripts (ut <script> commands) may - except for a few command line argument naming changes - be used as in earlier versions
and with earlier versions of utime projects. Old hyperparameter files with incompatible attribute names will to some extend
be attempted automatically converted. When not possible, the user will need to re-initialize the project with 'ut init'
to create a fresh set of hyperparameters.

0.0.1 (2019-01-11)
--------------------
* Project created and packaged
