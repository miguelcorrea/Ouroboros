# CHANGELOG

### v 0.3 (June 2019)
---
* Optimized model fitting code. This leads to a drastic decrease in running time without sacrificing model performance. An example case with the PDZ data is now around 18 times faster, while an example with the TCS data runs 27 times faster.
* Now requires at least scikit-learn v. 0.20
* Removed unused code

### v 0.2 (October 2018)
---
* Added unit testing
* Added the *input_handling* module
* Improvements in how user-provided parameters are checked and validated
* Improved validation of input alignments
* The code now produces a README file explaining different output files
* Improvements to project README file
* Added two-component system example data and parameter file
* Removed prior_inters option
* Added predict_contacts option (experimental, default: False)

### v 0.1 (March 2018)
---
First release