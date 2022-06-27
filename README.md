# hbn_analysis
This is the analysis workflow for "Evidence for large-scale encoding of contextualized emotion categories that refines across childhood and adolescence", currently under review.

## Overview
* 0_process_videos: Automatic feature extraction and processing of manual video codes using the EmoCodes tools
* 1_activation: scripts run on the cluster to extract activation maps using the EmoCodes outputs
* 2_SVM-emoclass: Support vector classification of activation maps, predicting label (e.g., anger, sadness, etc)
* 3_SVM-maturity: Support vector regression of activation maps, predicting age or puberty
* 4_similarity: Inter-subject representational similarity analysis and dynamic similarity analysis to test nonlinear trends across maturity.
