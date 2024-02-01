# hbn_analysis
This is the analysis workflow for:
Camacho, M.C., Nielsen, A.N., Balser, D. _et al._ Large-scale encoding of emotion concepts becomes increasingly similar between individuals from childhood to adolescence. _Nat Neurosci_ (2023). https://doi.org/10.1038/s41593-023-01358-9

Codes used in this analysis can be found here: https://wustl.app.box.com/v/emocodespublicdata/folder/165162908841
More information on the EmoCodes coding scheme is here: https://emocodes.org/

## Overview
* 0_process_videos: Automatic feature extraction and processing of manual video codes using the EmoCodes tools
* 1_activation: scripts run on the cluster to extract activation maps using the EmoCodes outputs
* 2_SVM-emoclass: Support vector classification of activation maps, predicting label (e.g., anger, sadness, etc)
* 3_SVM-maturity: Support vector regression of activation maps, predicting age or puberty
* 4_similarity: Inter-subject representational similarity analysis and dynamic similarity analysis to test nonlinear trends across maturity.
