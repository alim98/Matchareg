(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim> pwd
/nexus/posix0/MBR-neuralsystems/alim
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim> cd /nexus/posix
0/MBR-neuralsystems/alim/regdata/ThoraxCBCT
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT> ls
data.zip  __MACOSX  ThoraxCBCT
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT> cd ThoraxCBCT/
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> ls
imagesTr  info.txt  keypoints01Tr  keypoints02Tr  ThoraxCBCT_dataset.json
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> cd imagesTr/
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT/imagesTr> ls
ThoraxCBCT_0000_0000.nii.gz  ThoraxCBCT_0007_0000.nii.gz
ThoraxCBCT_0000_0001.nii.gz  ThoraxCBCT_0007_0001.nii.gz
ThoraxCBCT_0000_0002.nii.gz  ThoraxCBCT_0007_0002.nii.gz
ThoraxCBCT_0001_0000.nii.gz  ThoraxCBCT_0008_0000.nii.gz
ThoraxCBCT_0001_0001.nii.gz  ThoraxCBCT_0008_0001.nii.gz
ThoraxCBCT_0001_0002.nii.gz  ThoraxCBCT_0008_0002.nii.gz
ThoraxCBCT_0002_0000.nii.gz  ThoraxCBCT_0009_0000.nii.gz
ThoraxCBCT_0002_0001.nii.gz  ThoraxCBCT_0009_0001.nii.gz
ThoraxCBCT_0002_0002.nii.gz  ThoraxCBCT_0009_0002.nii.gz
ThoraxCBCT_0003_0000.nii.gz  ThoraxCBCT_0010_0000.nii.gz
ThoraxCBCT_0003_0001.nii.gz  ThoraxCBCT_0010_0001.nii.gz
ThoraxCBCT_0003_0002.nii.gz  ThoraxCBCT_0010_0002.nii.gz
ThoraxCBCT_0004_0000.nii.gz  ThoraxCBCT_0011_0000.nii.gz
ThoraxCBCT_0004_0001.nii.gz  ThoraxCBCT_0011_0001.nii.gz
ThoraxCBCT_0004_0002.nii.gz  ThoraxCBCT_0011_0002.nii.gz
ThoraxCBCT_0005_0000.nii.gz  ThoraxCBCT_0012_0000.nii.gz
ThoraxCBCT_0005_0001.nii.gz  ThoraxCBCT_0012_0001.nii.gz
ThoraxCBCT_0005_0002.nii.gz  ThoraxCBCT_0012_0002.nii.gz
ThoraxCBCT_0006_0000.nii.gz  ThoraxCBCT_0013_0000.nii.gz
ThoraxCBCT_0006_0001.nii.gz  ThoraxCBCT_0013_0001.nii.gz
ThoraxCBCT_0006_0002.nii.gz  ThoraxCBCT_0013_0002.nii.gz
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT/imagesTr> cd ..
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> ls 
imagesTr  info.txt  keypoints01Tr  keypoints02Tr  ThoraxCBCT_dataset.json
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> cat info.txt 
With the release of the ThoraxCBCT dataset the registration problem of image guided radiation therapy (IGRT) between pre-therapeutic fan beam CT (FBThe released image data is part of the 4D-Lung dataset from The Cancer Imaging Archive which contains four-dimensional lung images acquired during radiochemotherapy of locally-advanced, non-small cell lung cancer (NSCLC) patients https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageIdFor each patient one FBCT prior to therapy in maximum inspiration is provided which is denoted by the filename ending _0000.nii.gz. Further, there are two CBCTs for each patient, one at the beginning of therapy (_0001.nii.gz) and one at the end of therapy (_0002.nii.gz). Both are acquired in maxThe task is to find one solution for the registration of two pairs of imag1. The planning FBCT (inspiration) prior to therapy and the low dose CBCT at the beginning of therapy (expiration), which are acquired at similar ti2. The planning FBCT (inspiration) prior to therapy and the low dose CBCT at the end of therapy (expiration), where longer periods of time (usually The challenge in both pairs of images is the registration of images from the two different modalities FBCT and CBCT and also the shift in breathing phases between maximum inspiration and expiration. In the second subtask there is an additional challenge with the time shift between the planning CT at the beginning of therapy and the follow up CBCT at the end of therapyThe released dataset includes training images of 11 patients resulting in 11 FBCT and 22 CBCT images, and validation images of 3 patients with 3 FBCT and 6 CBCT images. The images are paired as described above with two image pairs per patient, resulting in 22 image pairs for training and 6 image pairs for validation. Additionally, Foerstner keypoints are provided in two folders: �keypoints01Tr� and �keypoints02Tr�. This has changed with regard to previous L2R releases, due to the FBCT images being used in two image pairings. The keypoints were created based on displacements resulting from deedsBCV registration for every image pair (https://github.com/mattiaspaul/deedsBCV), which leads to two different keypoint files for every FBCT image. The keypoints for the first subtask (FBCT, CBCT beginning of therapy) are stored in �keypoints01Tr� with the filename ending _0000.csv denoting the keypoints for the FBCT (_0000.nii.gz) and _0001.csv for the CBCT (Along this denotation the keypoints for the second subtask are stored in �keypoints02Tr� with again the FBCT keypoint filename ending being _0000.csv and _0002.csv for the CBCT at the end of therapy (_0002.nii.gz). The correspondences for the validation cases will not be published yet. 

In contrast to previous tasks, the focus of this task is not mainly on the lung, but also on other thoracic organs, especially organs at risk with regard to radiotherapy, which is why the keypoints were determined not only in the lung, but in the entire trunk region.

The test data will be manually annotated with multiple masks (lung(lobes), tumor, organs at risk (heart, spinal cord, esophagus, etc.)) as well as anatomical landmarks such as airway bifurcations, top of aortic arch, apex cordis and bone structures such as the sternum, ribs, clavicles and vertebFor details on the image acquisition (scanner details etc.) please see https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=21267414. All images are converted to nifti, resampled and cropped to the region of interest (thorax) resulting in an image size of 390x280x300 with a spacing of 1x1x1mm. 

(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> 
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> ls keypoints01Tr
ThoraxCBCT_0000_0000.csv  ThoraxCBCT_0005_0001.csv
ThoraxCBCT_0000_0001.csv  ThoraxCBCT_0006_0000.csv
ThoraxCBCT_0001_0000.csv  ThoraxCBCT_0006_0001.csv
ThoraxCBCT_0001_0001.csv  ThoraxCBCT_0007_0000.csv
ThoraxCBCT_0002_0000.csv  ThoraxCBCT_0007_0001.csv
ThoraxCBCT_0002_0001.csv  ThoraxCBCT_0008_0000.csv
ThoraxCBCT_0003_0000.csv  ThoraxCBCT_0008_0001.csv
ThoraxCBCT_0003_0001.csv  ThoraxCBCT_0009_0000.csv
ThoraxCBCT_0004_0000.csv  ThoraxCBCT_0009_0001.csv
ThoraxCBCT_0004_0001.csv  ThoraxCBCT_0010_0000.csv
ThoraxCBCT_0005_0000.csv  ThoraxCBCT_0010_0001.csv
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCB
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> cat keypoints01Tr/ThoraxCBCT_0000_0000.csv 
....373.081,179.940,91.080
368.042,180.007,57.308
371.531,183.965,25.345
372.277,184.561,11.033
378.824,187.972,132.852
375.239,193.618,141.378
367.853,189.450,216.637
370.309,194.953,24.070
367.651,193.868,223.003
364.102,201.772,48.432
370.792,200.881,137.183
360.588,207.183,75.090
369.246,205.538,138.225
358.175,212.870,53.935....

(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> cd keypoints02Tr
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT/keypoints02Tr> ls
ThoraxCBCT_0000_0000.csv  ThoraxCBCT_0005_0002.csv
ThoraxCBCT_0000_0002.csv  ThoraxCBCT_0006_0000.csv
ThoraxCBCT_0001_0000.csv  ThoraxCBCT_0006_0002.csv
ThoraxCBCT_0001_0002.csv  ThoraxCBCT_0007_0000.csv
ThoraxCBCT_0002_0000.csv  ThoraxCBCT_0007_0002.csv
ThoraxCBCT_0002_0002.csv  ThoraxCBCT_0008_0000.csv
ThoraxCBCT_0003_0000.csv  ThoraxCBCT_0008_0002.csv
ThoraxCBCT_0003_0002.csv  ThoraxCBCT_0009_0000.csv
ThoraxCBCT_0004_0000.csv  ThoraxCBCT_0009_0002.csv
ThoraxCBCT_0004_0002.csv  ThoraxCBCT_0010_0000.csv
ThoraxCBCT_0005_0000.csv  ThoraxCBCT_0010_0002.csv
(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT/keypoints02Tr> 

a(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCBCT/ThoraxCBCT> cat ThoraxCBCT_dataset.json 
{
    "name": "ThoraxCBCT",
    "release": "2",
    "description": "Training/Validation Subset ThoraxCBCT of Learn2Reg Dataset. Please see https://learn2reg.grand-challenge.org/ for more information. ",
    "licence": "THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE COMMONS PUBLIC LICENSE (\"CCPL\" OR \"LICENSE\"). THE WORK IS PROTECTED BY COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED. BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY BE CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.",
    "reference": "Hugo, G. D., Weiss, E., Sleeman, W. C., Balik, S., Keall, P. J., Lu, J., & Williamson, J. F. (2016). Data from 4D Lung Imaging of NSCLC Patients (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE; Hugo, G. D., Weiss, E., Sleeman, W. C., Balik, S., Keall, P. J., Lu, J., & Williamson, J. F. (2017). A longitudinal four-dimensional computed tomography and cone beam computed tomography dataset for image-guided radiation therapy research in lung cancer. In Medical Physics (Vol. 44, Issue 2, pp. 762\u2013771). Wiley. https://doi.org/10.1002/mp.12059; Balik, S., Weiss, E., Jan, N., Roman, N., Sleeman, W. C., Fatyga, M., Christensen, G. E., Zhang, C., Murphy, M. J., Lu, J., Keall, P., Williamson, J. F., & Hugo, G. D. (2013). Evaluation of 4-dimensional Computed Tomography to 4-dimensional Cone-Beam Computed Tomography Deformable Image Registration for Lung Cancer Adaptive Radiation Therapy. In International Journal of Radiation Oncology*Biology*Physics (Vol. 86, Issue 2, pp. 372\u2013379). Elsevier BV. PMCID: PMC3647023. https://doi.org/10.1016/j.ijrobp.2012.12.023; Roman, N. O., Shepherd, W., Mukhopadhyay, N., Hugo, G. D., & Weiss, E. (2012). Interfractional Positional Variability of Fiducial Markers and Primary Tumors in Locally Advanced Non-Small-Cell Lung Cancer During Audiovisual Biofeedback Radiotherapy. In International Journal of Radiation Oncology*Biology*Physics (Vol. 83, Issue 5, pp. 1566\u20131572). Elsevier BV. https://doi.org/10.1016/j.ijrobp.2011.10.051; Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7",
    "pairings": "paired",
    "provided_data": {
        "0": [
            "image",
            "keypoints01",
            "keypoints02"
        ]
    },
    "registration_direction": {
        "fixed": 1,
        "moving": 0
    },
    "modality": {
        "0": "FBCT",
        "1": "CBCT",
        "2": "CBCT"
    },
    "img_shift": {
        "fixed": "followup",
        "moving": "baseline"
    },
    "labels": {},
    "tensorImageSize": {
        "0": "3D"
    },
    "tensorImageShape": {
        "0": [
            390,
            280,
            300
        ]
    },
    "numTraining": 42,
    "numTest": 0,
    "training": [
        {
            "image": "./imagesTr/ThoraxCBCT_0000_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0000_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0000_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0000_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0000_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0000_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0000_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0001_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0001_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0001_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0001_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0001_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0001_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0001_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0002_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0002_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0002_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0002_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0002_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0002_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0002_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0003_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0003_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0003_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0003_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0003_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0003_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0003_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0004_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0004_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0004_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0004_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0004_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0004_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0004_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0005_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0005_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0005_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0005_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0005_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0005_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0005_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0006_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0006_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0006_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0006_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0006_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0006_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0006_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0007_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0007_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0007_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0007_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0007_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0007_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0007_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0008_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0008_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0008_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0008_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0008_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0008_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0008_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0009_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0009_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0009_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0009_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0009_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0009_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0009_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0010_0000.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0010_0000.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0010_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0010_0001.nii.gz",
            "keypoints01": "./keypoints01Tr/ThoraxCBCT_0010_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0010_0002.nii.gz",
            "keypoints02": "./keypoints02Tr/ThoraxCBCT_0010_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0011_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0011_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0011_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0012_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0012_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0012_0002.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0013_0000.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0013_0001.nii.gz"
        },
        {
            "image": "./imagesTr/ThoraxCBCT_0013_0002.nii.gz"
        }
    ],
    "test": [],
    "numPairedTraining": 22,
    "training_paired_images": [
        {
            "fixed": "./imagesTr/ThoraxCBCT_0000_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0000_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0001_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0001_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0002_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0002_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0003_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0003_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0004_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0004_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0005_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0005_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0006_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0006_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0007_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0007_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0008_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0008_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0009_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0009_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0010_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0010_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0000_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0000_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0001_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0001_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0002_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0002_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0003_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0003_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0004_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0004_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0005_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0005_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0006_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0006_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0007_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0007_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0008_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0008_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0009_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0009_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0010_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0010_0000.nii.gz"
        }
    ],
    "numPairedTest": 0,
    "test_paired_images": [],
    "numRegistration_val": 6,
    "registration_val": [
        {
            "fixed": "./imagesTr/ThoraxCBCT_0011_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0011_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0012_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0012_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0013_0001.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0013_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0011_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0011_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0012_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0012_0000.nii.gz"
        },
        {
            "fixed": "./imagesTr/ThoraxCBCT_0013_0002.nii.gz",
            "moving": "./imagesTr/ThoraxCBCT_0013_0000.nii.gz"
        }
    ],
    "numRegistration_test": 0,
    "registration_test": []
}(base) almik@raven03:/nexus/posix0/MBR-neuralsystems/alim/regdata/ThoraxCCT/ThoraxCBCT> 
are these enough? 


