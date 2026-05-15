Abstract
The International Cardiac Arrest REsearch consortium (I-CARE) Database includes baseline clinical information and continuous electroencephalogram (EEG) and electrocardiogram (ECG) recordings from comatose patients following cardiac arrest. The patients were admitted to an intensive care unit (ICU) in one of seven academic hospitals in the U.S. and Europe and monitored for several hours to several days. The long-term neurological function of the patients was determined using the Cerebral Performance Category scale.

Background
More than 6 million cardiac arrests happen every year worldwide, with survival rates ranging from 1% to 10% depending on geographic location [1]. Severe brain injury is the main determinant of poor outcome for patients surviving cardiac arrest resuscitation [1,2]. Most patients surviving to ICU admission will be comatose, and 50% to 80% will have life-sustaining therapies withdrawn due to a perceived poor neurological prognosis [3].

Brain monitoring with EEG aims to reduce the subjectivity in neurologic prognostication following cardiac arrest [4-9]. Clinical neurophysiologists have identified numerous patterns of brain activity that help to predict prognosis following cardiac arrest, including the presences of reduced voltage, burst suppression (alternating periods of high and low voltage), seizures, and a variety of seizure-like patterns [8]. The evolution of EEG patterns over time may provide additional predictive information [6,7]. However, qualitative interpretation of continuous EEG is laborious, expensive, and requires review from neurologists with advanced training in neurophysiology who are unavailable in most medical centers.

Automated analysis of continuous EEG and other data has the potential to improve prognostic accuracy and to increase access to brain monitoring where experts are not readily available [6,7]. However, the datasets used in most studies typically only have small numbers of patients (<100) from single hospitals, which are unsuitable for deployment of several types of machine learning methods for EEG data analysis. To overcome this limitation the International Cardiac Arrest REsearch consortium (I-CARE) assembled a large collection of clinical, EEG, and ECG data with neurologic outcomes from comatose patients following cardiac arrest. The I-CARE dataset includes seven hospitals from the United States and Europe.

Methods
The database originates from seven academic hospitals in the U.S. and Europe led by investigators part of the International Cardiac Arrest REsearch consortium (I-CARE) [10].

Rijnstate Hospital, Arnhem, The Netherlands (Jeannette Hofmeijer).
Medisch Spectrum Twente, Enschede, The Netherlands (Barry J. Ruijter, Marleen C. Tjepkema-Cloostermans, Michel J. A. M. van Putten).
Erasme Hospital, Brussels, Belgium (Nicolas Gaspard).
Massachusetts General Hospital, Boston, Massachusetts, USA (Edilberto Amorim, Wei-Long Zheng, Mohammad Ghassemi, and M. Brandon Westover).
Brigham and Women’s Hospital, Boston, Massachusetts, USA (Jong Woo Lee).
Beth Israel Deaconess Medical Center, Boston, Massachusetts, USA (Susan T. Herman).
Yale New Haven Hospital, New Haven, Connecticut, USA (Adithya Sivaraju).
This database consists of clinical, EEG, and ECG data from adult patients with out-of-hospital or in-hospital cardiac arrest who had return of heart function (i.e., return of spontaneous circulation [ROSC]) but remained comatose - defined as the inability to follow verbal commands and a Glasgow Coma Score inferior or equal to 8.

The initial database release contains data for over 32,712 hours of data in 80,809 recording segments from 607 patients - this is the public training set for the George B. Moody PhysioNet Challenge 2023. This database release does not contain data from the remaining 413 patients that we are retaining as the hidden validation and test sets for the Challenge.

All patients were admitted to an ICU and had their brain activity monitored with continuous EEG. Monitoring was typically started within hours of cardiac arrest and continued for several hours to several days depending on the patient's condition, so the recording start time and duration vary from patient to patient. This database includes EEG data and, when possible, ECG data for each patient. This project contains the part of the database that we have shared as a public training set for the PhysioNet Challenge 2023; the remainder of the database has been retained as private validation and test sets for the Challenge. Data from one hospital system were omitted from the training and validation sets to assess generalizability to unseen data.

Clinical Data
Patient information recorded at the time of admission (age, sex, and a hospital identifier), location of arrest (out or in-hospital), type of cardiac rhythm recorded at the time of resuscitation (shockable rhythms include ventricular fibrillation or ventricular tachycardia and non-shockable rhythms include asystole and pulseless electrical activity), and the time between cardiac arrest and ROSC. Patient temperature after cardiac arrest is controlled using a closed-loop feedback device (TTM) in most patients unless there are contraindications such as severe and difficult to control hypotension or delay in hospital admission. For patients undergoing TTM, the temperature level can be controlled at either 36 or 33 degrees Celsius.

Neurological Prognostication and Outcome Assessment
All participating hospitals have protocols for multimodal neurological prognostication that follow international guideline recommendations. Formal neurological prognostication is deferred until the normothermia phase, and confounding from sedatives can be minimized.

Patient Outcomes
Clinical outcome was determined prospectively in two centers by phone interview (at 6 months from ROSC), and at the remaining five hospitals retrospectively through chart review (at 3-6 months from ROSC). Neurological function was determined using the best Cerebral Performance Category (CPC) scale [11]. CPC is an ordinal scale ranging from 1 to 5, ranging from good neurological function to death.

De-identification
Clinical and EEG data were de-identified. Patients with age above 89 years old are listed with age "90". EEG timestamps are organized based on the time elapsed since ROSC. The hospital identifiers do not identify the hospital name.

Data Description
EEG Data
All EEG signal data are provided in WFDB format, with the signal data are stored in MATLAB MAT files (MAT v4 format). For example, the binary signal file 0284_001_004_EEG.mat contains the first segment of the EEG signal data, starting at 4 hours, 7 minutes, and 23 seconds after cardiac arrest and ending at 4 hours, 59 minutes, and 59 seconds after cardiac arrest, for patient 0284 of the I-CARE patient cohort. The plain text header file 0284_001_004_EEG.hea describes the contents of this signal file as well as the start time, stop time, and utility frequency (i.e., powerline frequency or mains frequency) for the data.

When possible, the channel names have been standardized between and within different hospitals. Different channels are available for different hospitals and different patients, including those from the same hospital. Even when a channel has been provided, it may be disconnected or noisy. The channels are organized into an EEG group, an ECG group, a reference (REF) group, and an other (OTHER) group:

EEG: Fp1, Fp2, F7, F8, F3, F4, T3, T4, C3, C4, T5, T6, P3, P4, O1, O2, Fz, Cz, Pz, Fpz, Oz, F9
ECG: ECG, ECG1, ECG2, ECGL, ECGR
REF: RAT1, RAT2, REF, C2, A1, A2, BIP1, BIP2, BIP3, BIP4, Cb2, M1, M2, In1-Ref2, In1-Ref3
OTHER: SpO2, EMG1, EMG2, EMG3, LAT1, LAT2, LOC, ROC, LEG1, LEG2
The recordings were segmented so that each segment ends at the hour, or the end of the recording, whichever occurs first. Noisy recordings with artifacts were intentionally preserved [12].

Clinical Data and Patient Outcome
The following clinical data is contained in each .txt file: 

Age (in years): Number
Sex: Male, Female
Hospital: A, B, C, D, E, F
ROSC (return of spontaneous circulation, in minutes): Time from cardiac arrest to return of spontaneous circulation
OHCA (out-of-hospital cardiac arrest): True = out of hospital cardiac arrest, False = in-hospital cardiac arrest
Shockable Rhythm: True = shockable rhythm, False = non-shockable rhythm
TTM (targeted temperature management; in Celsius): 33, 36, or NaN for no TTM
Outcome: Good (CPC score of 1-2), Poor (CPC score of 3-5)
CPC: Cerebral Performance Category (CPC) score (ordinal scale 1-5)
CPC = 1: good neurological function and independent for activities of daily living 
CPC = 2: moderate neurological disability but independent for activities of daily living
CPC = 3: severe neurological disability
CPC = 4: unresponsive wakefulness syndrome [previously known as vegetative state] 
CPC = 5: dead. 
We have grouped CPC scores in two categories: 

“Good outcome”: CPC = 1 or 2
“Poor outcome”: CPC = 3, 4, or 5
Usage Notes
These data were used as training data for the George B. Moody PhysioNet Challenge 2023 [13]. These data are in a WFDB-compatible format, and WFDB packages can be used to read them. We have implemented example prediction algorithms in MATLAB and Python that read the data:

MATLAB example at [14].
Python example at [15].
Release Notes
v2.1: The I-CARE Database v2.1 was released in December 2023. It add "nu" units to the ADC gain in the WFDB headers to clarify that the recordings do not have units, moves values from the ADC zero field to the baseline field in the WFDB headers, and computes 16-bit signed checksums in the WFDB headers.

v2.0: The I-CARE Database v2.0 was released on June 16, 2023. It changes from a sequential montage representation of the EEG recordings to a referential montage representation, adds additional EEG and non-EEG channels, and replaces 5-minute hourly time windows with full recordings.

v1.0: The I-CARE Database v1.0 was released on February 21, 2023.