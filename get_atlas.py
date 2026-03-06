from nilearn.datasets import fetch_atlas_schaefer_2018

atlas = fetch_atlas_schaefer_2018(
    n_rois=200,         # 200 regions is a good starting point
    yeo_networks=7,     # 7-network version (matches your theory notes)
    resolution_mm=2     # 2 mm (standard MNI)
)

print("Atlas NIfTI:", atlas.maps)     # path to the .nii.gz
print("Label file:", atlas.labels)    # path to the .csv with ROI names