# NASA Battery Data

This directory holds the raw `.mat` files from the NASA Battery Prognostics
dataset. They are NOT included in the repo (total size ~600 MB).

## To populate this directory:

```bash
bash ../scripts/download_nasa.sh
```

This downloads the official zip from the PHM S3 bucket
(`https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip`) and
extracts all 38 `.mat` files into `5. Battery Data Set/<sub-campaign>/`.

## Expected structure after download

```
data/nasa/
└── 5. Battery Data Set/
    ├── 1. BatteryAgingARC-FY08Q4/        # B0005, B0006, B0007, B0018
    ├── 2. BatteryAgingARC_25_26_27_28_P1/ # B0025, B0026, B0027, B0028
    ├── 3. BatteryAgingARC_25-44/          # B0025–B0044 (re-cycles B0025–B0028)
    ├── 4. BatteryAgingARC_45_46_47_48/    # B0045–B0048
    ├── 5. BatteryAgingARC_49_50_51_52/    # B0049–B0052
    └── 6. BatteryAgingARC_53_54_55_56/    # B0053–B0056
```

## Citation

B. Saha and K. Goebel, "Battery data set," NASA Ames Prognostics Data
Repository, 2007.
