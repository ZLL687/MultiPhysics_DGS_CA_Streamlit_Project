# Deployment notes

## Option A: Upload model package to GitHub

If `ea_pgcc_dgs_ca_model_package.pkl` is small enough, place it here:

```text
models/ea_pgcc_dgs_ca_model_package.pkl
```

Then remove this line from `.gitignore`:

```text
models/*.pkl
```

## Option B: Manual upload in Streamlit

Keep `.gitignore` unchanged. After the app starts, check:

```text
Use manual model package upload
```

Then upload:

```text
ea_pgcc_dgs_ca_model_package.pkl
```

## Option C: Google Drive download

Upload the model package to Google Drive, copy the file ID, and set this variable in `streamlitapp.py`:

```python
MODEL_PACKAGE_GDRIVE_ID = "your_google_drive_file_id"
```

The app will attempt to download the model package to:

```text
models/ea_pgcc_dgs_ca_model_package.pkl
```
