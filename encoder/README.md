# Encoder Training

This folder contains the observation-to-initial-condition encoder training work.

Pipeline A is the Aardvark-style pipeline:

1. Read Aardvark/Hugging Face observation data.
2. Read WeatherBench2 ERA5 ground-truth targets on the `240x121` grid.
3. Convert each timestamp into a model-ready batch:

   ```python
   {
       "assimilation": {...},          # observation tensors
       "y_target": torch.Tensor,       # (121, 240, 24), normalized
   }
   ```

4. Train an Aardvark-compatible `ConvCNPWeather(..., decoder="vit_assimilation")`
   encoder.
5. Evaluate the predicted initial condition against ERA5 targets.

## Data Reality

The Hugging Face Aardvark dataset contains the observation sources for 2007-2019.
It does not include ERA5 targets or climatology. Those should come from
WeatherBench2:

- ERA5 targets:
  `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`
- ERA5 climatology:
  `gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr`

Your MacBook's 100 GB free space is enough for smoke tests and small subsets, but
not enough for the full Hugging Face observation corpus plus ERA5 targets.

## Recommended Debug Order

1. Smoke-test the trainer on the existing model-ready sample pickle.
2. Download/inspect one small observation source file or a small range if the
   remote format supports partial reads.
3. Build the Hugging Face NetCDF observation adapter.
4. Train Pipeline A on two months, e.g. 2007-01-02 through 2007-02-28.
5. Scale only after the shapes, timestamps, normalization, and loss are verified.

