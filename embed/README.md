These scripts are designed to extract PanEcho-based video embeddings from a large directory of echo studies. 

1. `python 01_setup_folders.py [path to echo source] --out_dir [where embeddings will be stored]`
    - this creates a folder (`out_dir`), with 3 subdirectories, and populates `out_dir/incomplete/` with one csv per echo study folder (which contains the folder path).
2. `python 02_make_batches.py --out_dir [where embeddings will be stored] --batch_size [int]`
    - this creates on csv per batch in `out_dir/batches`
3. To process a batch, run `python process_batch.py --batch_num [number]`
    - Each job will process a batch, moving files from `/Incomplete/` to `/Complete/` as it finishes. 
    - if `--train_transforms` is specified, the samples in the batch will have random pertubations applied to improve training robustness.

An example script to process a single batch might look like:

```bash
src_dir="/path/to/echo/source/"
out_dir="./processed_echos"
python 01_setup_folders.py ${src_dir} --out_dir ${out_dir}
python 02_make_batches.py --out_dir ${out_dir}
python process_batch.py --batch_num 1 --out_dir ${out_dir}
```