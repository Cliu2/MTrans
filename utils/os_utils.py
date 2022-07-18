from os import path, listdir, makedirs

def verify_and_create_outdir(outdir):
    if not path.exists(outdir):
        makedirs(outdir)
    else:
        if not path.isdir(outdir) or len(listdir(outdir))>0:
            raise RuntimeError(f"Output directory is not empty: {outdir}")
