"""End-to-end CLI: NIfTI in -> NIfTI out, geometry preserved."""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from pyonlm import cli


def _write_nifti(path, data, affine):
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, str(path))


def test_cli_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n = 24
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2)
    clean = np.where(r < n / 3, 120.0, 0.0)
    vol = clean + rng.normal(0.0, 5.0, size=clean.shape)
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    affine[:3, 3] = [10.0, -20.0, 30.0]

    inp = tmp_path / "in.nii.gz"
    outp = tmp_path / "out.nii.gz"
    _write_nifti(inp, vol, affine)

    rc = cli.main([str(inp), str(outp), "-mt", "6", "-sigma", "0", "-beta", "1"])
    assert rc == 0
    assert outp.exists()

    out_img = nib.load(str(outp))
    assert np.allclose(out_img.affine, affine)
    assert out_img.shape == (n, n, n)
    out = out_img.get_fdata()
    # denoising should reduce variance inside the (uniform) object interior
    interior = r < n / 4
    assert out[interior].std() < vol[interior].std()


def test_cli_clobber_guard(tmp_path):
    affine = np.eye(4)
    vol = np.ones((16, 16, 16), dtype=np.float32) * 50.0
    inp = tmp_path / "in.nii.gz"
    outp = tmp_path / "out.nii.gz"
    _write_nifti(inp, vol, affine)
    outp.write_text("existing")
    rc = cli.main([str(inp), str(outp), "-sigma", "2"])
    assert rc == 1  # refuses to overwrite without -clobber
