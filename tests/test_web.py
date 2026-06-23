#!/usr/bin/env python3
"""FastAPI web-layer tests (TestClient).

Guards the web rework: typed request models (replacing untyped dict payloads), the
path-sandbox, and the concrete DoS/validation fixes. No server is started - starlette's
TestClient drives the ASGI app in-process.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multiseedgen")

from fastapi.testclient import TestClient  # noqa: E402

# Import the submodule (the package re-exports the FastAPI instance as `.app`).
_webapp = importlib.import_module("multiseedgen.web.app")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(_webapp.app)


def test_schema_endpoint(client):
    r = client.get("/api/schema")
    assert r.status_code == 200
    data = r.json()
    assert data["groups"] and "recommended" in data and "presets" in data


def test_index_served(client):
    assert client.get("/").status_code == 200


# --------------------------------------------------------------------------- #
# Request-model validation (formerly untyped dict payloads)                   #
# --------------------------------------------------------------------------- #
def test_start_run_requires_sources_and_out(client):
    r = client.post("/api/runs", json={"config": {}})
    assert r.status_code == 400
    assert "sources" in r.text and "out" in r.text


def test_start_run_rejects_malformed_body(client):
    # config must be an object, not a string -> 422 from the Pydantic model
    r = client.post("/api/runs", json={"config": "not-a-dict"})
    assert r.status_code == 422


def test_seg_check_requires_sources(client):
    r = client.post("/api/seg/check", json={"config": {}, "n": 4})
    assert r.status_code == 400
    assert "sources" in r.text


def test_config_save_requires_path(client):
    r = client.post("/api/config/save", json={"config": {"sources": "s", "out": "o"}})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Path sandbox + DoS fixes                                                    #
# --------------------------------------------------------------------------- #
def test_serve_file_rejects_traversal(client):
    r = client.get("/api/file", params={"path": "/etc/passwd"})
    assert r.status_code == 403


def test_fs_list_rejects_outside_sandbox(client):
    r = client.get("/api/fs/list", params={"path": "/etc"})
    assert r.status_code == 403


def test_dataset_image_rejects_bad_split(client, tmp_path):
    # split must be train|val; anything else is rejected (not silently coerced)
    r = client.get("/api/datasets/image", params={"path": str(tmp_path), "split": "../etc"})
    assert r.status_code in (400, 422)


def test_config_load_missing_file_is_400_not_500(client, tmp_path):
    r = client.get("/api/config/load", params={"path": str(tmp_path / "nope.yaml")})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Happy paths inside the sandbox (tmp_path lives under the allowed TMP base)   #
# --------------------------------------------------------------------------- #
def test_runs_list_returns_runs_collection(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert isinstance(r.json()["runs"], list)


def test_fs_list_respects_kind_filter(client, tmp_path):
    """The dir-picker (kind='dir') hides files; kind='file' reveals them. Both stay
    inside the sandbox and expose a climbable `parent`."""
    (tmp_path / "subdir").mkdir()
    (tmp_path / "note.txt").write_text("x")

    dirs = client.get("/api/fs/list", params={"path": str(tmp_path), "kind": "dir"}).json()
    names = {e["name"] for e in dirs["entries"]}
    assert "subdir" in names and "note.txt" not in names

    files = client.get("/api/fs/list", params={"path": str(tmp_path), "kind": "file"}).json()
    assert "note.txt" in {e["name"] for e in files["entries"]}


def test_serve_file_returns_content_with_media_type(client, tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    r = client.get("/api/file", params={"path": str(f)})
    assert r.status_code == 200
    assert r.content == b"hello world"
    assert r.headers["content-type"].startswith("text/plain")


def test_dataset_image_renders_annotated_png(client, tmp_path):
    import cv2
    import numpy as np

    img_dir = tmp_path / "train" / "images"
    lbl_dir = tmp_path / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "a.jpg"), np.full((32, 32, 3), 200, np.uint8))
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    r = client.get(
        "/api/datasets/image", params={"path": str(tmp_path), "split": "train", "idx": 0}
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["X-Total"] == "1"


def test_dataset_image_404_when_no_images(client, tmp_path):
    r = client.get("/api/datasets/image", params={"path": str(tmp_path), "split": "val"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Dataset download (new): zip a dataset / split, sandboxed                     #
# --------------------------------------------------------------------------- #
def _make_dataset(root):
    import cv2
    import numpy as np

    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        cv2.imwrite(str(root / split / "images" / "a.jpg"), np.full((16, 16, 3), 128, np.uint8))
        (root / split / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text("names: [seed]\n")
    (root / ".seg_cache").mkdir()  # dotdir must be excluded from the archive
    (root / ".seg_cache" / "junk.bin").write_bytes(b"\x00")


def test_dataset_download_zips_full_dataset(client, tmp_path):
    import io
    import zipfile

    ds = tmp_path / "mydataset"
    _make_dataset(ds)
    r = client.get("/api/datasets/download", params={"path": str(ds)})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "mydataset/data.yaml" in names
    assert "mydataset/train/images/a.jpg" in names
    assert "mydataset/val/images/a.jpg" in names
    assert not any(".seg_cache" in n for n in names)  # dotdirs pruned


def test_dataset_download_single_split(client, tmp_path):
    import io
    import zipfile

    ds = tmp_path / "ds2"
    _make_dataset(ds)
    r = client.get("/api/datasets/download", params={"path": str(ds), "split": "train"})
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert all(n.startswith("ds2/train/") for n in names)
    assert not any("/val/" in n for n in names)


def test_dataset_download_rejects_traversal(client):
    r = client.get("/api/datasets/download", params={"path": "/etc"})
    assert r.status_code == 403


def test_dataset_download_rejects_bad_split(client, tmp_path):
    ds = tmp_path / "ds3"
    _make_dataset(ds)
    r = client.get("/api/datasets/download", params={"path": str(ds), "split": "nope"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Seed upload (new): multipart, sandboxed, extension + content validated       #
# --------------------------------------------------------------------------- #
def _png_bytes():
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".png", np.full((8, 8, 3), 64, np.uint8))
    assert ok
    return buf.tobytes()


def test_upload_seeds_happy_path(client, tmp_path):
    dest = tmp_path / "seeds_in"
    files = [
        ("files", ("one.png", _png_bytes(), "image/png")),
        ("files", ("two.jpg", _png_bytes(), "image/jpeg")),
    ]
    r = client.post("/api/seeds/upload", data={"dest": str(dest)}, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert set(body["files"]) == {"one.png", "two.jpg"}
    assert (dest / "one.png").exists() and (dest / "two.jpg").exists()


def test_upload_seeds_rejects_outside_sandbox(client):
    files = [("files", ("x.png", _png_bytes(), "image/png"))]
    r = client.post("/api/seeds/upload", data={"dest": "/etc/evil"}, files=files)
    assert r.status_code == 403


def test_upload_seeds_rejects_bad_extension(client, tmp_path):
    files = [("files", ("evil.exe", b"MZ\x90\x00", "application/octet-stream"))]
    r = client.post("/api/seeds/upload", data={"dest": str(tmp_path / "s")}, files=files)
    assert r.status_code == 400
    assert "unsupported file type" in r.text


def test_upload_seeds_rejects_non_decodable_image(client, tmp_path):
    # right extension, but the bytes are not a real image -> content sniff rejects it
    files = [("files", ("fake.png", b"not really a png", "image/png"))]
    r = client.post("/api/seeds/upload", data={"dest": str(tmp_path / "s2")}, files=files)
    assert r.status_code == 400
    assert "decodable" in r.text


def test_upload_seeds_strips_path_components(client, tmp_path):
    # a malicious filename with traversal must be reduced to its base name
    dest = tmp_path / "s3"
    files = [("files", ("../../escape.png", _png_bytes(), "image/png"))]
    r = client.post("/api/seeds/upload", data={"dest": str(dest)}, files=files)
    assert r.status_code == 200
    assert (dest / "escape.png").exists()
    assert not (tmp_path / "escape.png").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
