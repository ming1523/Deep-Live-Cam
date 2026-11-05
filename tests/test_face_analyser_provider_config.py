import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


def _install_import_stubs():
    sys.modules.setdefault(
        "insightface",
        types.SimpleNamespace(app=types.SimpleNamespace(FaceAnalysis=object)),
    )
    sys.modules.setdefault(
        "cv2",
        types.SimpleNamespace(
            IMREAD_COLOR=1,
            imread=lambda *_args, **_kwargs: None,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                types.SimpleNamespace(tofile=lambda *_a, **_k: None),
            ),
        ),
    )
    sys.modules.setdefault(
        "numpy",
        types.SimpleNamespace(uint8=object, fromfile=lambda *_args, **_kwargs: b""),
    )
    sys.modules.setdefault(
        "tqdm",
        types.SimpleNamespace(tqdm=lambda iterable, **_kwargs: iterable),
    )
    sys.modules["modules.typing"] = types.SimpleNamespace(Frame=object)
    sys.modules["modules.cluster_analysis"] = types.SimpleNamespace(
        find_cluster_centroids=lambda *args, **kwargs: [],
        find_closest_centroid=lambda *args, **kwargs: (0, None),
    )
    sys.modules["modules.utilities"] = types.SimpleNamespace(
        get_temp_directory_path=lambda path: path,
        create_temp=lambda path: None,
        extract_frames=lambda path: None,
        clean_temp=lambda path: None,
        get_temp_frame_paths=lambda path: [],
    )
    sys.modules["modules.processors.frame._onnx_enhancer"] = types.SimpleNamespace(
        build_provider_config=lambda providers=None: list(
            providers
            if providers is not None
            else sys.modules["modules.globals"].execution_providers
        ),
    )


@contextmanager
def _load_face_analyser():
    """Import ``modules.face_analyser`` on top of stubbed dependencies.

    The stubs only live for the duration of the block: ``sys.modules`` is
    restored on exit, so tests running afterwards still get the real modules
    (for example ``modules.processors.frame._onnx_enhancer`` with
    ``create_onnx_session``) instead of the reduced stubs used here.
    """
    with patch.dict(sys.modules):
        _install_import_stubs()
        sys.modules.pop("modules.face_analyser", None)
        yield importlib.import_module("modules.face_analyser")


class FaceAnalyserProviderConfigTests(unittest.TestCase):
    def test_uses_cpu_provider_for_directml_face_analysis(self):
        with _load_face_analyser() as face_analyser:
            face_analyser.modules.globals.execution_providers = ["DmlExecutionProvider"]

            self.assertEqual(
                face_analyser.build_face_analyser_provider_config(),
                ["CPUExecutionProvider"],
            )

    def test_preserves_existing_provider_config_for_non_directml(self):
        with _load_face_analyser() as face_analyser:
            face_analyser.modules.globals.execution_providers = ["CUDAExecutionProvider"]

            self.assertEqual(
                face_analyser.build_face_analyser_provider_config(),
                ["CUDAExecutionProvider"],
            )

    def test_import_stubs_do_not_leak_into_sys_modules(self):
        with _load_face_analyser():
            pass

        enhancer = sys.modules.get("modules.processors.frame._onnx_enhancer")
        self.assertFalse(
            enhancer is not None and not hasattr(enhancer, "create_onnx_session"),
            "stubbed _onnx_enhancer module leaked into sys.modules",
        )


if __name__ == "__main__":
    unittest.main()
