import importlib
import sys
import types
import unittest
from unittest.mock import patch

# Third-party modules are only stubbed when they are not imported yet, so an
# installed package always takes precedence over the lightweight stub.
_THIRD_PARTY_MODULES = ("insightface", "cv2", "numpy", "tqdm")


def _import_stubs():
    """Return the ``sys.modules`` entries required to import ``modules.face_analyser``."""
    return {
        "insightface": types.SimpleNamespace(app=types.SimpleNamespace(FaceAnalysis=object)),
        "cv2": types.SimpleNamespace(
            IMREAD_COLOR=1,
            imread=lambda *_args, **_kwargs: None,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                types.SimpleNamespace(tofile=lambda *_a, **_k: None),
            ),
        ),
        "numpy": types.SimpleNamespace(uint8=object, fromfile=lambda *_args, **_kwargs: b""),
        "tqdm": types.SimpleNamespace(tqdm=lambda iterable, **_kwargs: iterable),
        "modules.typing": types.SimpleNamespace(Frame=object),
        "modules.cluster_analysis": types.SimpleNamespace(
            find_cluster_centroids=lambda *args, **kwargs: [],
            find_closest_centroid=lambda *args, **kwargs: (0, None),
        ),
        "modules.utilities": types.SimpleNamespace(
            get_temp_directory_path=lambda path: path,
            create_temp=lambda path: None,
            extract_frames=lambda path: None,
            clean_temp=lambda path: None,
            get_temp_frame_paths=lambda path: [],
        ),
        "modules.processors.frame._onnx_enhancer": types.SimpleNamespace(
            build_provider_config=lambda providers=None: list(
                providers
                if providers is not None
                else sys.modules["modules.globals"].execution_providers
            ),
        ),
    }


def _patched_imports():
    """Stub the imports of ``modules.face_analyser`` for the duration of a test.

    ``patch.dict`` restores ``sys.modules`` on exit, so the stubs never leak
    into tests that run later in the same process.  This matters for
    ``modules.processors.frame._onnx_enhancer``: the real enhancer modules
    import ``create_onnx_session`` from it, which the stub does not provide.
    """
    stubs = _import_stubs()
    for name in _THIRD_PARTY_MODULES:
        if name in sys.modules:
            del stubs[name]
    return patch.dict(sys.modules, stubs, clear=False)


def _load_face_analyser():
    sys.modules.pop("modules.face_analyser", None)
    return importlib.import_module("modules.face_analyser")


class FaceAnalyserProviderConfigTests(unittest.TestCase):
    def test_uses_cpu_provider_for_directml_face_analysis(self):
        with _patched_imports():
            face_analyser = _load_face_analyser()
            with patch.object(
                face_analyser.modules.globals,
                "execution_providers",
                ["DmlExecutionProvider"],
            ):
                self.assertEqual(
                    face_analyser.build_face_analyser_provider_config(),
                    ["CPUExecutionProvider"],
                )

    def test_preserves_existing_provider_config_for_non_directml(self):
        with _patched_imports():
            face_analyser = _load_face_analyser()
            with patch.object(
                face_analyser.modules.globals,
                "execution_providers",
                ["CUDAExecutionProvider"],
            ):
                self.assertEqual(
                    face_analyser.build_face_analyser_provider_config(),
                    ["CUDAExecutionProvider"],
                )


if __name__ == "__main__":
    unittest.main()
