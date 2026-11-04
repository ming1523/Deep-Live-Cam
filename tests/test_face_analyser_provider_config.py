import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


def _import_stubs():
    """Build the stub modules needed to import ``modules.face_analyser``.

    Third-party packages are only stubbed when they are not importable, so a
    real numpy/cv2 keeps being used when it is installed.
    """
    stubs = {
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
    }
    stubs = {name: module for name, module in stubs.items() if name not in sys.modules}
    stubs.update({
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
    })
    return stubs


@contextmanager
def _load_face_analyser():
    """Import ``modules.face_analyser`` with its heavy imports stubbed out.

    ``sys.modules`` is restored when the block exits, so tests running later
    still see the real ``modules.processors.frame._onnx_enhancer`` module with
    its normal helpers such as ``create_onnx_session``.
    """
    with patch.dict(sys.modules, _import_stubs(), clear=False):
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

    def test_restores_onnx_enhancer_module_after_test(self):
        enhancer_name = "modules.processors.frame._onnx_enhancer"
        missing = object()
        previous = sys.modules.get(enhancer_name, missing)

        with _load_face_analyser():
            self.assertIn(enhancer_name, sys.modules)

        if previous is missing:
            self.assertNotIn(enhancer_name, sys.modules)
        else:
            self.assertIs(sys.modules[enhancer_name], previous)


if __name__ == "__main__":
    unittest.main()
