import importlib
import sys
import types
import unittest
from contextlib import contextmanager


# Modules that are replaced by stubs while these tests run.  They have to be
# restored afterwards, otherwise tests that run later and import them get
# the stub instead of the real module (the ONNX enhancer stub, for instance,
# is missing ``create_onnx_session``).
_STUBBED_MODULE_NAMES = (
    "modules.typing",
    "modules.cluster_analysis",
    "modules.utilities",
    "modules.processors.frame._onnx_enhancer",
)

_MISSING = object()


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
def _stubbed_imports():
    """Install the import stubs for the duration of the block.

    ``build_face_analyser_provider_config`` imports the ONNX enhancer lazily,
    so the stub has to stay in place while the test runs.  The replaced
    modules are restored when the block ends, which keeps the stubs from
    leaking into the tests that run afterwards.
    """
    previous = {
        name: sys.modules.get(name, _MISSING) for name in _STUBBED_MODULE_NAMES
    }
    try:
        _install_import_stubs()
        yield
    finally:
        for name, original in previous.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        sys.modules.pop("modules.face_analyser", None)


def _load_face_analyser():
    sys.modules.pop("modules.face_analyser", None)
    return importlib.import_module("modules.face_analyser")


class FaceAnalyserProviderConfigTests(unittest.TestCase):
    def test_uses_cpu_provider_for_directml_face_analysis(self):
        with _stubbed_imports():
            face_analyser = _load_face_analyser()
            face_analyser.modules.globals.execution_providers = ["DmlExecutionProvider"]

            self.assertEqual(
                face_analyser.build_face_analyser_provider_config(),
                ["CPUExecutionProvider"],
            )

    def test_preserves_existing_provider_config_for_non_directml(self):
        with _stubbed_imports():
            face_analyser = _load_face_analyser()
            face_analyser.modules.globals.execution_providers = ["CUDAExecutionProvider"]

            self.assertEqual(
                face_analyser.build_face_analyser_provider_config(),
                ["CUDAExecutionProvider"],
            )

    def test_import_stubs_are_restored_afterwards(self):
        original = sys.modules.get(
            "modules.processors.frame._onnx_enhancer", _MISSING
        )

        with _stubbed_imports():
            stub = sys.modules["modules.processors.frame._onnx_enhancer"]
            self.assertTrue(hasattr(stub, "build_provider_config"))

        self.assertIs(
            sys.modules.get("modules.processors.frame._onnx_enhancer", _MISSING),
            original,
        )


if __name__ == "__main__":
    unittest.main()
