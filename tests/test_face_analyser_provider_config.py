import importlib
import sys
import types
import unittest
from contextlib import ExitStack, contextmanager


_MISSING = object()


def _import_stubs():
    """Stub modules that ``modules.face_analyser`` needs to be importable."""
    return {
        "insightface": types.SimpleNamespace(
            app=types.SimpleNamespace(FaceAnalysis=object)
        ),
        "cv2": types.SimpleNamespace(
            IMREAD_COLOR=1,
            imread=lambda *_args, **_kwargs: None,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                types.SimpleNamespace(tofile=lambda *_a, **_k: None),
            ),
        ),
        "numpy": types.SimpleNamespace(
            uint8=object, fromfile=lambda *_args, **_kwargs: b""
        ),
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


@contextmanager
def _scoped_import_stubs():
    """Install the import stubs for the duration of a test.

    ``modules.processors.frame._onnx_enhancer`` is a real module of this
    repository, so the stub that replaces it must not outlive the test:
    tests running later in the same interpreter would otherwise import the
    stub instead of the module with its real functions (e.g.
    ``create_onnx_session``).  Every ``sys.modules`` entry touched here is
    restored on exit.
    """
    previous = {}
    for name, stub in _import_stubs().items():
        previous[name] = sys.modules.get(name, _MISSING)
        # Third-party packages are only stubbed when they are not installed;
        # repository modules are stubbed for the whole scope and restored.
        if previous[name] is _MISSING or name.startswith("modules."):
            sys.modules[name] = stub

    # ``modules.face_analyser`` is imported against the stubs, so it must not
    # stay cached in ``sys.modules`` once the stubs are gone.
    previous["modules.face_analyser"] = sys.modules.get(
        "modules.face_analyser", _MISSING
    )
    sys.modules.pop("modules.face_analyser", None)

    try:
        yield
    finally:
        for name, old in previous.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_face_analyser():
    sys.modules.pop("modules.face_analyser", None)
    return importlib.import_module("modules.face_analyser")


class FaceAnalyserProviderConfigTests(unittest.TestCase):
    def setUp(self):
        scope = ExitStack()
        scope.enter_context(_scoped_import_stubs())
        self.addCleanup(scope.close)

    def test_uses_cpu_provider_for_directml_face_analysis(self):
        face_analyser = _load_face_analyser()
        face_analyser.modules.globals.execution_providers = ["DmlExecutionProvider"]

        self.assertEqual(
            face_analyser.build_face_analyser_provider_config(),
            ["CPUExecutionProvider"],
        )

    def test_preserves_existing_provider_config_for_non_directml(self):
        face_analyser = _load_face_analyser()
        face_analyser.modules.globals.execution_providers = ["CUDAExecutionProvider"]

        self.assertEqual(
            face_analyser.build_face_analyser_provider_config(),
            ["CUDAExecutionProvider"],
        )


class ImportStubIsolationTests(unittest.TestCase):
    def test_scoped_import_stubs_restore_sys_modules(self):
        sentinel = object()
        previous = sys.modules.get(
            "modules.processors.frame._onnx_enhancer", sentinel
        )

        with _scoped_import_stubs():
            self.assertIn("modules.processors.frame._onnx_enhancer", sys.modules)

        restored = sys.modules.get("modules.processors.frame._onnx_enhancer", sentinel)
        self.assertIs(restored, previous)


if __name__ == "__main__":
    unittest.main()
