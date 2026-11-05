import importlib
import sys
import types
import unittest

# Third-party stubs are only installed when the real package is unavailable,
# so a machine with the full requirements installed keeps using the real one.
_THIRD_PARTY_STUBS = {
    "insightface": types.SimpleNamespace(
        app=types.SimpleNamespace(FaceAnalysis=object),
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
        uint8=object,
        fromfile=lambda *_args, **_kwargs: b"",
    ),
    "tqdm": types.SimpleNamespace(tqdm=lambda iterable, **_kwargs: iterable),
}

# Project modules are always stubbed so ``modules.face_analyser`` can be
# imported without the heavy runtime dependencies.
_PROJECT_STUBS = {
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


def _install_import_stubs():
    """Install the import stubs and return the replaced ``sys.modules`` entries.

    The mapping is handed to :func:`_restore_import_stubs` so the stubs never
    leak into other tests running later in the same process.
    """
    replaced = {}
    for name, stub in _THIRD_PARTY_STUBS.items():
        if name in sys.modules:
            continue
        replaced[name] = None
        sys.modules[name] = stub
    for name, stub in _PROJECT_STUBS.items():
        replaced[name] = sys.modules.get(name)
        sys.modules[name] = stub
    return replaced


def _restore_import_stubs(replaced):
    """Undo :func:`_install_import_stubs`, restoring the previous modules."""
    for name, previous in replaced.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _load_face_analyser():
    sys.modules.pop("modules.face_analyser", None)
    return importlib.import_module("modules.face_analyser")


class FaceAnalyserProviderConfigTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_restore_import_stubs, _install_import_stubs())
        self.face_analyser = _load_face_analyser()

    def test_uses_cpu_provider_for_directml_face_analysis(self):
        self.face_analyser.modules.globals.execution_providers = [
            "DmlExecutionProvider"
        ]

        self.assertEqual(
            self.face_analyser.build_face_analyser_provider_config(),
            ["CPUExecutionProvider"],
        )

    def test_preserves_existing_provider_config_for_non_directml(self):
        self.face_analyser.modules.globals.execution_providers = [
            "CUDAExecutionProvider"
        ]

        self.assertEqual(
            self.face_analyser.build_face_analyser_provider_config(),
            ["CUDAExecutionProvider"],
        )


if __name__ == "__main__":
    unittest.main()
