import importlib
import sys
import types
import unittest


# Modules that are replaced by stubs while ``modules.face_analyser`` is
# imported.  Keep this list in sync with the modules touched by
# ``_install_import_stubs``.  ``modules.processors.frame._onnx_enhancer`` is
# stubbed as well because ``build_face_analyser_provider_config`` imports it
# lazily, at call time.
_STUBBED_MODULES = (
    "insightface",
    "cv2",
    "numpy",
    "tqdm",
    "modules.typing",
    "modules.cluster_analysis",
    "modules.utilities",
    "modules.processors.frame._onnx_enhancer",
)


def _install_import_stubs():
    """Install lightweight stubs for the modules ``modules.face_analyser`` needs.

    Returns the replaced ``sys.modules`` entries so the caller can restore
    them with :func:`_restore_import_stubs`.
    """
    previous = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
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
    return previous


def _restore_import_stubs(previous):
    """Put the original ``sys.modules`` entries back.

    Without this the stubs stay in ``sys.modules`` for the rest of the
    process, so a test running afterwards imports a ``SimpleNamespace`` that
    is missing the real module's functions.
    """
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_face_analyser():
    sys.modules.pop("modules.face_analyser", None)
    return importlib.import_module("modules.face_analyser")


class FaceAnalyserProviderConfigTests(unittest.TestCase):
    def setUp(self):
        # The stubs replace real modules in ``sys.modules``.  Restore the
        # original entries when the test finishes so a test running later
        # imports the real module instead of a stub.
        self._previous_modules = _install_import_stubs()
        self.addCleanup(_restore_import_stubs, self._previous_modules)

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

    def test_restores_stubbed_modules_after_the_test(self):
        enhancer = "modules.processors.frame._onnx_enhancer"
        self.assertIn(enhancer, sys.modules)

        self.doCleanups()

        self.assertIs(sys.modules.get(enhancer), self._previous_modules[enhancer])


if __name__ == "__main__":
    unittest.main()
