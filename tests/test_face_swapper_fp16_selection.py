"""Regression tests for the face swapper's FP16 model selection.

The FP16 inswapper model may only be selected on CUDA devices that support
FP16 inference natively (compute capability 7.0+).  Older cards emulate half
precision and can return NaNs, so they have to fall back to the FP32 model
instead of loading the FP16 one.
"""
import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

_FACE_SWAPPER = "modules.processors.frame.face_swapper"
_PLATFORM_INFO = "modules.platform_info"

# Modules that must be rebuilt against the stubs of the current scenario.
_REIMPORTED_MODULES = (
    "modules",
    "modules.globals",
    "modules.typing",
    "modules.gpu_processing",
    "modules.cluster_analysis",
    "modules.platform_info",
    "modules.processors",
    "modules.processors.frame",
    "modules.processors.frame.face_swapper",
)


class _FakeNDArray:
    """Stand-in for ``numpy.ndarray`` that supports ``ndarray[Any, Any]``."""

    def __class_getitem__(cls, item):
        return cls


class _FakeCV2(types.SimpleNamespace):
    """cv2 stub that resolves any uppercase constant (e.g. ``INTER_LINEAR``)."""

    def __getattr__(self, name):
        if name.isupper():
            return _constant(name)
        raise AttributeError(name)


_CONSTANTS = {}


def _constant(name: str) -> int:
    """Return a stable unique integer for an OpenCV-style constant."""
    if name not in _CONSTANTS:
        _CONSTANTS[name] = len(_CONSTANTS) + 1
    return _CONSTANTS[name]


def _numpy_stub() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        ndarray=_FakeNDArray,
        uint8=object,
        float32=object,
        float64=object,
        int32=object,
        int64=object,
        bool_=bool,
        fromfile=lambda *_args, **_kwargs: b"",
    )


def _torch_stub(cuda_available: bool, capability=None) -> types.SimpleNamespace:
    """Torch stub reporting a CUDA device with the given compute capability."""

    def get_device_capability(_index=0):
        if capability is None:
            raise RuntimeError("no CUDA device available")
        return capability

    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_capability=get_device_capability,
            empty_cache=lambda: None,
        )
    )


def _base_stubs(torch_stub) -> dict:
    return {
        "cv2": _FakeCV2(
            IMREAD_COLOR=1,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                types.SimpleNamespace(tofile=lambda *_a, **_k: None),
            ),
            cuda=types.SimpleNamespace(),
        ),
        "numpy": _numpy_stub(),
        "insightface": types.SimpleNamespace(
            model_zoo=types.SimpleNamespace(get_model=lambda *_args, **_kwargs: None)
        ),
        "insightface.app": types.SimpleNamespace(),
        "insightface.app.common": types.SimpleNamespace(Face=object),
        "onnxruntime": types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        ),
        "torch": torch_stub,
        "tqdm": types.SimpleNamespace(tqdm=lambda iterable=None, *_a, **_k: iterable),
        "sklearn": types.SimpleNamespace(),
        "sklearn.cluster": types.SimpleNamespace(KMeans=object),
        "modules.core": types.SimpleNamespace(update_status=lambda *_args, **_kwargs: None),
        "modules.face_analyser": types.SimpleNamespace(
            get_one_face=lambda *_args, **_kwargs: None,
            get_many_faces=lambda *_args, **_kwargs: None,
            default_source_face=lambda *_args, **_kwargs: None,
        ),
        "modules.utilities": types.SimpleNamespace(
            is_image=lambda *_args, **_kwargs: False,
            is_video=lambda *_args, **_kwargs: False,
        ),
        "modules.processors.frame.core": types.SimpleNamespace(),
    }


@contextmanager
def _patched_imports(torch_stub, extra_stubs=None):
    """Import the modules under test with the heavy dependencies stubbed out."""
    stubs = _base_stubs(torch_stub)
    stubs.update(extra_stubs or {})
    with patch.dict(sys.modules, stubs, clear=False):
        # Drop anything a sibling test may have imported with different
        # stubs, so the modules under test are rebuilt against these stubs.
        for name in _REIMPORTED_MODULES:
            sys.modules.pop(name, None)
        try:
            yield
        finally:
            for name in _REIMPORTED_MODULES:
                sys.modules.pop(name, None)


class Fp16CapabilityTests(unittest.TestCase):
    """``supports_fp16_inference`` must reflect the real GPU capability."""

    def test_older_cuda_gpu_is_not_fp16_capable(self):
        with _patched_imports(_torch_stub(True, (6, 1))):
            platform_info = importlib.import_module(_PLATFORM_INFO)
            self.assertFalse(platform_info.supports_fp16_inference())

    def test_modern_cuda_gpus_are_fp16_capable(self):
        for capability in ((7, 0), (7, 5), (8, 6), (9, 0), (12, 0)):
            with self.subTest(capability=capability):
                with _patched_imports(_torch_stub(True, capability)):
                    platform_info = importlib.import_module(_PLATFORM_INFO)
                    self.assertTrue(platform_info.supports_fp16_inference())

    def test_without_cuda_device_fp16_is_not_supported(self):
        with _patched_imports(_torch_stub(False)):
            platform_info = importlib.import_module(_PLATFORM_INFO)
            self.assertFalse(platform_info.supports_fp16_inference())


class SwapperModelSelectionTests(unittest.TestCase):
    """The FP16 model must only be loaded when the GPU supports it."""

    def _select(self, torch_stub, present=()):
        """Return (selected_path, fp32_path, fp16_path) for the given files."""
        with _patched_imports(torch_stub):
            face_swapper = importlib.import_module(_FACE_SWAPPER)
            fp32 = os.path.join(face_swapper.models_dir, "inswapper_128.onnx")
            fp16 = os.path.join(face_swapper.models_dir, "inswapper_128_fp16.onnx")
            available = {
                path for key, path in (("fp32", fp32), ("fp16", fp16)) if key in present
            }
            with patch.object(
                face_swapper.os.path, "exists", side_effect=lambda path: path in available
            ):
                selected = face_swapper._select_swapper_model()
            return selected, fp32, fp16

    def test_gpu_without_fp16_support_uses_the_fp32_model(self):
        selected, fp32, _fp16 = self._select(
            _torch_stub(True, (6, 1)), present=("fp32", "fp16")
        )
        self.assertEqual(selected, fp32)

    def test_gpu_without_fp16_support_never_selects_the_fp16_model(self):
        selected, _fp32, fp16 = self._select(
            _torch_stub(True, (6, 1)), present=("fp16",)
        )
        self.assertNotEqual(selected, fp16)

    def test_gpu_with_fp16_support_prefers_the_fp16_model(self):
        selected, _fp32, fp16 = self._select(
            _torch_stub(True, (7, 5)), present=("fp32", "fp16")
        )
        self.assertEqual(selected, fp16)

    def test_gpu_with_fp16_support_uses_fp32_when_fp16_is_absent(self):
        selected, fp32, _fp16 = self._select(
            _torch_stub(True, (7, 5)), present=("fp32",)
        )
        self.assertEqual(selected, fp32)

    def test_without_cuda_the_fp32_model_stays_the_default(self):
        selected, fp32, _fp16 = self._select(
            _torch_stub(False), present=("fp32", "fp16")
        )
        self.assertEqual(selected, fp32)

    def test_without_cuda_fp16_is_used_when_it_is_the_only_model(self):
        selected, _fp32, fp16 = self._select(_torch_stub(False), present=("fp16",))
        self.assertEqual(selected, fp16)


class SwapperDownloadPreferenceTests(unittest.TestCase):
    """``pre_check`` must not fetch the FP16 model for an FP16-incapable GPU."""

    def _requested_variants(self, torch_stub):
        requested = []

        def fake_ensure_any(names):
            requested.append(list(names))
            return "/models/inswapper_128.onnx"

        downloader = types.SimpleNamespace(ensure_any=fake_ensure_any)
        with _patched_imports(
            torch_stub, extra_stubs={"modules.model_downloader": downloader}
        ):
            face_swapper = importlib.import_module(_FACE_SWAPPER)
            with patch.object(face_swapper.os, "makedirs", lambda *_a, **_k: None):
                ok = face_swapper.pre_check()
        return ok, requested

    def test_gpu_without_fp16_support_requests_only_the_fp32_model(self):
        ok, requested = self._requested_variants(_torch_stub(True, (6, 1)))
        self.assertTrue(ok)
        self.assertEqual(requested, [["inswapper_128.onnx"]])

    def test_gpu_with_fp16_support_prefers_the_fp16_model(self):
        ok, requested = self._requested_variants(_torch_stub(True, (7, 5)))
        self.assertTrue(ok)
        self.assertEqual(
            requested, [["inswapper_128_fp16.onnx", "inswapper_128.onnx"]]
        )

    def test_without_cuda_the_fp32_model_is_preferred(self):
        ok, requested = self._requested_variants(_torch_stub(False))
        self.assertTrue(ok)
        self.assertEqual(
            requested, [["inswapper_128.onnx", "inswapper_128_fp16.onnx"]]
        )


if __name__ == "__main__":
    unittest.main()
