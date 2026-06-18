import importlib
import sys
import types


def import_predict(monkeypatch):
    cog = types.ModuleType("cog")
    cog.BasePredictor = object
    cog.Input = lambda *args, **kwargs: kwargs.get("default")
    cog.Path = str
    monkeypatch.setitem(sys.modules, "cog", cog)

    openai = types.ModuleType("openai")
    openai.OpenAI = object
    monkeypatch.setitem(sys.modules, "openai", openai)

    typhoon_ocr = types.ModuleType("typhoon_ocr")
    typhoon_ocr.prepare_ocr_messages = object
    monkeypatch.setitem(sys.modules, "typhoon_ocr", typhoon_ocr)

    sys.modules.pop("predict", None)
    return importlib.import_module("predict")


def test_t4_default_command_uses_cpu_offload(monkeypatch):
    predict = import_predict(monkeypatch)
    predictor = predict.Predictor()
    predictor.setup()

    monkeypatch.setattr(predictor, "_gpu_memory_mib", lambda: 14_900)

    command = predictor._build_vllm_command()

    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.80"
    assert command[command.index("--cpu-offload-gb") + 1] == "4"
