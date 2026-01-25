from vllm_metal.model_runner import MetalModelRunner

class MinimalModelConfig:
    def __init__(self, model: str) -> None:
        self.model = model


class MinimalVllmConfig:
    def __init__(self, model: str) -> None:
        self.model_config = MinimalModelConfig(model)


class SeqData:
    def __init__(self, token_ids: list[int]) -> None:
        self._token_ids = token_ids

    def get_token_ids(self) -> list[int]:
        return self._token_ids


class SeqGroup:
    def __init__(self, token_ids: list[int], is_prompt: bool) -> None:
        self.seq_data = {0: SeqData(token_ids)}
        self.is_prompt = is_prompt


runner = MetalModelRunner(MinimalVllmConfig("HuggingFaceTB/SmolLM2-135M"))  # type: ignore[arg-type]
runner.load_model()

prompts = [
    "Hello",
    "The capital of France is",
    "What is your name?",
    "Hi, what are you",
    "Tell me a joke about",
]
seq_groups = [SeqGroup(runner.tokenizer.encode(p), True) for p in prompts]

outputs = runner.execute_model(seq_groups)

print(f"{'idx':<4}{'prompt':<45}{'token':<45}token_id")
for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
    token_id = int(output["token_id"])
    token = runner.tokenizer.decode([token_id])
    print(f"{idx:<4}{repr(prompt):<45}{repr(token):<45}{token_id}")
