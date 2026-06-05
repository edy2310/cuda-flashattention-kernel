import numpy as np
import tritonclient.http as httpclient


def main():
    # Update the URL if Triton is running on a different host or port.
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Example input shape: [B, H, N, D].
    B, H, N, D = 1, 8, 256, 64
    q = np.random.randn(B, H, N, D).astype(np.float32)
    k = np.random.randn(B, H, N, D).astype(np.float32)
    v = np.random.randn(B, H, N, D).astype(np.float32)

    inputs = [
        httpclient.InferInput("Q", q.shape, "FP32"),
        httpclient.InferInput("K", k.shape, "FP32"),
        httpclient.InferInput("V", v.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(q)
    inputs[1].set_data_from_numpy(k)
    inputs[2].set_data_from_numpy(v)

    outputs = [httpclient.InferRequestedOutput("O")]
    result = client.infer(model_name="flash_attention", inputs=inputs, outputs=outputs)
    out = result.as_numpy("O")

    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.mean():.6f}")


if __name__ == "__main__":
    main()
