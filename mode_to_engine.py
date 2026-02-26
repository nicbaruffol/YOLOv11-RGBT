import tensorrt as trt
import sys

# 1. File paths
onnx_file_path = "runs/Anti-UAV/yolo11n-RGBRGB/weights/best.onnx"
engine_file_path = "runs/Anti-UAV/yolo11n-RGBRGB/weights/best.engine"

# 2. Set up TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path):
    # Initialize Builder, Network (with explicit batch), and Parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Set workspace size (e.g., 4GB limits the memory TensorRT can use during building)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    
    # Optional but highly recommended: Enable FP16 for much faster inference
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 optimization enabled.")

    # 3. Parse ONNX file
    print(f"Parsing ONNX file from {onnx_path}...")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file. Errors:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 4. Handle Dynamic Shapes (Batch Size)
    profile = builder.create_optimization_profile()
    
    # Format: (batch, channels, height, width)
    # Adjust the batch sizes (1, 1, 8) here depending on your actual deployment needs
    min_shape = (1, 3, 640, 640) # Minimum batch size
    opt_shape = (1, 3, 640, 640) # Typical/most common batch size
    max_shape = (8, 3, 640, 640) # Maximum batch size
    
    # Apply the profile to BOTH inputs
    profile.set_shape("input_rgb", min_shape, opt_shape, max_shape)
    profile.set_shape("input_ir", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 5. Build and save the engine
    print("Building TensorRT engine. This may take several minutes depending on your GPU...")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("ERROR: Failed to build engine.")
        return

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
        
    print(f"Successfully exported TensorRT engine to {engine_path}")

if __name__ == "__main__":
    build_engine(onnx_file_path, engine_file_path)