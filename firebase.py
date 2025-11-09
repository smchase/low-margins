import os
import uuid
import torch
import onnx
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
try:
    import firebase_admin
    from firebase_admin import credentials, storage as firebase_storage
except ImportError:
    firebase_admin = None


def push_to_firebase(model, step_count=None, timeout=10):
    """
    Convert PyTorch model to ONNX and upload to Firebase Storage.
    Non-blocking - runs in background thread.
    
    Args:
        model: PyTorch model (should be in eval mode)
        step_count: Optional step count for filename (e.g., mlp_50.onnx)
        timeout: Timeout in seconds for the upload operation (default: 10 seconds)
    """
    if firebase_admin is None:
        print("⚠️  firebase-admin not installed. Install it with: pip install firebase-admin")
        return
    
    try:
        # Initialize Firebase Admin SDK if not already initialized
        if not firebase_admin._apps:
            service_account_path = os.getenv(
                "FIREBASE_SERVICE_ACCOUNT_KEY", "firebase-service-account.json"
            )
            if os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(
                    cred, {"storageBucket": "low-margins.firebasestorage.app"}
                )
            else:
                firebase_admin.initialize_app(
                    options={"storageBucket": "low-margins.firebasestorage.app"}
                )
        
        # Clone model and convert to float32 for ONNX export (ONNX doesn't support bfloat16)
        # We need to properly clone to avoid modifying the original model
        from train.model import MLP
        model_copy = MLP(num_classes=10).float()
        # Load state dict and convert all parameters to float32
        state_dict = model.state_dict()
        float_state_dict = {}
        for k, v in state_dict.items():
            param = v.cpu().float()
            # Check for NaN/Inf values that might break ONNX export
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise Exception(f"Model has NaN/Inf values in {k} - skipping export for step {step_count}")
            float_state_dict[k] = param
        
        model_copy.load_state_dict(float_state_dict)
        model_copy.eval()
        
        # Create dummy input (MNIST: 1 channel, 28x28) in float32 - use same as working export
        dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
        
        # Use unique temp filename to avoid conflicts with concurrent exports
        temp_filename = f"temp_model_{uuid.uuid4().hex[:8]}_part.onnx"
        
        # Export to ONNX - use exact same method that worked before
        # Try the simple export first (same as model_exporter.py)
        with torch.no_grad():
            torch.onnx.export(
                model_copy,
                dummy_input,
                temp_filename,
                input_names=["input"],
                output_names=["logits"],
                opset_version=18,
                verbose=False,
            )
        
        # Load and save final ONNX model
        imported_model = onnx.load(temp_filename, load_external_data=True)
        onnx_filename = f"mlp_{step_count}.onnx" if step_count else "mlp.onnx"
        onnx.save(imported_model, onnx_filename)
        
        # Upload to Firebase with timeout
        bucket = firebase_storage.bucket()
        blob = bucket.blob(f"models/{onnx_filename}")
        
        print(f"Uploading {onnx_filename} to Firebase Storage...")
        
        def upload_with_timeout():
            blob.upload_from_filename(onnx_filename)
            blob.make_public()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(upload_with_timeout)
            future.result(timeout=timeout)
        
        print(f"✅ Successfully uploaded {onnx_filename} to Firebase Storage")
        print(f"   Public URL: {blob.public_url}")
        
        # Clean up local files
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            if os.path.exists(f"{temp_filename}.data"):
                os.remove(f"{temp_filename}.data")
            if os.path.exists(onnx_filename):
                os.remove(onnx_filename)
        except Exception as e:
            print(f"⚠️  Warning: Failed to clean up local files: {e}")
    except FuturesTimeoutError:
        print(f"⏱️  Upload timeout ({timeout}s) exceeded for step {step_count}")
        # Clean up temp file even on timeout
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            if os.path.exists(f"{temp_filename}.data"):
                os.remove(f"{temp_filename}.data")
            if 'onnx_filename' in locals() and os.path.exists(onnx_filename):
                os.remove(onnx_filename)
        except:
            pass
    except Exception as e:
        print(f"❌ Failed to export/upload model (step {step_count}): {e}")
        # Clean up temp file even on error
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            if os.path.exists(f"{temp_filename}.data"):
                os.remove(f"{temp_filename}.data")
            if 'onnx_filename' in locals() and os.path.exists(onnx_filename):
                os.remove(onnx_filename)
        except:
            pass


def clear_models_folder():
    """Delete all files in the models/ folder in Firebase Storage."""
    if firebase_admin is None:
        print("⚠️  firebase-admin not installed. Skipping Firebase cleanup.")
        return
    
    # Initialize Firebase Admin SDK if not already initialized
    if not firebase_admin._apps:
        service_account_path = os.getenv(
            "FIREBASE_SERVICE_ACCOUNT_KEY", "firebase-service-account.json"
        )
        if os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(
                cred, {"storageBucket": "low-margins.firebasestorage.app"}
            )
        else:
            firebase_admin.initialize_app(
                options={"storageBucket": "low-margins.firebasestorage.app"}
            )
    
    bucket = firebase_storage.bucket()
    models_folder = bucket.list_blobs(prefix="models/")
    
    deleted_count = 0
    for blob in models_folder:
        blob.delete()
        deleted_count += 1
    
    if deleted_count > 0:
        print(f"🗑️  Cleared {deleted_count} file(s) from Firebase models/ folder")
    else:
        print("📁 Firebase models/ folder is already empty")


def cleanup_temp_onnx_files():
    """Clean up any leftover temp ONNX files from failed exports."""
    import glob
    try:
        # Clean up temp files
        temp_files = glob.glob("temp_model_*_part.onnx") + glob.glob("temp_model_*_part.onnx.data")
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        
        # Also clean up any mlp_*.onnx files that might be left behind (they're already in Firebase)
        mlp_files = glob.glob("mlp_*.onnx")
        for mlp_file in mlp_files:
            try:
                if os.path.exists(mlp_file):
                    os.remove(mlp_file)
            except Exception:
                pass
    except Exception:
        pass


def push_to_firebase_async(model, step_count=None, timeout=10):
    """
    Push model to Firebase in a background thread (non-blocking).
    Returns immediately without waiting for upload to complete.
    """
    thread = threading.Thread(
        target=push_to_firebase,
        args=(model, step_count, timeout),
        daemon=True  # Daemon thread so it doesn't prevent process exit
    )
    thread.start()
    return thread  # Return thread in case caller wants to track it, but don't block

