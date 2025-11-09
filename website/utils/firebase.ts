import { initializeApp } from "firebase/app";
import { getStorage, ref, getDownloadURL, listAll } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyDPnnj34vq108muNzcsTnvWDAb4Dgia3oU",
  authDomain: "low-margins.firebaseapp.com",
  projectId: "low-margins",
  storageBucket: "low-margins.firebasestorage.app",
  messagingSenderId: "579038313925",
  appId: "1:579038313925:web:f43219995d66656e8d1fac",
};

export const app = initializeApp(firebaseConfig);
export const storage = getStorage(app);

/**
 * Get download URL for a model file from Firebase Storage
 * @param path Path to the model file in Firebase Storage (e.g., "models/mlp_50.onnx")
 * @returns Promise resolving to the download URL
 */
export async function getModelUrl(path: string): Promise<string> {
  const storageRef = ref(storage, path);
  return await getDownloadURL(storageRef);
}

/**
 * List all model files from Firebase Storage in the models/ directory
 * Filters for files matching mlp_*.onnx pattern and sorts by step number
 * @returns Promise resolving to sorted array of model paths (e.g., ["models/mlp_50.onnx", "models/mlp_100.onnx"])
 */
export async function listModelPaths(): Promise<string[]> {
  const modelsRef = ref(storage, "models");
  const result = await listAll(modelsRef);

  // Filter for mlp_*.onnx files and extract step numbers
  const modelFiles = result.items
    .filter((item) => {
      const name = item.name;
      return name.startsWith("mlp_") && name.endsWith(".onnx");
    })
    .map((item) => {
      const name = item.name;
      // Extract step number from mlp_{step}.onnx
      const match = name.match(/^mlp_(\d+)\.onnx$/);
      if (!match) return null;
      const step = parseInt(match[1], 10);
      return { step, path: `models/${name}` };
    })
    .filter((item): item is { step: number; path: string } => item !== null)
    .sort((a, b) => a.step - b.step)
    .map((item) => item.path);

  return modelFiles;
}
