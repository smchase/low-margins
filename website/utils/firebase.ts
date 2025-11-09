import { initializeApp } from "firebase/app";
import { getStorage, ref, getDownloadURL, listAll, getMetadata } from "firebase/storage";

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
 * Model file information with metadata
 */
export type ModelFileInfo = {
  path: string;
  createdAt: number; // timestamp in milliseconds
};

/**
 * List all model files from Firebase Storage in the models/ directory
 * Filters for files matching mlp_*.onnx pattern and sorts by createdAt timestamp
 * @returns Promise resolving to sorted array of model file info, chronologically ordered
 */
export async function listModelPaths(): Promise<ModelFileInfo[]> {
  const modelsRef = ref(storage, "models");
  const result = await listAll(modelsRef);

  // Filter for mlp_*.onnx files and get their metadata
  const modelFilesPromises = result.items
    .filter((item) => {
      const name = item.name;
      return name.startsWith("mlp_") && name.endsWith(".onnx");
    })
    .map(async (item) => {
      const name = item.name;
      const path = `models/${name}`;
      const metadata = await getMetadata(item);
      return {
        path,
        createdAt: metadata.timeCreated ? new Date(metadata.timeCreated).getTime() : 0,
      };
    });

  const modelFiles = await Promise.all(modelFilesPromises);

  // Sort by createdAt timestamp (chronologically)
  return modelFiles.sort((a, b) => a.createdAt - b.createdAt);
}
