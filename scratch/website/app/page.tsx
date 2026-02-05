"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import * as ort from "onnxruntime-web";
import { fetchMNISTImages, fetchMNISTLabels } from "@/utils/mnist";
import { getModelUrl, listModelPaths } from "@/utils/firebase";

const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

type MnistResult = {
  id: number;
  label: number;
  pred: number;
  pixels: Uint8Array;
};

function MnistCanvas({ pixels }: { pixels: Uint8Array }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = 28;
    const imageData = ctx.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const v = pixels[i];
      imageData.data[i * 4 + 0] = v;
      imageData.data[i * 4 + 1] = v;
      imageData.data[i * 4 + 2] = v;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    ctx.imageSmoothingEnabled = false;
  }, [pixels]);

  return (
    <canvas
      ref={canvasRef}
      width={28}
      height={28}
      className="h-full w-full bg-white"
    />
  );
}

export default function MnistPage() {
  const [status, setStatus] = useState("loading images…");
  const [items, setItems] = useState<MnistResult[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const imagesRef = useRef<Uint8Array[]>([]);
  const labelsRef = useRef<Uint8Array>(new Uint8Array());
  const indicesRef = useRef<number[]>([]);
  const processedPathsRef = useRef<Set<string>>(new Set());
  const shouldContinueSimulatingRef = useRef<boolean>(false);

  // Function to run inference with a specific model
  const runInferenceWithModel = useCallback(async (
    modelStoragePath: string,
    images: Uint8Array[],
    labels: Uint8Array,
    indices: number[]
  ) => {
    // Fetch the download URL from Firebase Storage
    const modelUrl = await getModelUrl(modelStoragePath);

    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: hasWebGPU ? ["webgpu", "wasm"] : ["wasm"],
    });

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    // Run inference on all items
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const pixels = images[idx];
      const label = labels[idx];

      const data = new Float32Array(1 * 1 * 28 * 28);
      for (let j = 0; j < pixels.length; j++) {
        data[j] = pixels[j] / 255.0;
      }

      const tensor = new ort.Tensor("float32", data, [1, 1, 28, 28]);
      const out = await session.run({ [inputName]: tensor });
      const logits = out[outputName].data as Float32Array;

      // argmax
      let bestIdx = 0;
      let bestVal = logits[0];
      for (let j = 1; j < logits.length; j++) {
        if (logits[j] > bestVal) {
          bestVal = logits[j];
          bestIdx = j;
        }
      }

      // Update this specific item with its prediction
      setItems((prev) => {
        const updated = [...prev];
        updated[i] = {
          id: idx,
          label,
          pred: bestIdx,
          pixels,
        };
        return updated;
      });
    }
  }, []);

  // Function to simulate all steps sequentially
  const simulateSteps = useCallback(async () => {
    if (isSimulating) return;

    setIsSimulating(true);
    shouldContinueSimulatingRef.current = true;

    try {
      // Reset predictions to -1 before starting
      setItems((prev) =>
        prev.map((item) => ({ ...item, pred: -1 }))
      );

      setStatus("running inference…");

      // Continuous loop: process files chronologically and check for new ones
      while (shouldContinueSimulatingRef.current) {
        // Fetch fresh list of model files from Firebase Storage (sorted by createdAt)
        const allModelFiles = await listModelPaths();

        // Filter out already processed files
        const unprocessedFiles = allModelFiles.filter(
          (file) => !processedPathsRef.current.has(file.path)
        );

        if (unprocessedFiles.length === 0) {
          // No new files available, wait before checking again
          setStatus("waiting for new steps…");
          await new Promise((resolve) => setTimeout(resolve, 5000));
          continue;
        }

        // Process files one at a time in chronological order
        for (const file of unprocessedFiles) {
          // Check if we should continue before each iteration
          if (!shouldContinueSimulatingRef.current) break;

          const fileName = file.path.split("/").pop() || file.path;
          setStatus(`processing ${fileName}…`);
          console.log(`Running inference with model: ${fileName}`);

          // Run inference with current model file
          await runInferenceWithModel(
            file.path,
            imagesRef.current,
            labelsRef.current,
            indicesRef.current
          );

          // Mark this file as processed
          processedPathsRef.current.add(file.path);

          // Small delay before checking for next file
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
      }

      setStatus("done");
    } catch (err) {
      console.error(err);
      setStatus("error");
    } finally {
      shouldContinueSimulatingRef.current = false;
      setIsSimulating(false);
    }
  }, [isSimulating, runInferenceWithModel]);

  // Load images and labels on mount
  useEffect(() => {
    (async () => {
      try {
        setStatus("loading images…");
        const [{ images, numImages }, { labels }] = await Promise.all([
          fetchMNISTImages("/mnist/t10k-images.idx3-ubyte"),
          fetchMNISTLabels("/mnist/t10k-labels.idx1-ubyte"),
        ]);

        const count = Math.min(300, numImages);
        const indices = Array.from({ length: count }, () =>
          Math.floor(Math.random() * numImages)
        );

        // Store references for reuse
        imagesRef.current = images;
        labelsRef.current = labels;
        indicesRef.current = indices;

        // Create initial items with just images and labels (no predictions yet)
        const initialItems: MnistResult[] = indices.map((idx) => ({
          id: idx,
          label: labels[idx],
          pred: -1, // -1 indicates prediction not ready yet
          pixels: images[idx],
        }));

        setItems(initialItems);
        setStatus("ready");
      } catch (err) {
        console.error(err);
        setStatus("error");
      }
    })();
  }, []);

  // Handle spacebar keypress to start simulation
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Only trigger on spacebar and when not already simulating
      if (e.code === "Space" && !isSimulating && status !== "loading images…") {
        e.preventDefault(); // Prevent page scroll
        simulateSteps();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [isSimulating, status, simulateSteps]);

  const correctCount = items.filter((i) => i.pred !== -1 && i.label === i.pred).length;
  const itemsWithPredictions = items.filter((i) => i.pred !== -1);
  const accuracy =
    itemsWithPredictions.length > 0
      ? Math.round((correctCount / itemsWithPredictions.length) * 100)
      : 0;

  return (
    <div className="mx-auto flex flex-col gap-2 px-6 py-10">
      <header className="flex flex-wrap items-center justify-center gap-4">
        <div className="text-center">
          <h1 className="text-5xl font-semibold text-slate-900">
            wccl 🎥
          </h1>
        </div>
      </header>

      <div className="flex flex-col items-center justify-center mb-8">
        <div className="rounded-lg px-4 py-1 text-lg text-slate-900">
          Accuracy:{" "}
          <span
            className={
              "font-semibold " +
              (accuracy >= 80
                ? "text-green-600"
                : accuracy >= 50
                  ? "text-yellow-600"
                  : "text-red-600")
            }
          >
            {accuracy}%
          </span>
        </div>
        <p className="text-xs text-slate-600">
          {correctCount}/{itemsWithPredictions.length || items.length} correct
        </p>
      </div>

      {/* grid of images */}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(80px,1fr))] gap-2">
        {items.map((item, idx) => {
          const hasPrediction = item.pred !== -1;
          const correct = hasPrediction && item.label === item.pred;
          return (
            <div
              key={item.id + "-" + idx}
              className={`aspect-square w-20 h-20 border-4 ${!hasPrediction
                ? "border-slate-300"
                : correct
                  ? "border-green-500"
                  : "border-red-500"
                }`}
            >
              <MnistCanvas pixels={item.pixels} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
