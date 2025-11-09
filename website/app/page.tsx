"use client";

import { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";
import { fetchMNISTImages, fetchMNISTLabels } from "@/utils/mnist";

const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

type MnistResult = {
  id: number;
  label: number;
  pred: number;
  pixels: Uint8Array;
};

// Extract step numbers from model filenames and sort chronologically
function getStepModelPaths(): string[] {
  const steps = [
    { step: 50, path: "/models/steps/model_weights_step_50.onnx" },
    { step: 100, path: "/models/steps/model_weights_step_100.onnx" },
    { step: 150, path: "/models/steps/model_weights_step_150.onnx" },
    { step: 200, path: "/models/steps/model_weights_step_200.onnx" },
    { step: 250, path: "/models/steps/model_weights_step_250.onnx" },
    { step: 300, path: "/models/steps/model_weights_step_300.onnx" },
    { step: 350, path: "/models/steps/model_weights_step_350.onnx" },
    { step: 400, path: "/models/steps/model_weights_step_400.onnx" },
    { step: 450, path: "/models/steps/model_weights_step_450.onnx" },
    { step: 500, path: "/models/steps/model_weights.onnx" },
  ];

  return steps.sort((a, b) => a.step - b.step).map(s => s.path);
}


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
  const [currentStep, setCurrentStep] = useState<string>("");
  const [isSimulating, setIsSimulating] = useState(false);
  const imagesRef = useRef<Uint8Array[]>([]);
  const labelsRef = useRef<Uint8Array>(new Uint8Array());
  const indicesRef = useRef<number[]>([]);

  // Function to run inference with a specific model
  const runInferenceWithModel = async (
    modelPath: string,
    images: Uint8Array[],
    labels: Uint8Array,
    indices: number[]
  ) => {
    const session = await ort.InferenceSession.create(modelPath, {
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
  };

  // Function to simulate all steps sequentially
  const simulateSteps = async () => {
    if (isSimulating) return;

    setIsSimulating(true);
    const stepPaths = getStepModelPaths();

    try {
      // Reset predictions to -1 before starting
      setItems((prev) =>
        prev.map((item) => ({ ...item, pred: -1 }))
      );

      setStatus("running inference…");

      for (let stepIdx = 0; stepIdx < stepPaths.length; stepIdx++) {
        const modelPath = stepPaths[stepIdx];
        const stepName = modelPath.split("/").pop() || `Step ${stepIdx}`;
        setCurrentStep(stepName);

        // Run inference with current step model
        await runInferenceWithModel(
          modelPath,
          imagesRef.current,
          labelsRef.current,
          indicesRef.current
        );

        // Wait 5 seconds before next step (except for the last step)
        if (stepIdx < stepPaths.length - 1) {
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      }

      setCurrentStep("");
      setStatus("done");
    } catch (err) {
      console.error(err);
      setStatus("error");
      setCurrentStep("");
    } finally {
      setIsSimulating(false);
    }
  };

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
        <button
          onClick={simulateSteps}
          disabled={isSimulating || status === "loading images…"}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${isSimulating || status === "loading images…"
            ? "bg-slate-200 text-slate-500 cursor-not-allowed"
            : "bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800"
            }`}
        >
          {isSimulating ? "Simulating..." : "Simulate Training Steps"}
        </button>
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
