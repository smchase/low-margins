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
  const [status, setStatus] = useState("loading…");
  const [items, setItems] = useState<MnistResult[]>([]);

  useEffect(() => {
    (async () => {
      try {
        const session = await ort.InferenceSession.create("/models/mlp.onnx", {
          executionProviders: hasWebGPU ? ["webgpu", "wasm"] : ["wasm"],
        });

        const { images, numImages } = await fetchMNISTImages(
          "/mnist/t10k-images.idx3-ubyte"
        );
        const { labels } = await fetchMNISTLabels(
          "/mnist/t10k-labels.idx1-ubyte"
        );

        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        const count = Math.min(300, numImages);
        const indices = Array.from({ length: count }, () =>
          Math.floor(Math.random() * numImages)
        );

        const results: MnistResult[] = [];

        for (const idx of indices) {
          const pixels = images[idx];
          const label = labels[idx];

          const data = new Float32Array(1 * 1 * 28 * 28);
          for (let i = 0; i < pixels.length; i++) {
            data[i] = pixels[i] / 255.0;
          }

          const tensor = new ort.Tensor("float32", data, [1, 1, 28, 28]);
          const out = await session.run({ [inputName]: tensor });
          const logits = out[outputName].data as Float32Array;

          // argmax
          let bestIdx = 0;
          let bestVal = logits[0];
          for (let i = 1; i < logits.length; i++) {
            if (logits[i] > bestVal) {
              bestVal = logits[i];
              bestIdx = i;
            }
          }

          results.push({
            id: idx,
            label,
            pred: bestIdx,
            pixels,
          });
        }

        setItems(results);
        setStatus("done");
      } catch (err) {
        console.error(err);
        setStatus("error");
      }
    })();
  }, []);

  const correctCount = items.filter((i) => i.label === i.pred).length;
  const accuracy =
    items.length > 0 ? Math.round((correctCount / items.length) * 100) : 0;

  return (
    <div className="mx-auto flex flex-col gap-6 px-6 py-10">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">
            MNIST Browser Inference
          </h1>
          <p className="text-sm text-slate-600">
            {status === "done"
              ? `Ran ${items.length} samples from t10k.`
              : "Loading model & dataset…"}
          </p>
        </div>
        {/* <div
          className={`rounded-lg px-4 py-2 text-sm font-medium ${status === "done"
            ? "bg-slate-100 ring-1 ring-slate-300 text-slate-900"
            : "bg-yellow-100 text-yellow-900 ring-1 ring-yellow-300"
            }`}
        >
          Status: <span className="font-bold capitalize">{status}</span>
        </div> */}
      </header>

      <div className="flex items-center gap-3">
        <div className="rounded-lg bg-slate-100 px-4 py-2 text-sm text-slate-900 ring-1 ring-slate-300">
          Accuracy:{" "}
          <span
            className={
              accuracy >= 90
                ? "text-green-600"
                : accuracy >= 70
                  ? "text-yellow-600"
                  : "text-red-600"
            }
          >
            {accuracy}%
          </span>
        </div>
        <p className="text-xs text-slate-600">
          {correctCount}/{items.length} correct
        </p>
      </div>

      {/* grid of images */}
      <div className="grid grid-cols-[repeat(auto-fit,minmax(80px,1fr))]">
        {items.map((item, idx) => {
          const correct = item.label === item.pred;
          return (
            <div
              key={item.id + "-" + idx}
              className={`m-px aspect-square w-20 h-20 border-8 ${correct
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
