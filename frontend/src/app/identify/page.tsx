"use client";

import React, { useState } from "react";
import { Clover } from "lucide-react";
import ImageUpload from "@/components/ImageUpload";

const fasterRCNNClassMapping: Record<number, string> = {
  0: "Crops",
  1: "Powdery Mildew",
  2: "Aphids",
  3: "Army Worm",
  4: "Bacterial Blight",
  5: "Curl Virus",
  6: "Healthy",
  7: "Target Spot",
};

// Helper function to map the model names
const getModelName = (model: string) => {
  if (model.toLowerCase() === "fasterrcnn") return "Faster R-CNN";
  if (model.toLowerCase() === "yolov8") return "YOLO-V8";
  return model;
};

export default function Home() {
  const [selectedModel, setSelectedModel] = useState("fasterrcnn");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [results, setResults] = useState<
    { class: number; confidence: number; model?: string }[]
  >([]);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (file: File) => {
    setLoading(true);
    setImagePreview(URL.createObjectURL(file)); // Generate preview for uploaded image
    setProcessedImage(null);
    setResults([]);

    const formData = new FormData();
    formData.append("image", file);

    const apiEndpoint = `http://127.0.0.1:5000/test/${selectedModel}`;
    try {
      const response = await fetch(apiEndpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to identify plant");
      }

      const data = await response.json();
      console.log(data);

      if (data.image) {
        setProcessedImage(`data:image/jpeg;base64,${data.image}`);
      }

      if (selectedModel === "comparison" && data.best_predictions) {
        setResults(
          data.best_predictions.map((pred: any) => ({
            class: pred.class,
            confidence: pred.confidence,
            model: pred.model,
          }))
        );
      } else if (data.predictions) {
        setResults([
          {
            class: data.predictions[0].class,
            confidence: data.predictions[0].confidence,
          },
        ]);
      }
    } catch (error) {
      console.error("Error:", error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to map the class based on model
  const getClassName = (res: { class: number; model?: string }) => {
    // For non-comparison models, use the selectedModel
    if (selectedModel === "fasterrcnn") {
      return fasterRCNNClassMapping[res.class] || res.class;
    } else if (selectedModel === "yolov8") {
      // For YOLOv8, results are shifted so add one to the class index
      return fasterRCNNClassMapping[res.class + 1] || res.class + 1;
    } else if (selectedModel === "comparison") {
      // In the comparison model, check the model property from result
      if (res.model === "fasterrcnn") {
        return fasterRCNNClassMapping[res.class] || res.class;
      } else if (res.model === "yolov8") {
        return fasterRCNNClassMapping[res.class + 1] || res.class + 1;
      }
      return res.class;
    }
    return res.class;
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-center font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold mb-8 flex items-center">
          <Clover className="mr-2" /> Plant Identifier
        </h1>
      </div>

      <div className="w-full max-w-2xl">
        {/* Model Selection Dropdown */}
        <div className="mb-4">
          <label className="block text-lg font-semibold mb-2">
            Select Model:
          </label>
          <select
            className="w-full p-3 border rounded-lg shadow-sm"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            <option value="fasterrcnn">Faster R-CNN</option>
            <option value="yolov8">YOLOv8</option>
            <option value="comparison">Comparison</option>
          </select>
        </div>

        {/* Custom Image Upload Component */}
        <ImageUpload onUpload={handleUpload} />

        {loading && <p className="mt-4 text-center">Identifying plant...</p>}

        {/* Display Uploaded Image */}
        {imagePreview && (
          <div className="mt-8 p-6 bg-white rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">Uploaded Image:</h2>
            <img
              src={imagePreview}
              alt="Uploaded plant"
              className="w-full rounded-lg shadow-md"
            />
          </div>
        )}

        {/* Display Processed Image (from API response) */}
        {processedImage && (
          <div className="mt-8 p-6 bg-white rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">Processed Image:</h2>
            <img
              src={processedImage}
              alt="Processed plant"
              className="w-full rounded-lg shadow-md"
            />
          </div>
        )}

        {/* Display Prediction Results */}
        {results.length > 0 && (
          <div className="mt-8 p-6 bg-white rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">
              Identification Results:
            </h2>
            {results.map((res, index) => (
              <div key={index} className="mb-4 p-4 border rounded-lg shadow-sm">
                {res.model && (
                  <p className="text-lg font-medium">
                    Model: <strong>{getModelName(res.model)}</strong>
                  </p>
                )}
                <p className="text-lg">
                  Class: <strong>{getClassName(res)}</strong>
                </p>
                <p className="text-lg">
                  Confidence:{" "}
                  <strong>{(res.confidence * 100).toFixed(2)}%</strong>
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
