"use client";

import React, { useState, useRef, useEffect } from "react";
import { Upload, Camera, Link as LinkIcon } from "lucide-react";

interface ImageUploadProps {
  onUpload: (file: File) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onUpload }) => {
  const [dragActive, setDragActive] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  // Open camera preview modal
  const openCamera = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      setStream(mediaStream);
      setShowCamera(true);
    } catch (error) {
      console.error("Camera access error:", error);
      alert(
        "Unable to access camera. Please check permissions or try a different browser."
      );
    }
  };

  // Capture the photo when user clicks "Capture"
  const capturePhoto = () => {
    if (videoRef.current) {
      const video = videoRef.current;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "captured-image.jpg", {
              type: blob.type,
            });
            onUpload(file);
          }
        }, "image/jpeg");
      }
    }
    closeCamera();
  };

  const closeCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setShowCamera(false);
  };

  // When the camera modal opens, assign the video stream to the video element
  useEffect(() => {
    if (showCamera && videoRef.current && stream) {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    }
  }, [showCamera, stream]);

  const handleUrlUpload = () => {
    const url = prompt("Enter the URL of the image:");
    if (url) {
      fetch(url)
        .then((response) => response.blob())
        .then((blob) => {
          const file = new File([blob], "image_from_url", { type: blob.type });
          onUpload(file);
        })
        .catch((error) => {
          console.error("Error fetching image from URL:", error);
          alert("Failed to fetch image from URL. Please try again.");
        });
    }
  };

  return (
    <>
      <div
        className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer ${
          dragActive ? "border-green-500 bg-green-50" : "border-gray-300"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload className="w-12 h-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-600">
          Drag and drop or click to upload an image
        </p>
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={handleFileChange}
          accept="image/*"
        />
        <div className="flex mt-4 space-x-4">
          <button
            onClick={openCamera}
            className="z-20 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <Camera className="w-4 h-4 mr-2 inline-block" />
            Use Camera
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleUrlUpload();
            }}
            className="px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
          >
            <LinkIcon className="w-4 h-4 mr-2 inline-block" />
            Upload from URL
          </button>
        </div>
      </div>

      {/* Camera Preview Modal */}
      {showCamera && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-4 rounded-lg flex flex-col items-center">
            <video
              ref={videoRef}
              className="w-full max-w-md rounded-md"
              autoPlay
            />
            <div className="mt-4 flex space-x-4">
              <button
                onClick={capturePhoto}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Capture
              </button>
              <button
                onClick={closeCamera}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ImageUpload;
