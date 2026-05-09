import React, { useEffect, useRef, useState } from 'react';
import { AlertCircle, Camera, Loader2, Volume2 } from 'lucide-react';
import '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

type SafetyLevel = 'safe' | 'caution' | 'danger';
type Position = 'left' | 'center' | 'right';

type DetectedObject = {
  name: string;
  distance: string;
  position: Position;
  safety: SafetyLevel;
};

const DANGER_OBJECTS = new Set([
  'person',
  'car',
  'truck',
  'bus',
  'bicycle',
  'motorcycle',
  'chair',
  'couch',
  'sofa',
  'bench',
  'stairs',
  'step',
  'potted plant',
  'backpack',
  'suitcase',
]);

export default function VisionAssist() {
  const [isScanning, setIsScanning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [needsCameraPermission, setNeedsCameraPermission] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Preparing detector...');
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanIntervalRef = useRef<number | null>(null);
  const lastSpeakTimeRef = useRef(0);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const modelLoadRef = useRef<Promise<cocoSsd.ObjectDetection> | null>(null);

  useEffect(() => {
    void preloadModel();

    return () => {
      stopCamera();
    };
  }, []);

  const preloadModel = async () => {
    try {
      await ensureModelLoaded();
      setStatusMessage('Model ready. Tap Enable Camera to begin scanning.');
    } catch (error) {
      console.error('Model preload error:', error);
      setStatusMessage('Model failed to load. Tap Retry Startup.');
    }
  };

  const speak = (text: string) => {
    const currentTime = Date.now();
    if (currentTime - lastSpeakTimeRef.current < 5000) {
      return;
    }

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
      lastSpeakTimeRef.current = currentTime;
    }
  };

  const ensureModelLoaded = async () => {
    if (modelRef.current) {
      return modelRef.current;
    }

    if (!modelLoadRef.current) {
      modelLoadRef.current = cocoSsd.load();
    }

    modelRef.current = await modelLoadRef.current;
    return modelRef.current;
  };

  const startCamera = async () => {
    try {
      setNeedsCameraPermission(false);
      setStatusMessage('Requesting camera access...');
      await ensureModelLoaded();

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' }, width: 1280, height: 720 },
          audio: false,
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }

      setStatusMessage('Camera ready. Starting automatic scanning...');
      window.setTimeout(() => {
        void startScanning();
      }, 1000);
    } catch (error) {
      console.error('Startup error:', error);
      const startupError = error as DOMException;
      if (startupError?.name === 'NotAllowedError') {
        setNeedsCameraPermission(true);
        setStatusMessage('Camera permission denied. Tap Enable Camera and allow access.');
        speak('Camera permission denied. Please tap enable camera and allow access.');
      } else {
        setStatusMessage('Unable to start camera or model. Tap Enable Camera to retry.');
        speak('Unable to start camera. Please tap enable camera to retry.');
      }
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
  };

  const getPosition = (bbox: [number, number, number, number], imageWidth: number): Position => {
    const centerX = bbox[0] + bbox[2] / 2;
    const third = imageWidth / 3;

    if (centerX < third) return 'left';
    if (centerX > third * 2) return 'right';
    return 'center';
  };

  const estimateDistance = (bbox: [number, number, number, number], label: string) => {
    const boxHeight = bbox[3];
    const area = bbox[2] * bbox[3];

    if (boxHeight > 280 || area > 100000) return '0.5-1.0';
    if (boxHeight > 180 || area > 50000) return '1.0-2.0';
    if (boxHeight > 100 || area > 20000) return '2.0-4.0';

    if (DANGER_OBJECTS.has(label)) {
      return '3.0-5.0';
    }

    return '4.0+';
  };

  const getSafety = (label: string, distance: string): SafetyLevel => {
    const closeRange = distance.startsWith('0.5') || distance.startsWith('1.0');
    const normalized = label.toLowerCase();

    if (DANGER_OBJECTS.has(normalized)) {
      return closeRange ? 'danger' : 'caution';
    }

    if (closeRange) {
      return 'caution';
    }

    return 'safe';
  };

  const captureAndDetect = async () => {
    if (isLoading) return;

    const video = videoRef.current;
    const model = modelRef.current;

    if (!video || !model || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }

    setIsLoading(true);

    try {
      const predictions = await model.detect(video);
      const objects: DetectedObject[] = predictions
        .filter(prediction => prediction.score >= 0.5)
        .map(prediction => {
          const bbox = prediction.bbox as [number, number, number, number];
          const normalizedLabel = prediction.class.toLowerCase();
          const distance = estimateDistance(bbox, normalizedLabel);

          return {
            name: prediction.class,
            distance,
            position: getPosition(bbox, video.videoWidth || 1280),
            safety: getSafety(normalizedLabel, distance),
          };
        })
        .sort((left, right) => {
          const leftClose = left.distance.startsWith('0.5') || left.distance.startsWith('1.0');
          const rightClose = right.distance.startsWith('0.5') || right.distance.startsWith('1.0');

          if (left.safety !== right.safety) {
            if (left.safety === 'danger') return -1;
            if (right.safety === 'danger') return 1;
            if (left.safety === 'caution') return -1;
            if (right.safety === 'caution') return 1;
          }

          if (leftClose !== rightClose) {
            return leftClose ? -1 : 1;
          }

          return 0;
        })
        .slice(0, 5);

      const primaryObstacle = objects.find(object => object.safety !== 'safe');

      if (primaryObstacle) {
        const message = `${primaryObstacle.name} ${primaryObstacle.position}, about ${primaryObstacle.distance} meters. ${primaryObstacle.safety === 'danger' ? 'Danger.' : 'Caution.'}`;
        setStatusMessage(message);
        speak(message);
      } else {
        setStatusMessage(objects.length > 0 ? 'Objects detected. Path appears clear.' : 'Path clear. Scanning...');
      }
    } catch (error) {
      console.error('TensorFlow detection error:', error);
      setStatusMessage('Unable to analyze. Continuing to scan...');
    } finally {
      setIsLoading(false);
    }
  };

  const startScanning = async () => {
    setIsScanning(true);
    setStatusMessage('Scanning environment...');
    speak('Scanning started. I will alert you of obstacles.');

    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
    }

    scanIntervalRef.current = window.setInterval(() => {
      void captureAndDetect();
    }, 3000);

    window.setTimeout(() => {
      void captureAndDetect();
    }, 500);
  };

  const stopScanning = () => {
    setIsScanning(false);
    setStatusMessage('Scanning stopped.');
    speak('Scanning stopped.');

    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      <header className="bg-gray-800 p-4 shadow-lg">
        <div className="flex items-center justify-center gap-3">
          <Camera className="w-8 h-8 text-blue-400" />
          <h1 className="text-2xl font-bold">VisionAssist</h1>
          {isScanning && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-red-400">LIVE</span>
            </div>
          )}
        </div>
      </header>

      <div className="flex-1 relative bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />

        {isLoading && (
          <div className="absolute top-4 right-4 bg-blue-500 rounded-full p-3">
            <Loader2 className="w-6 h-6 animate-spin" />
          </div>
        )}

        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-16 h-16 border-2 border-blue-400 rounded-full opacity-30"></div>
        </div>
      </div>

      <div className="bg-gray-800 p-6 border-t-4 border-blue-500">
        <div className="flex items-start gap-3">
          <Volume2 className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
          <div className="flex-1">
            <p className="text-lg font-semibold mb-2">Current Status:</p>
            <p className="text-xl leading-relaxed">{statusMessage}</p>
            {!isScanning && (
              <button
                onClick={() => {
                  void startCamera();
                }}
                className="mt-4 py-3 px-5 rounded-xl text-base font-semibold bg-blue-600 hover:bg-blue-700 transition-colors"
              >
                {needsCameraPermission ? 'Enable Camera' : 'Start Camera'}
              </button>
            )}
          </div>
        </div>
      </div>

      {isScanning && (
        <div className="bg-gray-800 p-6">
          <button
            onClick={stopScanning}
            className="w-full py-6 px-8 rounded-2xl text-2xl font-bold transition-all transform active:scale-95 flex items-center justify-center gap-4 bg-red-600 hover:bg-red-700"
          >
            <div className="w-8 h-8 bg-white rounded"></div>
            Stop Scanning
          </button>
        </div>
      )}

      <footer className="bg-gray-800 p-4 text-center text-sm text-gray-400 border-t border-gray-700">
        <div className="flex items-center justify-center gap-2">
          <AlertCircle className="w-4 h-4" />
          <p>Automatic scanning • Voice alerts every 5 seconds • On-device TensorFlow object detection</p>
        </div>
      </footer>
    </div>
  );
}
